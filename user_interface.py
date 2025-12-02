import streamlit as st
import os
from pathlib import Path
from PIL import Image
import torch
from dialogue_system import ExoskeletonDialogueSystem
# New import for speech functionality
from speech_pipeline import SpeechPipeline

# Optional voice recorder component
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

# Set page config
st.set_page_config(
    page_title="Exoskeleton Interface",
    page_icon="🦿",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system" not in st.session_state:
    st.session_state.system = None
if "speech_pipeline" not in st.session_state:
    try:
        st.session_state.speech_pipeline = SpeechPipeline()
    except Exception as e:
        print(f"Speech pipeline initialization failed (likely missing API key): {e}")
        st.session_state.speech_pipeline = None

# Helper to capture audio input into the prompt
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

def save_baseline(value):
    with open("baseline_preference.txt", "w") as f:
        f.write(str(value))

@st.cache_resource
def load_system(model_name):
    """Load the dialogue system. Cached to avoid reloading model."""
    # Ensure baseline file exists to prevent input() prompt
    if not os.path.exists("baseline_preference.txt"):
        with open("baseline_preference.txt", "w") as f:
            f.write("5.0")  # Default backup
            
    # Use ExoskeletonDialogueSystem which implements the two-part CoT system
    # as verified in dialogue_system.py methods identify_task() and analyze_feedback()
    return ExoskeletonDialogueSystem(model_name)

def main():
    # App header with SVG logo (use Streamlit's image loader so path is resolved correctly)
    try:
        st.image("svg_logo_white.svg", width=170)
    except Exception:
        st.title("HeyExo")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        # Default to 2B model for faster inference on local machines
        model_options = [
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "gpt-5.1-2025-11-13",
            "gpt-5-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-4.1-2025-04-14",
        ]
        model_name = st.selectbox("Model Name", options=model_options, index=0)
        
        if "gpt" in model_name:
             st.caption("Note: Using OpenAI models requires OPENAI_API_KEY in .env file.")
        else:
             st.caption("Note: The 2B model is faster. Use 7B for better accuracy if you have a powerful GPU.")
        
        # Image Selection
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])
        
        # Sample images
        data_dir = Path("data/image")
        if data_dir.exists():
            sample_images = list(data_dir.glob("*"))
            if sample_images:
                selected_sample = st.selectbox(
                    "Or choose sample", 
                    ["None"] + [p.name for p in sample_images]
                )
            else:
                selected_sample = "None"
        else:
            selected_sample = "None"
            
        current_image_path = None
        if uploaded_file:
            # Save uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            current_image_path = "temp_image.jpg"
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        elif selected_sample != "None":
            current_image_path = str(data_dir / selected_sample)
            st.image(current_image_path, caption=selected_sample, use_column_width=True)

        # Baseline Preference
        st.subheader("Preferences")
        if os.path.exists("baseline_preference.txt"):
            with open("baseline_preference.txt", "r") as f:
                current_baseline = float(f.read().strip())
        else:
            current_baseline = 5.0
            
        new_baseline = st.slider("Baseline Assistance", 0.0, 20.0, current_baseline, 0.5)
        if new_baseline != current_baseline:
            save_baseline(new_baseline)
            st.success(f"Baseline updated to {new_baseline}%")
            # Reload system to pick up change if needed, or just update the attribute if possible
            if st.session_state.system:
                st.session_state.system.baseline_assistance = new_baseline

        # Clear History Button
        if st.button("Clear History & Reset"):
            if os.path.exists("previous_assistance.txt"):
                os.remove("previous_assistance.txt")
            st.session_state.messages = []
            st.rerun()

    # Main Content
    if not current_image_path:
        st.info("Please select or upload an image to begin.")
        return

    # Initialize System
    if st.session_state.system is None:
        with st.spinner("Loading model... (this may take a minute)"):
            try:
                st.session_state.system = load_system(model_name)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

    # Update image in system
    try:
        st.session_state.system.set_image(current_image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Format system response
                result = message["content"]
                if isinstance(result, dict):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Task", result.get('task', 'Unknown'))
                        st.metric("Terrain", result.get('terrain_difficulty', 'Unknown'))
                    with col2:
                        st.metric("Assistance Delta", f"{result.get('assistance_delta', 0):+.1f}%")
                        st.metric("Final Level", f"{result.get('final_assistance_level', 0)}%")
                    
                    st.markdown(f"**Reasoning:** {result.get('reasoning', '')}")
                    with st.expander("Detailed Stats"):
                        st.json(result)
                else:
                    st.write(result)

    # Chat Input Area with Microphone
    
    st.divider()
    
    # Sound Selection Section
    st.subheader("🔊 Audio Input")
    st.markdown(
        "You can:\n"
        "1. **Record** a new voice message\n"
        "2. **Upload / capture** audio from your browser\n"
        "3. **Choose** a pre-recorded sample from the list"
    )
    
    col_audio_1, col_audio_2 = st.columns([1, 1])
    
    selected_audio_path = None
    
    with col_audio_1:
        audio_value = None
        recorded_bytes = None

        # 1) Record button (if component available)
        if HAS_AUDIO_RECORDER:
            st.markdown("**(1) Record voice**")
            recorded_bytes = audio_recorder(
                text="Click to start / stop recording",
                recording_color="#e74c3c",
                neutral_color="#95a5a6",
                icon_name="microphone",
                icon_size="2x",
            )

        # 2) Native audio input (newer Streamlit) or fallback uploader
        st.markdown("**(2) Or upload / capture audio**")
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input("Record or upload voice feedback")
        else:
            audio_value = st.file_uploader("Upload voice feedback", type=["wav", "mp3", "m4a"], key="voice_upload")
    
    with col_audio_2:
        # Sample Sounds Selection
        sound_dir = Path("data/sound")
        sample_sounds = []
        if sound_dir.exists():
            sample_sounds = list(sound_dir.glob("*.mp3")) + list(sound_dir.glob("*.wav"))
        
        selected_sound_name = st.selectbox(
            "(3) Or choose a pre-recorded sample", 
            ["None"] + [p.name for p in sample_sounds]
        )
        
        if selected_sound_name != "None":
            selected_audio_path = sound_dir / selected_sound_name
            st.audio(str(selected_audio_path), format="audio/mp3")

    # Determine which audio to process
    audio_to_process = None
    
    # Priority: Live recorded > audio_input/uploaded > selected sample
    if recorded_bytes:
        audio_to_process = recorded_bytes
    elif audio_value:
        # audio_input gives a BytesIO-like object, uploader gives UploadedFile
        audio_to_process = audio_value.read()
    elif selected_audio_path:
        if st.button("Process Selected Audio"):
            with open(selected_audio_path, "rb") as f:
                audio_to_process = f.read()

    if audio_to_process:
        if st.session_state.speech_pipeline:
            with st.spinner("Transcribing voice..."):
                try:
                    # process_streamlit_audio handles bytes
                    transcription = st.session_state.speech_pipeline.process_streamlit_audio(audio_to_process)
                    if transcription:
                        st.session_state.transcribed_text = transcription
                        st.info(f"Transcribed: {transcription}")
                except Exception as e:
                    st.error(f"Speech processing error: {e}")

    # Use the transcribed text if available, or just let user type
    st.subheader("✍️ **Text Input**")
    st.markdown("**You can also type your instruction below:**")
    prompt = st.chat_input("How does the assistance feel? / what do you need?")
    
    # Determine the final prompt to use
    final_prompt = None
    if prompt:
        final_prompt = prompt
        st.session_state.transcribed_text = "" # Clear after use
    elif st.session_state.transcribed_text:
        # If we have transcribed text but no manual input yet, we can offer a button to send it
        if st.button(f"Send Voice Command: '{st.session_state.transcribed_text}'"):
            final_prompt = st.session_state.transcribed_text
            st.session_state.transcribed_text = ""

    if final_prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.write(final_prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing environment and feedback..."):
                try:
                    # Run the heavy computation
                    user_prompt = str(final_prompt)
                    result = st.session_state.system.analyze_feedback(user_prompt)
                    
                    # Update system state (save previous assistance)
                    st.session_state.system._save_previous_assistance(result['final_assistance_level'])
                    
                    # Show result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Task", result.get('task', 'Unknown'))
                        st.metric("Terrain", result.get('terrain_difficulty', 'Unknown'))
                    with col2:
                        st.metric("Assistance Delta", f"{result.get('assistance_delta', 0):+.1f}%")
                        st.metric("Final Level", f"{result.get('final_assistance_level', 0):.1f}%")
                    
                    st.markdown(f"**Reasoning:** {result.get('reasoning', 'No reasoning provided')}")
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    st.error(f"Error analyzing feedback: {e}")
                    import traceback
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
