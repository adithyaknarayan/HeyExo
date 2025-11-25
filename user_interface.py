import streamlit as st
import os
from pathlib import Path
from PIL import Image
import torch
from dialogue_system import ExoskeletonDialogueSystem

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
    st.title("🦿 Exoskeleton Dialogue Interface")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        # Default to 2B model for faster inference on local machines
        model_options = [
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
        ]
        model_name = st.selectbox("Model Name", options=model_options, index=0)
        st.caption("Note: The 2B model is faster. Use 7B for better accuracy if you have a powerful GPU.")
        
        # Image Selection
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])
        
        # Sample images
        data_dir = Path("data")
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

    # Chat Input
    if prompt := st.chat_input("How does the assistance feel? / what do you need?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing environment and feedback..."):
                try:
                    # Run the heavy computation in a way that doesn't block UI updates if possible, 
                    # but here we just need to make sure we catch everything.
                    # Explicitly convert prompt to string just in case
                    user_prompt = str(prompt)
                    
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

