import os
from openai import OpenAI
import tempfile
from pathlib import Path

class SpeechPipeline:
    def __init__(self, api_key=None):
        """Initialize the speech pipeline with OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for speech pipeline. Please set it in environment or pass to constructor.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.supported_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm']

    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file using OpenAI Whisper API.
        
        Args:
            audio_file_path (str or Path): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        # Ensure file extension is supported
        file_ext = Path(audio_file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            pass

        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            return transcription
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise e

    def process_streamlit_audio(self, audio_bytes):
        """
        Process audio bytes from Streamlit audio recorder.
        
        Args:
            audio_bytes (bytes): Raw audio bytes
            
        Returns:
            str: Transcribed text
        """
        if not audio_bytes:
            return None
            
        # Save bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name
            
        try:
            text = self.transcribe_audio(temp_path)
            return text
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

