from TTS.api import TTS
import torch
import os
from pathlib import Path

class CoquiTTSGenerator:
    def __init__(self):
        # Use CUDA if available for faster processing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize with the best model for YouTube Shorts
        self.model_name = "tts_models/en/jenny/jenny"  # Natural female voice
        self.tts = TTS(model_name=self.model_name, progress_bar=True, gpu=torch.cuda.is_available())
        
        # Create output directory
        self.output_dir = Path("output/tts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_voiceover(self, text, output_filename, speaker=None):
        """
        Generate a voiceover for the given text.
        
        Args:
            text (str): The text to convert to speech
            output_filename (str): The name of the output file (without path)
            speaker (str, optional): Speaker name for multi-speaker models
        
        Returns:
            Path: Path to the generated audio file
        """
        # Clean up the text
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")

        # Prepare output path
        output_path = self.output_dir / output_filename
        
        try:
            # Generate the speech
            self.tts.tts_to_file(
                text=text,
                speaker=speaker,
                file_path=str(output_path)
            )
            print(f"Generated voiceover: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating voiceover: {str(e)}")
            raise

def test_voiceover():
    """Test the voiceover generation with a sample text."""
    generator = CoquiTTSGenerator()
    
    # Test text with different styles
    test_texts = [
        "Hey everyone! Welcome to another exciting video about artificial intelligence!",
        "Did you know that neural networks can now generate human-like speech? That's pretty amazing!",
        "Don't forget to like and subscribe for more awesome tech content!"
    ]
    
    # Generate samples for each test text
    for i, text in enumerate(test_texts, 1):
        try:
            output_file = f"test_sample_{i}.wav"
            generator.generate_voiceover(text, output_file)
            print(f"Generated test sample {i}")
        except Exception as e:
            print(f"Error generating test sample {i}: {str(e)}")

if __name__ == "__main__":
    print("Testing Coqui TTS for YouTube Shorts...")
    test_voiceover()
    print("Test complete! Check the output/tts directory for the generated samples.") 