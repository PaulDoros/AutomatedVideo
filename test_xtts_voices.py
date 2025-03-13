from TTS.api import TTS
import torch
from termcolor import colored
import os

def list_english_speakers():
    print(colored("=== Testing XTTS-v2 English Speakers ===", "cyan"))
    
    try:
        # Initialize XTTS-v2 model
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(colored(f"\nLoading model: {model_name}", "yellow"))
        
        # Initialize TTS with XTTS-v2
        tts = TTS(model_name=model_name)
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(colored(f"Using device: {device}", "green"))
        
        # English speakers - these are known to work well with English
        english_speakers = [
            "Aaron Dreschner",
            "Andrew Chipper",
            "Damien Black",
            "Craig Gutsy",
            "Viktor Eka",
            "Wulf Carlevaro",
            "Daisy Studious",
            "Gracie Wise",
            "Sofia Hellen",
            "Tammy Grit",
            "Alexandra Hisakawa",
            "Brenda Stern"
        ]
        
        print(colored("\nPrimary English Speakers:", "cyan"))
        for speaker in sorted(english_speakers):
            print(f"- {speaker}")
            
        # Test sample for first speaker
        test_text = "Hello! This is a test of the English text-to-speech system."
        print(colored("\nGenerating test audio for first speaker...", "yellow"))
        
        os.makedirs("temp/tts/english_test", exist_ok=True)
        output_file = f"temp/tts/english_test/english_test_{english_speakers[0]}.wav"
        
        try:
            tts.tts_to_file(
                text=test_text,
                speaker=english_speakers[0],
                language="en",
                file_path=output_file
            )
            print(colored(f"✓ Test audio generated: {output_file}", "green"))
        except Exception as e:
            print(colored(f"Error generating test audio: {str(e)}", "red"))
                
    except Exception as e:
        print(colored(f"Error loading XTTS-v2 model: {str(e)}", "red"))

if __name__ == "__main__":
    list_english_speakers() 