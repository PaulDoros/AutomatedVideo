import torch
from TTS.api import TTS
import os

def test_coqui_voices():
    """Test different Coqui TTS voices and save samples."""
    
    # Create output directory if it doesn't exist
    os.makedirs("test_voices", exist_ok=True)
    
    # Initialize TTS with GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test text for comparison
    test_text = "Hey everyone! Welcome to another exciting YouTube Shorts video. Today we're going to talk about something amazing!"
    
    # List of models to test
    models_to_test = [
        "tts_models/en/vctk/vits",  # Multi-speaker, natural sounding
        "tts_models/en/jenny/jenny",  # Single female speaker, very natural
        "tts_models/en/multi-dataset/tortoise-v2",  # High quality but slower
        "tts_models/en/ljspeech/tacotron2-DDC",  # Classic TTS model
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            
            # Initialize TTS with the current model
            tts = TTS(model_name=model_name, progress_bar=True, gpu=torch.cuda.is_available())
            
            # Get available speakers for this model
            speakers = tts.speakers if hasattr(tts, "speakers") else [None]
            
            for speaker in speakers[:3]:  # Test up to 3 speakers per model
                try:
                    # Create filename based on model and speaker
                    speaker_str = f"_{speaker}" if speaker else ""
                    filename = f"test_voices/sample_{model_name.split('/')[-1]}{speaker_str}.wav"
                    
                    # Generate speech
                    print(f"Generating speech for {speaker if speaker else 'default'}")
                    tts.tts_to_file(
                        text=test_text,
                        speaker=speaker,
                        file_path=filename
                    )
                    print(f"Saved to: {filename}")
                    
                except Exception as e:
                    print(f"Error with speaker {speaker}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    print("Starting Coqui TTS voice test...")
    test_coqui_voices()
    print("\nTest complete! Check the 'test_voices' directory for the generated samples.") 