from TTS.api import TTS
import torch

def test_tts():
    print("Loading VITS model...")
    
    # Initialize TTS with VITS
    tts = TTS(
        model_name="tts_models/en/ljspeech/vits",
        progress_bar=True,
        gpu=torch.cuda.is_available()
    )
    
    print("\nModel loaded successfully!")
    
    # Test text
    text = "Why do programmers need coffee? It's simple: caffeine is the only thing that debugs us!"
    
    print("\nGenerating speech...")
    # Generate speech
    try:
        tts.tts_to_file(
            text=text,
            file_path="test_output_vits.wav",
            speed=1.0
        )
        print("\nSpeech generated successfully! Check test_output_vits.wav")
    except Exception as e:
        print(f"\nError generating speech: {str(e)}")

if __name__ == "__main__":
    test_tts() 