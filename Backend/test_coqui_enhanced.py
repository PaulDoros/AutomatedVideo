import asyncio
from coqui_integration import CoquiTTSAPI
from termcolor import colored
import os

async def test_coqui_enhanced():
    """Test enhanced Coqui TTS implementation with XTTS-v2"""
    try:
        print(colored("\n=== Testing Enhanced Coqui TTS Integration ===", "blue"))
        
        # Initialize Coqui TTS
        coqui = CoquiTTSAPI()
        
        # Test text with different emotions and content types
        test_cases = [
            {
                "text": "Hey everyone! Welcome to another exciting tech video. Today we're going to explore something amazing!",
                "content_type": "tech_humor",
                "emotion": "cheerful"
            },
            {
                "text": "Here's a professional coding tip that will make your development workflow much more efficient.",
                "content_type": "coding_tips",
                "emotion": "professional"
            },
            {
                "text": "Let me share a life-changing productivity hack that saved me hours of work!",
                "content_type": "life_hack",
                "emotion": "friendly"
            }
        ]
        
        # Create test output directory
        os.makedirs("temp/tts/test", exist_ok=True)
        
        # Test each case
        for i, case in enumerate(test_cases, 1):
            print(colored(f"\nTest Case {i}: {case['content_type']}", "cyan"))
            print(colored(f"Text: {case['text']}", "cyan"))
            print(colored(f"Emotion: {case['emotion']}", "cyan"))
            
            # Generate voice
            audio_path = await coqui.generate_voice(
                text=case['text'],
                emotion=case['emotion']
            )
            
            if audio_path:
                print(colored(f"✓ Generated audio: {audio_path}", "green"))
            else:
                print(colored("✗ Failed to generate audio", "red"))
        
        # Test voice cloning if sample available
        sample_path = "assets/tts/samples/reference_voice.wav"
        if os.path.exists(sample_path):
            print(colored("\nTesting voice cloning...", "blue"))
            cloned_audio = await coqui.clone_voice(
                reference_audio=sample_path,
                text="This is a test of voice cloning using XTTS-v2. The voice should sound similar to the reference audio."
            )
            
            if cloned_audio:
                print(colored(f"✓ Generated cloned voice: {cloned_audio}", "green"))
            else:
                print(colored("✗ Failed to clone voice", "red"))
        
        # Get available voices
        print(colored("\nGetting available voices...", "blue"))
        voices = coqui.get_available_voices()
        for model, speakers in voices.items():
            print(colored(f"\nModel: {model}", "cyan"))
            print(colored(f"Available speakers: {len(speakers)}", "green"))
            if len(speakers) > 0:
                print(colored(f"Sample speakers: {speakers[:3]}", "cyan"))
        
        # Get supported languages
        print(colored("\nGetting supported languages...", "blue"))
        languages = coqui.get_supported_languages()
        print(colored(f"Supported languages: {languages}", "cyan"))
        
        print(colored("\n✓ Enhanced Coqui TTS test complete!", "green"))
        
    except Exception as e:
        print(colored(f"\n✗ Error during test: {str(e)}", "red"))

if __name__ == "__main__":
    asyncio.run(test_coqui_enhanced()) 