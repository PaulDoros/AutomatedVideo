import asyncio
from coqui_integration import CoquiTTSAPI
from termcolor import colored

async def test_simple_voice_generation():
    """Test basic voice generation with Coqui TTS"""
    
    print(colored("\n=== Testing Basic Voice Generation ===", "blue"))
    
    # Initialize the Coqui TTS API
    coqui_api = CoquiTTSAPI()
    
    # Test text
    test_text = "This is a simple test of the Coqui TTS voice generation."
    
    # Get available voices
    if hasattr(coqui_api, 'speakers') and coqui_api.speakers:
        print(colored(f"Available voices: {len(coqui_api.speakers)}", "green"))
        if len(coqui_api.speakers) > 0:
            test_voice = coqui_api.speakers[0]
            print(colored(f"Testing with voice: {test_voice}", "cyan"))
            
            # Generate voice
            result = await coqui_api.generate_voice(
                text=test_text,
                speaker=test_voice,
                language="en",
                emotion="neutral",
                speed=1.0
            )
            
            if result:
                print(colored(f"✓ Successfully generated voice", "green"))
                print(colored(f"Output file: {result}", "green"))
            else:
                print(colored(f"✗ Failed to generate voice", "red"))
        else:
            print(colored("No voices available", "red"))
    else:
        print(colored("No speakers attribute or it's empty", "red"))
    
    print(colored("\n=== Basic Voice Generation Test Complete ===", "blue"))

if __name__ == "__main__":
    asyncio.run(test_simple_voice_generation()) 