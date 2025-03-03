import asyncio
from voice_diversification import VoiceDiversification
from termcolor import colored

async def test_voice_diversification_integration():
    """Test the voice diversification integration with a simple example"""
    
    print(colored("\n=== Testing Voice Diversification Integration ===", "blue"))
    
    # Initialize the voice diversification system
    diversifier = VoiceDiversification()
    
    # Test text
    test_text = "This is a test of the voice diversification system integration. We want to make sure it works correctly with the video generator."
    
    # Test with different emotions
    emotions = ["cheerful", "professional", "friendly"]
    
    for emotion in emotions:
        print(colored(f"\nTesting with emotion: {emotion}", "cyan"))
        
        # Generate a random voiceover with the specified emotion
        result = await diversifier.generate_random_voiceover(
            text=test_text,
            gender=None,  # Random gender
            emotion=emotion
        )
        
        if result:
            print(colored(f"✓ Successfully generated voiceover with {emotion} emotion", "green"))
            print(colored(f"Output file: {result}", "green"))
        else:
            print(colored(f"✗ Failed to generate voiceover with {emotion} emotion", "red"))
    
    print(colored("\n=== Voice Diversification Integration Test Complete ===", "blue"))

if __name__ == "__main__":
    asyncio.run(test_voice_diversification_integration()) 