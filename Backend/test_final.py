import asyncio
from voice_diversification import VoiceDiversification
from termcolor import colored
import time

async def test_final():
    """Test the voice diversification with the fast model"""
    
    print(colored("\n=== Testing Voice Diversification with Fast Model ===", "blue"))
    
    # Initialize the voice diversification system
    diversifier = VoiceDiversification()
    
    # Test text
    test_text = "This is a test of the voice diversification system with the fast model. We want to make sure it works correctly with the video generator."
    
    # Test with different emotions
    emotions = ["cheerful", "professional", "friendly"]
    
    for emotion in emotions:
        print(colored(f"\nTesting with emotion: {emotion}", "cyan"))
        
        # Time the generation
        start_time = time.time()
        
        # Generate a random voiceover with the specified emotion
        result = await diversifier.generate_random_voiceover(
            text=test_text,
            gender=None,  # Random gender
            emotion=emotion
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if result:
            print(colored(f"✓ Successfully generated voiceover with {emotion} emotion in {elapsed_time:.2f} seconds", "green"))
            print(colored(f"Output file: {result}", "green"))
        else:
            print(colored(f"✗ Failed to generate voiceover with {emotion} emotion", "red"))
    
    print(colored("\n=== Voice Diversification Test Complete ===", "blue"))

if __name__ == "__main__":
    asyncio.run(test_final()) 