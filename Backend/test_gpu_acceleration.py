import asyncio
from coqui_integration import CoquiTTSAPI
from termcolor import colored
import time

async def test_gpu_acceleration():
    """Test GPU acceleration and emoji/tag handling in Coqui TTS"""
    
    print(colored("\n=== Testing GPU Acceleration and Text Cleaning ===", "blue"))
    
    # Initialize the Coqui TTS API
    coqui_api = CoquiTTSAPI()
    
    # Test text with emojis and emotion tags
    test_text = "[friendly] This is a test of GPU acceleration and emoji handling 🚀 with Coqui TTS 🎵. Let's see if it works faster! 🔥"
    
    # Time the generation
    start_time = time.time()
    
    # Generate voice - don't specify speaker if not supported
    result = await coqui_api.generate_voice(
        text=test_text,
        language="en",
        emotion="cheerful",
        speed=1.0
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if result:
        print(colored(f"✓ Successfully generated voice in {elapsed_time:.2f} seconds", "green"))
        print(colored(f"Output file: {result}", "green"))
    else:
        print(colored(f"✗ Failed to generate voice", "red"))
    
    print(colored("\n=== GPU Acceleration Test Complete ===", "blue"))

if __name__ == "__main__":
    asyncio.run(test_gpu_acceleration()) 