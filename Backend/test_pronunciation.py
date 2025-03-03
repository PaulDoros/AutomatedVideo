import asyncio
from coqui_integration import CoquiTTSAPI
from termcolor import colored
import time
import os

async def test_pronunciation():
    """Test pronunciation of technical terms with improved Coqui TTS"""
    
    print(colored("\n=== Testing Improved Pronunciation of Technical Terms ===", "blue"))
    
    # Initialize the Coqui TTS API
    coqui_api = CoquiTTSAPI()
    
    # Create output directory
    os.makedirs("temp/tts/pronunciation_test", exist_ok=True)
    
    # Test texts with technical terms
    test_texts = [
        "My computer has an i9-14900KF processor with 32 CPU cores running at 3.2 GHz.",
        "I'm using an NVIDIA GeForce RTX 4080 SUPER GPU with 32GB of RAM for AI processing.",
        "The GPU acceleration makes text-to-speech much faster and higher quality.",
        "When comparing CPU vs GPU performance, the GPU is significantly faster for neural networks."
    ]
    
    # Test with different voices and emotions
    voices_emotions = [
        (None, "professional"),  # Default voice with professional emotion
        ("Callum Bungey", "neutral"),  # Male voice with neutral emotion
        ("Lilya Stainthorpe", "cheerful")  # Female voice with cheerful emotion
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        for voice, emotion in voices_emotions:
            # Skip voices if model doesn't support multiple speakers
            if voice and not coqui_api.is_multi_speaker:
                continue
                
            print(colored(f"\nGenerating voice for text {i+1} with {'default voice' if voice is None else voice} ({emotion}):", "cyan"))
            print(colored(f"Text: {text}", "cyan"))
            
            # Time the generation
            start_time = time.time()
            
            # Generate voice
            output_path = f"temp/tts/pronunciation_test/tech_terms_{i+1}_{emotion}_{int(time.time())}.wav"
            result = await coqui_api.generate_voice(
                text=text,
                speaker=voice,
                language="en",
                emotion=emotion,
                speed=1.0,
                output_path=output_path
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            if result:
                print(colored(f"✓ Successfully generated voice in {elapsed_time:.2f} seconds", "green"))
                print(colored(f"Output file: {result}", "green"))
                results.append((text, voice, emotion, result, elapsed_time))
            else:
                print(colored(f"✗ Failed to generate voice", "red"))
    
    # Print summary
    print(colored("\n=== Pronunciation Test Results ===", "blue"))
    print(colored(f"Generated {len(results)} voice samples", "green"))
    
    for text, voice, emotion, output_file, elapsed_time in results:
        print(colored(f"\nText: {text}", "cyan"))
        print(colored(f"Voice: {voice if voice else 'Default'}", "cyan"))
        print(colored(f"Emotion: {emotion}", "cyan"))
        print(colored(f"Output: {output_file}", "green"))
        print(colored(f"Time: {elapsed_time:.2f} seconds", "yellow"))
    
    print(colored("\n=== Pronunciation Test Complete ===", "blue"))
    print(colored("Please listen to the generated files to verify pronunciation quality", "yellow"))
    print(colored(f"Files are located in: {os.path.abspath('temp/tts/pronunciation_test')}", "yellow"))

if __name__ == "__main__":
    asyncio.run(test_pronunciation()) 