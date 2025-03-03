import asyncio
from coqui_integration import CoquiTTSAPI
from termcolor import colored
import time
import os
import random

async def test_joke_script():
    """Test how the improved TTS system reads a programmer joke script with emojis"""
    
    print(colored("\n=== Testing Programmer Joke Script Reading ===", "blue"))
    
    # Initialize the Coqui TTS API
    coqui_api = CoquiTTSAPI()
    
    # Create output directory
    os.makedirs("temp/tts/joke_script", exist_ok=True)
    
    # The joke script with emojis
    joke_script = """Why do programmers always talk to their coffee? ☕
Because it understands Java better than anyone else! 💻
Even more than their rubber ducks... 🦆
Turns out, coffee's the real debugging hero! 😂
Hit like for more caffeine-driven code!"""

    print(colored("\nOriginal Script:", "cyan"))
    print(colored(joke_script, "yellow"))
    
    # Test with different voices and emotions if available
    emotions = ["cheerful", "friendly", "professional"]
    
    # If we have speaker support, use some specific voices
    if coqui_api.is_multi_speaker and coqui_api.speakers:
        # Try to find some good voices for jokes
        potential_voices = ["Callum Bungey", "Craig Gutsy", "Lilya Stainthorpe", "Annmarie Nele"]
        available_voices = [voice for voice in potential_voices if voice in coqui_api.speakers]
        
        # If none of our preferred voices are available, pick random ones
        if not available_voices and coqui_api.speakers:
            available_voices = random.sample(coqui_api.speakers, min(3, len(coqui_api.speakers)))
    else:
        available_voices = [None]  # Just use default voice
    
    results = []
    
    for voice in available_voices:
        for emotion in emotions:
            # Skip some combinations to keep the test shorter
            if voice is None and emotion != "cheerful":
                continue
                
            voice_name = voice if voice else "Default"
            print(colored(f"\nGenerating voice for joke script with {voice_name} ({emotion}):", "cyan"))
            
            # Time the generation
            start_time = time.time()
            
            # Generate voice
            output_path = f"temp/tts/joke_script/joke_{voice_name.replace(' ', '_')}_{emotion}_{int(time.time())}.wav"
            result = await coqui_api.generate_voice(
                text=joke_script,
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
                results.append((voice_name, emotion, result, elapsed_time))
            else:
                print(colored(f"✗ Failed to generate voice", "red"))
    
    # Print summary
    print(colored("\n=== Joke Script Test Results ===", "blue"))
    print(colored(f"Generated {len(results)} voice samples", "green"))
    
    for voice, emotion, output_file, elapsed_time in results:
        print(colored(f"\nVoice: {voice}", "cyan"))
        print(colored(f"Emotion: {emotion}", "cyan"))
        print(colored(f"Output: {output_file}", "green"))
        print(colored(f"Time: {elapsed_time:.2f} seconds", "yellow"))
    
    print(colored("\n=== Joke Script Test Complete ===", "blue"))
    print(colored("Please listen to the generated files to hear how the system reads the joke script", "yellow"))
    print(colored(f"Files are located in: {os.path.abspath('temp/tts/joke_script')}", "yellow"))

if __name__ == "__main__":
    asyncio.run(test_joke_script()) 