import asyncio
import os
import time
from coqui_integration import CoquiTTSAPI

# Selected voices (3 male, 3 female)
MALE_VOICES = [
    "Viktor Menelaos",   # Male voice with deeper tone
    "Damien Black",      # Male voice with authoritative tone
    "Luis Moray"         # Male voice with accent
]

FEMALE_VOICES = [
    "Daisy Studious",    # Female voice with clear articulation
    "Sofia Hellen",      # Female voice with warm tone
    "Alma María"         # Female voice with accent
]

# Test texts
JOKE_TEXT = "Why don't scientists trust atoms? Because they make up everything! And speaking of making things up, my code works on the first try."

SERIOUS_TEXT = "Artificial intelligence represents one of the most significant technological advancements of our time. It has the potential to solve complex problems across healthcare, climate science, and education."

# Emotions to test
EMOTIONS = ["cheerful", "professional", "friendly", "serious"]

async def test_voice(api, voice, text, emotion, content_type=""):
    print(f"\n=== Testing {voice} with {emotion} emotion ===")
    
    # Create a unique filename based on voice, emotion, and content type
    safe_voice = voice.replace(" ", "_").replace(".", "").lower()
    filename = f"{safe_voice}_{emotion}_{content_type}_{int(time.time())}.wav"
    output_dir = "temp/tts/voice_samples"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    
    # Generate the speech
    result = await api.generate_voice(
        text=text,
        speaker=voice,
        language="en",
        emotion=emotion,
        speed=1.0,
        output_path=output_path  # Use our custom path
    )
    
    if result:
        print(f"✓ Audio generated: {output_path}")
        # Get absolute path for easier finding
        abs_path = os.path.abspath(output_path)
        print(f"Absolute path: {abs_path}")
        return abs_path
    else:
        print("✗ Failed to generate audio")
        return None

async def main():
    api = CoquiTTSAPI()
    
    # Create a list to store all generated files
    generated_files = []
    
    print("\n=== MALE VOICES WITH JOKE ===")
    for voice in MALE_VOICES:
        path = await test_voice(api, voice, JOKE_TEXT, "cheerful", "joke")
        if path:
            generated_files.append(path)
    
    print("\n=== FEMALE VOICES WITH JOKE ===")
    for voice in FEMALE_VOICES:
        path = await test_voice(api, voice, JOKE_TEXT, "cheerful", "joke")
        if path:
            generated_files.append(path)
    
    print("\n=== MALE VOICES WITH SERIOUS CONTENT ===")
    for voice in MALE_VOICES:
        path = await test_voice(api, voice, SERIOUS_TEXT, "professional", "serious")
        if path:
            generated_files.append(path)
    
    print("\n=== FEMALE VOICES WITH SERIOUS CONTENT ===")
    for voice in FEMALE_VOICES:
        path = await test_voice(api, voice, SERIOUS_TEXT, "professional", "serious")
        if path:
            generated_files.append(path)
    
    # Test one voice with different emotions
    print("\n=== TESTING DIFFERENT EMOTIONS ===")
    for emotion in EMOTIONS:
        path = await test_voice(api, "Daisy Studious", SERIOUS_TEXT, emotion, "emotion_test")
        if path:
            generated_files.append(path)
    
    print("\n✓ Voice test complete!")
    print(f"\nGenerated {len(generated_files)} unique audio files:")
    for i, file in enumerate(generated_files, 1):
        print(f"{i}. {file}")

if __name__ == "__main__":
    asyncio.run(main()) 