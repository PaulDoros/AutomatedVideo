import asyncio
import os
from coqui_integration import CoquiTTSAPI

# Selected voices (3 male, 3 female)
MALE_VOICES = [
    "Andrew Chipper",    # Male voice with energetic tone
    "Damien Black",      # Male voice with deeper tone
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

async def test_voice(api, voice, text, emotion):
    print(f"\n=== Testing {voice} with {emotion} emotion ===")
    output_path = await api.generate_voice(
        text=text,
        speaker=voice,
        language="en",
        emotion=emotion,
        speed=1.0
    )
    
    if output_path:
        print(f"✓ Audio generated: {output_path}")
        # Get absolute path for easier finding
        abs_path = os.path.abspath(output_path)
        print(f"Absolute path: {abs_path}")
    else:
        print("✗ Failed to generate audio")

async def main():
    api = CoquiTTSAPI()
    
    print("\n=== MALE VOICES WITH JOKE ===")
    for voice in MALE_VOICES:
        await test_voice(api, voice, JOKE_TEXT, "cheerful")
    
    print("\n=== FEMALE VOICES WITH JOKE ===")
    for voice in FEMALE_VOICES:
        await test_voice(api, voice, JOKE_TEXT, "cheerful")
    
    print("\n=== MALE VOICES WITH SERIOUS CONTENT ===")
    for voice in MALE_VOICES:
        await test_voice(api, voice, SERIOUS_TEXT, "professional")
    
    print("\n=== FEMALE VOICES WITH SERIOUS CONTENT ===")
    for voice in FEMALE_VOICES:
        await test_voice(api, voice, SERIOUS_TEXT, "professional")
    
    # Test one voice with different emotions
    print("\n=== TESTING DIFFERENT EMOTIONS ===")
    for emotion in EMOTIONS:
        await test_voice(api, "Daisy Studious", SERIOUS_TEXT, emotion)
    
    print("\n✓ Voice test complete!")

if __name__ == "__main__":
    asyncio.run(main()) 