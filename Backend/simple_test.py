import asyncio
from coqui_integration import CoquiTTSAPI

async def main():
    api = CoquiTTSAPI()
    print("API initialized")
    print(f"Available speakers: {len(api.speakers)}")
    print(f"First 5 speakers: {api.speakers[:5]}")
    
    # Test one voice
    voice = "Andrew Chipper"
    text = "This is a test of the Coqui TTS system with the Andrew Chipper voice."
    
    print(f"\nTesting voice: {voice}")
    output_path = await api.generate_voice(
        text=text,
        speaker=voice,
        language="en",
        emotion="cheerful",
        speed=1.0
    )
    
    if output_path:
        print(f"✓ Audio generated: {output_path}")
    else:
        print("✗ Failed to generate audio")

if __name__ == "__main__":
    asyncio.run(main()) 