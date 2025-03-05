import asyncio
import os
from music_provider import music_provider
from termcolor import colored
from dotenv import load_dotenv

async def test_music_provider():
    """Test the music provider functionality"""
    print(colored("\n=== Testing Music Provider ===", "blue"))
    
    # Test getting music for each channel type
    channel_types = [
        'tech_humor',
        'ai_money',
        'baby_tips',
        'quick_meals',
        'fitness_motivation'
    ]
    
    for channel_type in channel_types:
        print(colored(f"\nTesting music for {channel_type}:", "cyan"))
        
        # Get music for the channel
        music_path = await music_provider.get_music_for_channel(channel_type)
        
        if music_path and os.path.exists(music_path):
            print(colored(f"✓ Successfully got music: {music_path}", "green"))
            
            # Get file size
            file_size = os.path.getsize(music_path) / 1024  # KB
            print(colored(f"  File size: {file_size:.2f} KB", "green"))
            
            # Check if it's a valid audio file
            if music_path.endswith('.mp3'):
                print(colored(f"  Valid MP3 file", "green"))
            else:
                print(colored(f"  Warning: Not an MP3 file", "yellow"))
        else:
            print(colored(f"✗ Failed to get music for {channel_type}", "red"))
    
    print(colored("\n=== Music Provider Test Complete ===", "blue"))

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(test_music_provider()) 