import asyncio
import os
import sys
import random
from termcolor import colored
from dotenv import load_dotenv
import platform
import subprocess
import time

# Add parent directory to path to import from Backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Backend.music_provider import MusicProvider

async def test_music_provider(channel_type=None):
    """Test the music provider functionality"""
    print(colored("\n===== TESTING MUSIC PROVIDER =====", "cyan"))
    
    if not channel_type:
        print(colored("Error: Please provide a channel type", "red"))
        print("Usage: python test_music_provider.py [channel_type]")
        print("Available channel types: tech_humor, ai_money, baby_tips, quick_meals, fitness_motivation")
        return
    
    # Initialize music provider
    music_provider = MusicProvider()
    
    print(colored(f"Testing music for channel: {channel_type}", "cyan"))
    
    # Get music for the channel
    music_path = await music_provider.get_music_for_channel(channel_type)
    
    if not music_path or not os.path.exists(music_path):
        print(colored(f"❌ Failed to get music for {channel_type}", "red"))
        return
    
    # Print music details
    file_size = os.path.getsize(music_path) / (1024 * 1024)  # Convert to MB
    print(colored(f"✓ Music found: {music_path}", "green"))
    print(colored(f"✓ File size: {file_size:.2f} MB", "green"))
    
    # Try to play the music
    try:
        print(colored(f"Playing music for 5 seconds...", "yellow"))
        
        # Different play commands based on OS
        if platform.system() == "Windows":
            subprocess.Popen(["start", music_path], shell=True)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["afplay", music_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", music_path])
        
        # Play for 5 seconds then stop
        time.sleep(5)
        
        print(colored("✓ Music playback completed", "green"))
    except Exception as e:
        print(colored(f"❌ Error playing music: {str(e)}", "red"))

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print(colored("Error: Please provide a channel type", "red"))
        print("Usage: python test_music_provider.py [channel_type]")
        print("Available channel types: tech_humor, ai_money, baby_tips, quick_meals, fitness_motivation")
        sys.exit(1)
    
    channel_type = sys.argv[1]
    
    # Run the test
    asyncio.run(test_music_provider(channel_type)) 