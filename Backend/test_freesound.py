import os
import sys
import asyncio
import random
from termcolor import colored
from dotenv import load_dotenv
import platform
import subprocess
import time

# Add parent directory to path to import from Backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Backend.music_provider import MusicProvider

async def test_freesound_api(mood=None, channel_type=None):
    """Test the Freesound API integration"""
    print(colored("\n===== TESTING FREESOUND API =====", "cyan"))
    
    # Initialize music provider
    music_provider = MusicProvider()
    
    # Check if Freesound client is initialized
    if not music_provider.freesound_client:
        print(colored("❌ Freesound client not initialized. Please check your API key in the .env file.", "red"))
        print(colored("   Make sure you have set FREESOUND_API_KEY to your actual API key, not the placeholder.", "yellow"))
        return
    
    print(colored("✓ Freesound client initialized successfully", "green"))
    
    # If no mood or channel type provided, use defaults
    if not mood and not channel_type:
        moods = ["upbeat electronic", "gentle ambient", "corporate background", "fitness motivation", "cooking music"]
        mood = random.choice(moods)
        print(colored(f"No mood specified, using random mood: '{mood}'", "blue"))
    
    if channel_type:
        # Get a random mood for the channel type
        channel_moods = music_provider.music_moods.get(channel_type, ['background', 'ambient'])
        mood = random.choice(channel_moods)
        print(colored(f"Testing with channel '{channel_type}' and mood '{mood}'", "cyan"))
    else:
        print(colored(f"Testing with mood '{mood}'", "cyan"))
    
    # Test getting music from Freesound
    print(colored("\nTesting direct Freesound API search...", "cyan"))
    music_path = await music_provider.get_music_from_freesound(mood)
    
    if not music_path or not os.path.exists(music_path):
        print(colored(f"❌ Failed to get music from Freesound for '{mood}'", "red"))
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
    
    # Test getting music for a channel
    if channel_type:
        print(colored(f"\nTesting channel music retrieval for '{channel_type}'...", "cyan"))
        channel_music = await music_provider.get_music_for_channel(channel_type)
        
        if not channel_music or not os.path.exists(channel_music):
            print(colored(f"❌ Failed to get music for channel '{channel_type}'", "red"))
        else:
            file_size = os.path.getsize(channel_music) / (1024 * 1024)  # Convert to MB
            print(colored(f"✓ Channel music found: {channel_music}", "green"))
            print(colored(f"✓ File size: {file_size:.2f} MB", "green"))
    
    print(colored("\n===== FREESOUND API TEST COMPLETE =====", "cyan"))

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Check if it's a channel type
        channel_types = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
        if sys.argv[1] in channel_types:
            asyncio.run(test_freesound_api(channel_type=sys.argv[1]))
        else:
            # Assume it's a mood
            asyncio.run(test_freesound_api(mood=sys.argv[1]))
    else:
        # No arguments, run with defaults
        asyncio.run(test_freesound_api()) 