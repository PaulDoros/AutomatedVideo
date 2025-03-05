#!/usr/bin/env python
import os
import sys
import asyncio
import random
from termcolor import colored
from dotenv import load_dotenv
from music_provider import MusicProvider

async def test_freesound_music(channel_type=None, mood=None):
    """Test the improved Freesound music integration"""
    print(colored("=== Testing Improved Freesound Music Integration ===", "cyan"))
    
    # Initialize music provider
    music_provider = MusicProvider()
    
    if not channel_type and not mood:
        print(colored("Please provide either a channel type or a specific mood", "red"))
        print(colored("Usage: python test_improved_freesound.py [channel_type] [mood]", "yellow"))
        return
    
    # If channel type is provided, get a random mood for that channel
    if channel_type:
        if channel_type not in music_provider.music_moods:
            print(colored(f"Invalid channel type: {channel_type}", "red"))
            print(colored(f"Available channel types: {list(music_provider.music_moods.keys())}", "yellow"))
            return
        
        print(colored(f"Testing with channel: {channel_type}", "blue"))
        
        # Get a random mood for this channel
        if not mood:
            mood = random.choice(music_provider.music_moods[channel_type])
        
        print(colored(f"Using mood: {mood}", "blue"))
        
        # Test direct Freesound search
        print(colored("\n1. Testing direct Freesound search:", "magenta"))
        music_path = await music_provider.get_music_from_freesound(mood)
        
        if music_path:
            print(colored(f"✓ Successfully downloaded music: {music_path}", "green"))
            file_size = os.path.getsize(music_path) / (1024 * 1024)  # MB
            print(colored(f"  File size: {file_size:.2f} MB", "green"))
            
            # Try to play the music
            try:
                if sys.platform == "win32":
                    os.system(f'start {music_path}')
                elif sys.platform == "darwin":
                    os.system(f'open {music_path}')
                else:
                    os.system(f'xdg-open {music_path}')
                print(colored("  Playing music for 5 seconds...", "blue"))
                await asyncio.sleep(5)
            except Exception as e:
                print(colored(f"  Error playing music: {str(e)}", "red"))
        else:
            print(colored("✗ Failed to download music from Freesound", "red"))
        
        # Test channel music retrieval
        print(colored("\n2. Testing channel music retrieval:", "magenta"))
        channel_music = await music_provider.get_music_for_channel(channel_type)
        
        if channel_music:
            print(colored(f"✓ Successfully retrieved channel music: {channel_music}", "green"))
            file_size = os.path.getsize(channel_music) / (1024 * 1024)  # MB
            print(colored(f"  File size: {file_size:.2f} MB", "green"))
        else:
            print(colored("✗ Failed to retrieve channel music", "red"))
    
    # If only mood is provided, test direct Freesound search
    elif mood:
        print(colored(f"Testing with mood: {mood}", "blue"))
        
        # Test direct Freesound search
        print(colored("\nTesting direct Freesound search:", "magenta"))
        music_path = await music_provider.get_music_from_freesound(mood)
        
        if music_path:
            print(colored(f"✓ Successfully downloaded music: {music_path}", "green"))
            file_size = os.path.getsize(music_path) / (1024 * 1024)  # MB
            print(colored(f"  File size: {file_size:.2f} MB", "green"))
            
            # Try to play the music
            try:
                if sys.platform == "win32":
                    os.system(f'start {music_path}')
                elif sys.platform == "darwin":
                    os.system(f'open {music_path}')
                else:
                    os.system(f'xdg-open {music_path}')
                print(colored("  Playing music for 5 seconds...", "blue"))
                await asyncio.sleep(5)
            except Exception as e:
                print(colored(f"  Error playing music: {str(e)}", "red"))
        else:
            print(colored("✗ Failed to download music from Freesound", "red"))

async def main():
    # Load environment variables
    load_dotenv()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        channel_type = sys.argv[1] if sys.argv[1] in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation'] else None
        mood = sys.argv[1] if not channel_type else (sys.argv[2] if len(sys.argv) > 2 else None)
        
        await test_freesound_music(channel_type, mood)
    else:
        print(colored("Please provide a channel type or mood to test", "yellow"))
        print(colored("Usage: python test_improved_freesound.py [channel_type] [mood]", "yellow"))
        print(colored("Available channel types: tech_humor, ai_money, baby_tips, quick_meals, fitness_motivation", "yellow"))
        print(colored("Example moods: 'upbeat electronic', 'gentle lullaby', etc.", "yellow"))

if __name__ == "__main__":
    asyncio.run(main()) 