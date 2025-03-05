import os
import sys
import asyncio
import requests
from termcolor import colored
from dotenv import load_dotenv

# Add parent directory to path to import from Backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Backend.music_provider import MusicProvider

# Load environment variables
load_dotenv()

# Create directories if they don't exist
os.makedirs("assets/music/default", exist_ok=True)
for channel in ["tech_humor", "ai_money", "baby_tips", "quick_meals", "fitness_motivation"]:
    os.makedirs(f"assets/music/{channel}", exist_ok=True)

# List of royalty-free music URLs from SoundHelix (these are publicly available and free to use)
DEFAULT_MUSIC_URLS = {
    "default": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    ],
    "tech_humor": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
    ],
    "ai_money": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
    ],
    "baby_tips": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
    ],
    "quick_meals": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
    ],
    "fitness_motivation": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-12.mp3",
    ]
}

def download_music(url, output_path):
    """Download music from URL to output path"""
    try:
        print(colored(f"Downloading music from {url}", "blue"))
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(colored(f"✓ Downloaded music: {os.path.basename(output_path)} ({file_size:.2f} MB)", "green"))
        return True
    except Exception as e:
        print(colored(f"Error downloading music: {str(e)}", "red"))
        return False

def main():
    """Download default music files"""
    print(colored("=== Downloading Default Music ===", "blue"))
    
    # Download default music
    for category, urls in DEFAULT_MUSIC_URLS.items():
        print(colored(f"\nDownloading music for {category}...", "blue"))
        
        for i, url in enumerate(urls):
            output_path = f"assets/music/{category}/{category}_default_{i+1}.mp3"
            download_music(url, output_path)
    
    print(colored("\n=== Default Music Download Complete ===", "green"))

if __name__ == "__main__":
    main() 