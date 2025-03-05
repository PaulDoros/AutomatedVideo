import os
import asyncio
import random
import json
from termcolor import colored
from dotenv import load_dotenv
from music_provider import MusicProvider

# Load environment variables
load_dotenv()

# Music themes and keywords for different video types
MUSIC_THEMES = {
    'tech_humor': {
        'themes': ['tech', 'comedy', 'funny', 'quirky', 'upbeat'],
        'keywords': ['technology', 'computer', 'software', 'gadget', 'digital', 'innovation', 
                    'humor', 'comedy', 'funny', 'joke', 'laugh', 'quirky', 'playful']
    },
    'ai_money': {
        'themes': ['business', 'technology', 'professional', 'corporate', 'modern'],
        'keywords': ['artificial intelligence', 'machine learning', 'income', 'money', 'finance', 
                    'investment', 'success', 'growth', 'business', 'professional']
    },
    'baby_tips': {
        'themes': ['gentle', 'soothing', 'calm', 'peaceful', 'lullaby'],
        'keywords': ['baby', 'child', 'infant', 'parenting', 'family', 'care', 'gentle', 
                    'soothing', 'calm', 'peaceful', 'lullaby', 'soft']
    },
    'quick_meals': {
        'themes': ['upbeat', 'cheerful', 'energetic', 'cooking', 'kitchen'],
        'keywords': ['food', 'cooking', 'recipe', 'kitchen', 'meal', 'quick', 'easy', 
                    'delicious', 'tasty', 'preparation', 'chef', 'culinary']
    },
    'fitness_motivation': {
        'themes': ['energetic', 'motivational', 'workout', 'powerful', 'dynamic'],
        'keywords': ['fitness', 'workout', 'exercise', 'gym', 'training', 'health', 
                    'strength', 'motivation', 'energy', 'power', 'dynamic', 'active']
    }
}

# Number of music files to download per theme
MUSIC_PER_THEME = 5

async def download_music_library():
    """Download a variety of music for different video themes"""
    print(colored("=== Downloading Music Library ===", "cyan"))
    
    # Initialize music provider
    music_provider = MusicProvider()
    
    # Track successful downloads
    successful_downloads = {}
    
    # Process each theme
    for theme, data in MUSIC_THEMES.items():
        print(colored(f"\n=== Downloading music for {theme} ===", "cyan"))
        successful_downloads[theme] = []
        
        # Create directory if it doesn't exist
        theme_dir = os.path.join("Assets", "music", theme)
        os.makedirs(theme_dir, exist_ok=True)
        
        # Count existing files
        existing_files = [f for f in os.listdir(theme_dir) if f.endswith('.mp3')]
        print(colored(f"Found {len(existing_files)} existing music files for {theme}", "blue"))
        
        # Calculate how many more to download
        to_download = max(0, MUSIC_PER_THEME - len(existing_files))
        
        if to_download == 0:
            print(colored(f"✓ Already have enough music for {theme}", "green"))
            continue
        
        print(colored(f"Downloading {to_download} new music files for {theme}...", "blue"))
        
        # Download music files
        for i in range(to_download):
            # Select a random theme and keywords
            selected_theme = random.choice(data['themes'])
            selected_keywords = random.sample(data['keywords'], min(5, len(data['keywords'])))
            
            print(colored(f"Download {i+1}/{to_download}: Theme: {selected_theme}, Keywords: {', '.join(selected_keywords)}", "blue"))
            
            # Download music
            try:
                music_path = await music_provider.download_music_for_video(selected_theme, selected_keywords)
                
                if music_path and os.path.exists(music_path):
                    # Copy to theme directory with a descriptive name
                    new_filename = f"{theme}_music_{i+1}.mp3"
                    new_path = os.path.join(theme_dir, new_filename)
                    
                    # Copy file
                    with open(music_path, 'rb') as src, open(new_path, 'wb') as dst:
                        dst.write(src.read())
                    
                    print(colored(f"✓ Downloaded and saved music: {new_filename}", "green"))
                    successful_downloads[theme].append(new_path)
                else:
                    print(colored(f"✗ Failed to download music for {theme} (attempt {i+1})", "red"))
            except Exception as e:
                print(colored(f"✗ Error downloading music: {str(e)}", "red"))
        
        # Summary for this theme
        print(colored(f"Downloaded {len(successful_downloads[theme])}/{to_download} music files for {theme}", "cyan"))
    
    # Overall summary
    total_downloaded = sum(len(files) for files in successful_downloads.values())
    print(colored(f"\n=== Music Library Download Summary ===", "cyan"))
    print(colored(f"Total music files downloaded: {total_downloaded}", "green"))
    for theme, files in successful_downloads.items():
        print(colored(f"  {theme}: {len(files)} new files", "green"))
    
    print(colored("=== Music Library Download Complete ===", "cyan"))

if __name__ == "__main__":
    # Run the async function
    asyncio.run(download_music_library()) 