import os
import requests
from termcolor import colored
import zipfile
import shutil
from pytube import YouTube, Playlist
import json
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

class MusicDownloader:
    def __init__(self):
        load_dotenv()
        self.music_dirs = {
            'tech_humor': 'assets/music/tech',
            'ai_money': 'assets/music/business',
            'baby_tips': 'assets/music/parenting',
            'quick_meals': 'assets/music/cooking',
            'fitness_motivation': 'assets/music/fitness'
        }
        
        # Load Pixabay API key from .env
        self.pixabay_api_key = os.getenv('PIXABAY_API_KEY')
        
        # Create directories
        for directory in self.music_dirs.values():
            os.makedirs(directory, exist_ok=True)

    def download_from_pixabay(self, category, keywords, count=5):
        """Download music from Pixabay Music API"""
        try:
            # Construct API URL with correct endpoint
            base_url = "https://pixabay.com/api/"  # Changed from api/music to api
            params = {
                'key': self.pixabay_api_key,
                'q': ' '.join(keywords),
                'per_page': count,
                'media_type': 'music'
            }
            
            print(colored(f"\nDownloading {category} music from Pixabay...", "blue"))
            
            # Get music list
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(colored(f"Pixabay API error: {response.status_code}", "red"))
                return
                
            data = response.json()
            
            if 'hits' not in data or not data['hits']:
                print(colored(f"No music found for {category}", "yellow"))
                return
            
            # Download each track
            target_dir = self.music_dirs[category]
            for i, track in enumerate(data['hits'], 1):
                try:
                    audio_url = track.get('audio') or track.get('previewURL')
                    if not audio_url:
                        continue
                        
                    filename = f"{category}_{i}.mp3"
                    filepath = os.path.join(target_dir, filename)
                    
                    response = requests.get(audio_url)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(colored(f"✓ Downloaded: {filename}", "green"))
                except Exception as e:
                    print(colored(f"✗ Error downloading track {i}: {str(e)}", "red"))
            
        except Exception as e:
            print(colored(f"✗ Error accessing Pixabay API: {str(e)}", "red"))

    def download_from_youtube_playlist(self, category, playlist_url):
        """Download music from YouTube playlist (No Copyright Music)"""
        try:
            playlist = Playlist(playlist_url)
            target_dir = self.music_dirs[category]
            
            print(colored(f"\nDownloading {category} music from YouTube playlist...", "blue"))
            
            # Get first 5 videos from playlist
            videos = list(playlist.videos)[:5]
            
            for i, video in enumerate(videos, 1):
                try:
                    # Download audio only
                    audio_stream = video.streams.filter(only_audio=True).first()
                    if not audio_stream:
                        continue
                        
                    filename = f"{category}_yt_{i}.mp3"
                    filepath = os.path.join(target_dir, filename)
                    
                    # Download to temporary file first
                    temp_file = audio_stream.download(
                        output_path=target_dir,
                        filename=f"temp_{filename}"
                    )
                    
                    # Rename to final filename
                    os.rename(temp_file, filepath)
                    
                    print(colored(f"✓ Downloaded: {filename}", "green"))
                except Exception as e:
                    print(colored(f"✗ Error downloading video {i}: {str(e)}", "red"))
                    
        except Exception as e:
            print(colored(f"✗ Error accessing YouTube playlist: {str(e)}", "red"))

    def setup_default_music(self):
        """Setup music for all channels"""
        # Real no-copyright music playlists
        categories = {
            'tech_humor': {
                'keywords': ['funny', 'upbeat', 'electronic'],
                'youtube_playlist': 'https://www.youtube.com/playlist?list=PLzCxunOM5WFJ7sbHi_9Rl1MuIOqz6nY-Z'  # NCS Electronic
            },
            'ai_money': {
                'keywords': ['corporate', 'motivation', 'business'],
                'youtube_playlist': 'https://www.youtube.com/playlist?list=PLwJjxqYuirCLkq42mGw4XKGQlpZSfxsYd'  # Business Background
            },
            'baby_tips': {
                'keywords': ['gentle', 'calm', 'lullaby'],
                'youtube_playlist': 'https://www.youtube.com/playlist?list=PLwJjxqYuirCL-VvZUWBzahZZ3D-p4zOIS'  # Calm Music
            },
            'quick_meals': {
                'keywords': ['cheerful', 'acoustic', 'light'],
                'youtube_playlist': 'https://www.youtube.com/playlist?list=PLwJjxqYuirCLkXqX1FpJrwUi8_WJ89ZKF'  # Happy Background
            },
            'fitness_motivation': {
                'keywords': ['energetic', 'workout', 'pump'],
                'youtube_playlist': 'https://www.youtube.com/playlist?list=PLwJjxqYuirCLK5hVXyKRYSkxH8yP3PkGE'  # Workout Music
            }
        }
        
        # Download music for each category
        for category, info in categories.items():
            print(colored(f"\n=== Setting up music for {category} ===", "blue"))
            
            # Try Pixabay first
            self.download_from_pixabay(category, info['keywords'])
            
            # If not enough music, try YouTube
            music_dir = self.music_dirs[category]
            if len([f for f in os.listdir(music_dir) if f.endswith('.mp3')]) < 5:
                self.download_from_youtube_playlist(category, info['youtube_playlist'])

def setup_music():
    """Main function to setup music library"""
    downloader = MusicDownloader()
    downloader.setup_default_music()
    print(colored("\n✓ Music setup complete!", "green"))

if __name__ == "__main__":
    setup_music() 