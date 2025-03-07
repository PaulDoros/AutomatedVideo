import os
import requests
import random
import json
import uuid
from urllib.parse import quote
from termcolor import colored
import asyncio
import aiohttp
from dotenv import load_dotenv
from pathlib import Path
import time
import re
import shutil
import freesound

# Load environment variables
load_dotenv()

class MusicProvider:
    """
    A class to provide copyright-free music for videos from various sources.
    Supports multiple APIs and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the music provider"""
        # Load environment variables
        load_dotenv()
        
        # API keys
        self.pixabay_api_key = os.getenv('PIXABAY_API_KEY')
        self.freesound_api_key = os.getenv('FREESOUND_API_KEY')
        
        # Initialize Freesound client if API key is available
        self.freesound_client = None
        if self.freesound_api_key and self.freesound_api_key != 'your_freesound_api_key':
            try:
                self.freesound_client = freesound.FreesoundClient()
                self.freesound_client.set_token(self.freesound_api_key, "token")
                print(colored("✓ Freesound API client initialized", "green"))
            except Exception as e:
                print(colored(f"Error initializing Freesound client: {str(e)}", "red"))
        
        # Directory paths
        self.cache_dir = "cache/music"
        self.default_music_dir = "Assets/music"
        
        # Create directories if they don't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.default_music_dir, exist_ok=True)
        
        # Create channel-specific directories
        self.channel_dirs = {
            'tech_humor': os.path.join(self.default_music_dir, 'tech_humor'),
            'ai_money': os.path.join(self.default_music_dir, 'ai_money'),
            'baby_tips': os.path.join(self.default_music_dir, 'baby_tips'),
            'quick_meals': os.path.join(self.default_music_dir, 'quick_meals'),
            'fitness_motivation': os.path.join(self.default_music_dir, 'fitness_motivation')
        }
        
        # Create channel directories
        for channel_dir in self.channel_dirs.values():
            os.makedirs(channel_dir, exist_ok=True)
        
        # Music moods for each channel type
        self.music_moods = {
            'tech_humor': ['upbeat electronic', 'quirky instrumental', 'tech soundtrack', 'comedy background music', 'playful instrumental'],
            'ai_money': ['corporate instrumental', 'business background music', 'technology soundtrack', 'professional ambient', 'modern electronic'],
            'baby_tips': ['gentle lullaby', 'soft instrumental', 'peaceful ambient', 'calm background music', 'soothing melody'],
            'quick_meals': ['cheerful kitchen music', 'cooking soundtrack', 'upbeat acoustic', 'positive instrumental', 'light background music'],
            'fitness_motivation': ['energetic workout music', 'motivational instrumental', 'gym soundtrack', 'exercise beat', 'training music']
        }
        
        # Track used music to avoid repetition
        self.used_music = {}
        self._load_used_music()
        
        # NCS API endpoints
        self.ncs_api_url = "https://ncs.io/music-search"
        
        # Unminus API endpoint
        self.unminus_api_url = "https://www.unminus.com/api/v1/tracks"
    
    def _load_used_music(self):
        """Load the list of recently used music to avoid repetition"""
        try:
            used_music_file = os.path.join(self.cache_dir, "used_music.json")
            if os.path.exists(used_music_file):
                with open(used_music_file, 'r') as f:
                    self.used_music = json.load(f)
                print(colored(f"Loaded {sum(len(v) for v in self.used_music.values())} used music entries", "blue"))
            else:
                self.used_music = {}
        except Exception as e:
            print(colored(f"Error loading used music: {str(e)}", "yellow"))
            self.used_music = {}
    
    def _save_used_music(self, channel_type, music_path):
        """Save a music path as recently used"""
        try:
            # Initialize if not exists
            if channel_type not in self.used_music:
                self.used_music[channel_type] = []
            
            # Add to used list
            self.used_music[channel_type].append(music_path)
            
            # Keep only the last 10 used music files per channel
            if len(self.used_music[channel_type]) > 10:
                self.used_music[channel_type] = self.used_music[channel_type][-10:]
            
            # Save to file
            used_music_file = os.path.join(self.cache_dir, "used_music.json")
            with open(used_music_file, 'w') as f:
                json.dump(self.used_music, f)
        except Exception as e:
            print(colored(f"Error saving used music: {str(e)}", "yellow"))
    
    async def get_music_from_ncs(self, mood, duration=None):
        """
        Get music from NoCopyrightSounds (NCS) website
        This is a simplified implementation as NCS doesn't have an official API
        """
        try:
            # We'll use a web scraping approach to find music links
            # For now, we'll use a predefined list of NCS tracks by genre
            # In a production environment, you'd implement proper web scraping
            
            # Simulate NCS API response with predefined tracks
            ncs_tracks = {
                "electronic": [
                    "https://ncs.io/track/jim-yosef-anna-yvette-linked-1",
                    "https://ncs.io/track/unknown-brain-matilda-foy-dancing-with-your-ghost"
                ],
                "upbeat": [
                    "https://ncs.io/track/diviners-savannah-falling",
                    "https://ncs.io/track/lost-sky-fearless-pt-ii-feat-chris-linton"
                ],
                "energetic": [
                    "https://ncs.io/track/syn-cole-feel-good",
                    "https://ncs.io/track/elektronomia-sky-high"
                ],
                "corporate": [
                    "https://ncs.io/track/ellis-feel-that",
                    "https://ncs.io/track/unknown-brain-why-do-i-feat-bri-tolani"
                ],
                "gentle": [
                    "https://ncs.io/track/elektronomia-jjd-free",
                    "https://ncs.io/track/culture-code-make-me-move-feat-karra"
                ]
            }
            
            # Find matching tracks for the mood
            matching_tracks = []
            for key, tracks in ncs_tracks.items():
                if key in mood.lower() or any(m.lower() in key for m in mood.lower().split()):
                    matching_tracks.extend(tracks)
            
            if not matching_tracks:
                # Use a default category if no match
                matching_tracks = ncs_tracks["electronic"]
            
            # Select a random track
            track_url = random.choice(matching_tracks)
            
            # In a real implementation, you would download the track from the URL
            # For now, we'll return a placeholder path
            return f"ncs_track_{uuid.uuid4()}.mp3"
            
        except Exception as e:
            print(colored(f"Error getting music from NCS: {str(e)}", "yellow"))
            return None
    
    async def get_music_from_unminus(self, mood):
        """Get music from Unminus API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.unminus_api_url) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if not data:
                        return None
                    
                    # Filter tracks by mood if possible
                    # Unminus doesn't have mood filtering in API, so we'll do basic text matching
                    matching_tracks = []
                    for track in data:
                        track_name = track.get('name', '').lower()
                        track_description = track.get('description', '').lower()
                        
                        # Check if any mood keyword matches the track name or description
                        if any(m.lower() in track_name or m.lower() in track_description 
                               for m in mood.lower().split()):
                            matching_tracks.append(track)
                    
                    # If no matches, use all tracks
                    if not matching_tracks:
                        matching_tracks = data
                    
                    # Select a random track
                    track = random.choice(matching_tracks)
                    
                    # Download the track
                    download_url = track.get('audio_url')
                    if not download_url:
                        return None
                    
                    # Generate a unique filename
                    filename = f"unminus_{uuid.uuid4()}.mp3"
                    filepath = os.path.join(self.music_cache_dir, filename)
                    
                    # Download the file
                    async with session.get(download_url) as audio_response:
                        if audio_response.status != 200:
                            return None
                        
                        # Check content type to ensure it's an audio file
                        content_type = audio_response.headers.get('Content-Type', '')
                        if not content_type.startswith('audio/'):
                            print(colored(f"Warning: Unminus file is not audio (type: {content_type})", "yellow"))
                            # Try to get a different track
                            if len(matching_tracks) > 1:
                                matching_tracks.remove(track)
                                track = random.choice(matching_tracks)
                                download_url = track.get('audio_url')
                                if not download_url:
                                    return None
                                # Recursive call to try again with new track
                                return await self.get_music_from_unminus(mood)
                            return None
                        
                        with open(filepath, 'wb') as f:
                            f.write(await audio_response.read())
                        
                        # Verify file is a valid audio file
                        if os.path.getsize(filepath) < 1000:  # Less than 1KB is suspicious
                            print(colored(f"Warning: Unminus file is too small ({os.path.getsize(filepath)} bytes)", "yellow"))
                            os.remove(filepath)
                            return None
                        
                        return filepath
                    
        except Exception as e:
            print(colored(f"Error getting music from Unminus: {str(e)}", "yellow"))
            return None
    
    async def get_music_from_pixabay(self, mood, duration=None):
        """Get music from Pixabay API"""
        try:
            if not self.pixabay_api_key:
                print(colored("No Pixabay API key found", "yellow"))
                return None
            
            # Pixabay music API endpoint
            base_url = "https://pixabay.com/api/videos/"
            
            # Add "music" to the search query to focus on audio content
            search_query = f"{mood} music"
            
            params = {
                'key': self.pixabay_api_key,
                'q': search_query,
                'per_page': 20,
                'category': 'music'  # Specifically target music category
            }
            
            print(colored(f"Requesting music from Pixabay with query: {search_query} (category: music)", "blue"))
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        print(colored(f"Pixabay API error: Status {response.status}", "yellow"))
                        return None
                    
                    data = await response.json()
                    print(colored(f"Pixabay API response received: {len(data.get('hits', []))} hits", "blue"))
                    
                    if 'hits' not in data or not data['hits']:
                        print(colored(f"No results found for query: {search_query}", "yellow"))
                        return None
                    
                    # Select a random hit
                    hit = random.choice(data['hits'])
                    
                    # Get the music URL from the hit
                    music_url = None
                    if 'audio' in hit and hit['audio']:
                        music_url = hit['audio']
                    elif 'videos' in hit and hit['videos'].get('tiny', {}).get('url'):
                        # Extract audio from video if no direct audio URL
                        music_url = hit['videos']['tiny']['url']
                    
                    if not music_url:
                        print(colored(f"No music URL found in hit", "yellow"))
                        return None
                    
                    print(colored(f"Selected music: {hit.get('tags', 'Unknown')} - URL: {music_url}", "blue"))
                    
                    # Generate a unique filename
                    filename = f"pixabay_{uuid.uuid4()}.mp3"
                    filepath = os.path.join(self.cache_dir, filename)
                    
                    # Download the file
                    async with session.get(music_url) as audio_response:
                        if audio_response.status != 200:
                            print(colored(f"Failed to download music: Status {audio_response.status}", "yellow"))
                            return None
                        
                        with open(filepath, 'wb') as f:
                            f.write(await audio_response.read())
                        
                        # Verify file is a valid audio file
                        file_size = os.path.getsize(filepath)
                        if file_size < 1000:  # Less than 1KB is suspicious
                            print(colored(f"Warning: Downloaded file is too small ({file_size} bytes)", "yellow"))
                            os.remove(filepath)
                            return None
                        
                        print(colored(f"✓ Successfully downloaded Pixabay music: {file_size/1024/1024:.2f} MB", "green"))
                        return filepath
                    
        except Exception as e:
            print(colored(f"Error getting music from Pixabay: {str(e)}", "yellow"))
            return None
    
    async def _search_pixabay_audio(self, query, duration=None):
        """Helper method to search for audio on Pixabay using alternative methods"""
        try:
            # Try searching on Pixabay's audio section directly
            # This is a fallback method that tries to find audio content
            
            if not self.pixabay_api_key:
                return None
                
            base_url = "https://pixabay.com/api/"
            params = {
                'key': self.pixabay_api_key,
                'q': query,
                'per_page': 20,
                'safesearch': 'true',  # Safe content only
            }
            
            print(colored(f"Trying alternative Pixabay search: {query}", "blue"))
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    
                    if 'hits' not in data or not data['hits']:
                        return None
                    
                    # Try to find any audio content in the results
                    for hit in data['hits']:
                        # Look for any URL that might be an audio file
                        for key, value in hit.items():
                            if isinstance(value, str) and value.endswith('.mp3'):
                                # Found an MP3 URL
                                audio_url = value
                                
                                # Generate a unique filename
                                filename = f"pixabay_{uuid.uuid4()}.mp3"
                                filepath = os.path.join(self.music_cache_dir, filename)
                                
                                # Download the file
                                async with session.get(audio_url) as audio_response:
                                    if audio_response.status != 200:
                                        continue
                                    
                                    # Check content type
                                    content_type = audio_response.headers.get('Content-Type', '')
                                    if not content_type.startswith('audio/'):
                                        continue
                                    
                                    with open(filepath, 'wb') as f:
                                        f.write(await audio_response.read())
                                    
                                    # Verify file size
                                    if os.path.getsize(filepath) < 1000:
                                        os.remove(filepath)
                                        continue
                                    
                                    print(colored(f"✓ Found audio via alternative search: {os.path.getsize(filepath)/1024:.2f} KB", "green"))
                                    return filepath
            
            # If we get here, no audio was found
            return None
            
        except Exception as e:
            print(colored(f"Error in alternative Pixabay search: {str(e)}", "yellow"))
            return None
    
    def get_default_music(self, channel_type):
        """Get default music for a channel type from the assets directory"""
        try:
            # Check if we have default music for this channel
            default_dir = os.path.join("assets", "music", channel_type)
            if os.path.exists(default_dir):
                music_files = [f for f in os.listdir(default_dir) if f.endswith('.mp3')]
                if music_files:
                    # Select a random file
                    selected_file = random.choice(music_files)
                    music_path = os.path.join(default_dir, selected_file)
                    print(colored(f"✓ Using default music: {selected_file}", "green"))
                    return music_path
            
            # If no channel-specific default music, try the general default music
            general_dir = os.path.join("assets", "music", "default")
            os.makedirs(general_dir, exist_ok=True)
            
            if os.path.exists(general_dir):
                music_files = [f for f in os.listdir(general_dir) if f.endswith('.mp3')]
                if music_files:
                    # Select a random file
                    selected_file = random.choice(music_files)
                    music_path = os.path.join(general_dir, selected_file)
                    print(colored(f"✓ Using general default music: {selected_file}", "green"))
                    return music_path
            
            print(colored(f"No default music found for {channel_type}", "yellow"))
            return None
            
        except Exception as e:
            print(colored(f"Error getting default music: {str(e)}", "yellow"))
            return None
    
    async def get_music_from_freesound(self, mood, duration=None):
        """Get music from Freesound API"""
        if not self.freesound_client:
            print(colored("Freesound client not initialized. Check your API key.", "yellow"))
            return None
        
        try:
            print(colored(f"Searching Freesound for '{mood}' music...", "blue"))
            
            # Set up search parameters
            fields = "id,name,previews,duration,license,tags,avg_rating,num_downloads,description"
            
            # Add duration filter if specified
            filter_string = "duration:[15 TO 120]"  # 15-120 seconds (good for shorts)
            if duration:
                max_duration = min(int(duration) + 15, 120)  # Add some buffer, max 120 seconds
                filter_string = f"duration:[{max(15, int(duration) - 5)} TO {max_duration}]"
            
            # Add license filter for royalty-free content
            filter_string += " license:\"Creative Commons 0\""
            
            # Enhance search query for better music results
            search_query = mood
            if "music" not in mood.lower():
                search_query += " music"
                
            # Add quality indicators to search
            search_query += " instrumental"
            
            print(colored(f"Using search query: '{search_query}'", "blue"))
            print(colored(f"With filters: {filter_string}", "blue"))
            
            # Search for sounds
            results = self.freesound_client.text_search(
                query=search_query,
                filter=filter_string,
                fields=fields,
                sort="rating_desc",
                page_size=20
            )
            
            if not results or results.count == 0:
                print(colored(f"No results found for '{mood}' on Freesound", "yellow"))
                
                # Try a more specific music search if specific search fails
                generic_query = "background music instrumental"
                print(colored(f"Trying generic search: '{generic_query}'...", "blue"))
                
                results = self.freesound_client.text_search(
                    query=generic_query,
                    filter=filter_string,
                    fields=fields,
                    sort="rating_desc",
                    page_size=20
                )
                
                if not results or results.count == 0:
                    print(colored(f"No results found for generic search either", "yellow"))
                    return None
            
            # Get a random sound from the results
            sounds = list(results)
            if not sounds:
                print(colored("No sounds in results", "yellow"))
                return None
            
            # Filter for actual music by checking tags and description
            music_sounds = []
            for sound in sounds:
                # Check if it has music-related tags
                has_music_tags = False
                if hasattr(sound, 'tags'):
                    music_related_tags = ['music', 'instrumental', 'soundtrack', 'background-music', 
                                         'background', 'melody', 'ambient', 'score']
                    for tag in music_related_tags:
                        if tag in sound.tags:
                            has_music_tags = True
                            break
                
                # Check description for music indicators
                has_music_description = False
                if hasattr(sound, 'description'):
                    music_indicators = ['music', 'instrumental', 'soundtrack', 'background', 
                                       'melody', 'ambient', 'score', 'track', 'song']
                    for indicator in music_indicators:
                        if indicator.lower() in sound.description.lower():
                            has_music_description = True
                            break
                
                # Add to music sounds if it passes filters
                if has_music_tags or has_music_description:
                    music_sounds.append(sound)
            
            # Use filtered music sounds if available, otherwise use all sounds
            target_sounds = music_sounds if music_sounds else sounds
            
            # Sort by rating and downloads if available
            rated_sounds = []
            for s in target_sounds:
                rating_score = 0
                if hasattr(s, 'avg_rating') and s.avg_rating:
                    rating_score += float(s.avg_rating) * 2  # Weight rating more heavily
                
                download_score = 0
                if hasattr(s, 'num_downloads') and s.num_downloads:
                    download_score += min(float(s.num_downloads) / 100, 5)  # Cap at 5 points
                
                rated_sounds.append((s, rating_score + download_score))
            
            # Sort by combined score
            rated_sounds.sort(key=lambda x: x[1], reverse=True)
            
            # Choose from top sounds
            if rated_sounds:
                # Select from top 5 or 33% of sounds, whichever is greater
                top_count = max(5, len(rated_sounds) // 3)
                sound = random.choice(rated_sounds[:top_count])[0]
            else:
                sound = random.choice(target_sounds)
            
            # Download the preview
            filename = f"freesound_{sound.id}_{uuid.uuid4()}.mp3"
            filepath = os.path.join(self.cache_dir, filename)
            
            # Use the high-quality preview
            preview_url = sound.previews.preview_hq_mp3
            
            print(colored(f"Downloading '{sound.name}' from Freesound...", "blue"))
            
            # Download the file
            response = requests.get(preview_url)
            if response.status_code != 200:
                print(colored(f"Failed to download from Freesound: Status {response.status_code}", "red"))
                return None
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Check file size
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            if file_size < 0.1:  # Less than 100KB is suspicious
                print(colored(f"File too small: {file_size:.2f} MB", "yellow"))
                os.remove(filepath)
                return None
            
            print(colored(f"✓ Downloaded music from Freesound: {sound.name}", "green"))
            print(colored(f"  File size: {file_size:.2f} MB", "green"))
            
            return filepath
            
        except Exception as e:
            print(colored(f"Error getting music from Freesound: {str(e)}", "red"))
            return None
    
    async def get_music_for_channel(self, channel_type, duration=None):
        """Get appropriate music for a channel type"""
        try:
            # First check if we have local music for this channel
            channel_dir = self.channel_dirs.get(channel_type)
            if channel_dir and os.path.exists(channel_dir):
                music_files = [f for f in os.listdir(channel_dir) if f.endswith('.mp3')]
                
                # Filter out recently used music
                unused_music = [f for f in music_files if os.path.join(channel_dir, f) not in self.used_music.get(channel_type, [])]
                
                # If we have unused music, use it
                if unused_music:
                    selected_file = random.choice(unused_music)
                    music_path = os.path.join(channel_dir, selected_file)
                    
                    # Mark as used
                    self._save_used_music(channel_type, music_path)
                    
                    print(colored(f"✓ Using local music: {selected_file}", "green"))
                    return music_path
            
            # If no local music, use default music
            default_music = self.get_default_music(channel_type)
            if default_music:
                print(colored(f"✓ Using default music for {channel_type}", "green"))
                return default_music
            
            # If all fails, try to download from APIs as a last resort
            moods = self.music_moods.get(channel_type, ['background', 'ambient'])
            mood = random.choice(moods)
            
            print(colored(f"Searching for {mood} music for {channel_type}...", "blue"))
            
            # Try Freesound first (if API key is available)
            if self.freesound_client:
                music_path = await self.get_music_from_freesound(mood, duration)
                if music_path:
                    # Copy to channel directory
                    channel_dir = self.channel_dirs.get(channel_type)
                    if channel_dir:
                        new_filename = f"{channel_type}_{uuid.uuid4()}.mp3"
                        new_path = os.path.join(channel_dir, new_filename)
                        
                        # Copy file
                        with open(music_path, 'rb') as src, open(new_path, 'wb') as dst:
                            dst.write(src.read())
                        
                        # Mark as used
                        self._save_used_music(channel_type, new_path)
                        
                        print(colored(f"✓ Downloaded new music for {channel_type} from Freesound", "green"))
                        return new_path
                    
                    return music_path
            
            # Try Unminus next
            music_path = await self.get_music_from_unminus(mood)
            
            if not music_path:
                music_path = await self.get_music_from_ncs(mood, duration)
            
            if not music_path:
                # Try Pixabay as last resort
                music_path = await self.get_music_from_pixabay(mood, duration)
            
            # If we found music from APIs, save it
            if music_path and os.path.exists(music_path):
                # Copy to channel directory
                channel_dir = self.channel_dirs.get(channel_type)
                if channel_dir:
                    new_filename = f"{channel_type}_{uuid.uuid4()}.mp3"
                    new_path = os.path.join(channel_dir, new_filename)
                    
                    # Copy file
                    with open(music_path, 'rb') as src, open(new_path, 'wb') as dst:
                        dst.write(src.read())
                    
                    # Mark as used
                    self._save_used_music(channel_type, new_path)
                    
                    print(colored(f"✓ Downloaded new music for {channel_type}", "green"))
                    return new_path
                
                return music_path
            
            # If all fails, return None
            print(colored(f"Could not find suitable music for {channel_type}", "yellow"))
            return None
            
        except Exception as e:
            print(colored(f"Error getting music for channel: {str(e)}", "red"))
            return None

    async def download_music_for_video(self, video_theme, video_keywords=None, duration=None):
        """
        Download music specifically matched to a video's theme and keywords
        
        Args:
            video_theme (str): The main theme of the video (e.g., 'tech', 'cooking')
            video_keywords (list): List of keywords related to the video content
            duration (float): Desired duration of the music in seconds
            
        Returns:
            str: Path to the downloaded music file
        """
        try:
            # Create a search query based on video theme and keywords
            search_terms = []
            
            # Add the video theme
            if video_theme:
                search_terms.append(video_theme)
            
            # Add relevant keywords (limit to 3 most relevant)
            if video_keywords and isinstance(video_keywords, list):
                # Filter out common words that aren't useful for music search
                common_words = ['the', 'and', 'or', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
                filtered_keywords = [k for k in video_keywords if k.lower() not in common_words and len(k) > 2]
                
                # Add up to 3 keywords
                search_terms.extend(filtered_keywords[:3])
            
            # Add music-related terms to improve results
            music_terms = ['background music', 'soundtrack', 'instrumental']
            search_terms.append(random.choice(music_terms))
            
            # Create the search query
            search_query = ' '.join(search_terms)
            print(colored(f"Searching for music with query: '{search_query}'", "blue"))
            
            # Try different music sources in order of preference
            
            # 1. Try Freesound first (if API key is available)
            if self.freesound_client:
                print(colored("Trying Freesound API...", "blue"))
                music_path = await self.get_music_from_freesound(search_query, duration)
                if music_path:
                    print(colored(f"✓ Found matching music on Freesound for '{video_theme}'", "green"))
                    return music_path
            
            # 2. Try Pixabay next
            print(colored("Trying Pixabay API...", "blue"))
            music_path = await self.get_music_from_pixabay(search_query, duration)
            if music_path:
                print(colored(f"✓ Found matching music on Pixabay for '{video_theme}'", "green"))
                return music_path
            
            # 3. Try other sources
            print(colored("Trying other music sources...", "blue"))
            music_path = await self.get_music_from_unminus(search_query)
            if music_path:
                print(colored(f"✓ Found matching music on Unminus for '{video_theme}'", "green"))
                return music_path
            
            # 4. Fall back to channel-based music if all else fails
            if video_theme in self.music_moods:
                channel_type = video_theme
                print(colored(f"Falling back to channel-based music for '{channel_type}'", "yellow"))
                return await self.get_music_for_channel(channel_type, duration)
            
            # If all fails, return None
            print(colored(f"Could not find suitable music for video theme: '{video_theme}'", "yellow"))
            return None
            
        except Exception as e:
            print(colored(f"Error downloading music for video: {str(e)}", "red"))
            return None

    def initialize_freesound_client(self):
        """Initialize the Freesound client with API key"""
        try:
            if not self.freesound_api_key or self.freesound_api_key == 'your_freesound_api_key':
                print(colored("Freesound API key not configured", "yellow"))
                return False
                
            if self.freesound_client is None:
                self.freesound_client = freesound.FreesoundClient()
                self.freesound_client.set_token(self.freesound_api_key, "token")
                print(colored("✓ Freesound API client initialized", "green"))
            
            return self.freesound_client is not None
        except Exception as e:
            print(colored(f"Error initializing Freesound client: {str(e)}", "red"))
            return False
            
    def load_used_music(self):
        """Alias for _load_used_music for backward compatibility"""
        return self._load_used_music()

# Singleton instance
music_provider = MusicProvider() 