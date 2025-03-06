from moviepy.editor import ImageClip
import pysrt
from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
import subprocess
from termcolor import colored
from content_validator import ContentValidator, ScriptGenerator
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip, concatenate_audioclips, AudioClip, CompositeAudioClip
from moviepy.video.tools.subtitles import SubtitlesClip
from tiktokvoice import tts
from video import generate_video, generate_subtitles, combine_videos, save_video, generate_tts_audio, trim_audio_file
from test_content_quality import ContentQualityChecker
import json
import time
from datetime import datetime
import asyncio
from pathlib import Path
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import shutil
from search import search_for_stock_videos  # Add this import
from dotenv import load_dotenv
import uuid
import re
from pydub import AudioSegment
import openai
import glob
import traceback
import logging
import requests

# Third-party imports
from TTS.api import TTS
import torch
from scipy.io import wavfile

# Import voice diversification system
from voice_diversification import VoiceDiversification


class AudioManager:
    def __init__(self):
        self.voices = {
            'tech_humor': {
                'voices': {
                    'nova': {'weight': 3, 'description': 'Energetic female, perfect for tech humor'},
                    'echo': {'weight': 3, 'description': 'Dynamic male, great for casual tech content'}
                },
                'style': 'humorous',
                'prompt': "Read this text with an energetic, playful tone and a sense of humor."
            },
            'ai_money': {
                'voices': {
                    'onyx': {'weight': 3, 'description': 'Professional male voice'},
                    'shimmer': {'weight': 2, 'description': 'Clear female voice'}
                },
                'style': 'professional',
                'prompt': "Deliver the following content in a clear, confident, and professional manner."
            },
            'default': {
                'voices': {
                    'echo': {'weight': 2, 'description': 'Balanced male voice'},
                    'nova': {'weight': 2, 'description': 'Engaging female voice'}
                },
                'style': 'casual',
                'prompt': "Please read the following text in a natural, conversational tone."
            }
        }
        
        # Initialize voice diversification system
        self.voice_diversifier = VoiceDiversification()
        
        self.voiceovers_dir = "temp/tts"
        os.makedirs(self.voiceovers_dir, exist_ok=True)
        
        # Map content types to emotions
        self.content_emotions = {
            'tech_humor': ['cheerful', 'friendly'],
            'ai_money': ['professional', 'serious'],
            'baby_tips': ['friendly', 'cheerful'],
            'quick_meals': ['cheerful', 'friendly'],
            'fitness_motivation': ['professional', 'serious'],
            'default': ['neutral', 'friendly']
        }
        
        # Initialize OpenAI client
        load_dotenv()
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def select_voice(self, channel_type):
        """Select appropriate voice based on content type"""
        channel_config = self.voices.get(channel_type, self.voices['default'])
        voices = channel_config['voices']
        
        # Weight-based selection
        total_weight = sum(v['weight'] for v in voices.values())
        r = random.uniform(0, total_weight)
        
        current_weight = 0
        for voice, config in voices.items():
            current_weight += config['weight']
            if r <= current_weight:
                return voice, channel_config['style']
        
        return list(voices.keys())[0], channel_config['style']

    def select_coqui_voice(self, channel_type, gender=None):
        """Select a Coqui TTS voice based on channel type and gender preference"""
        try:
            # Get appropriate emotion for the channel using the new emotion selection system
            if hasattr(self, 'voice_diversifier') and self.voice_diversifier:
                emotion = self.voice_diversifier.get_emotion_for_content(channel_type)
            else:
                # Fallback to basic emotion mapping
                emotion_map = {
                    'tech_humor': 'humorous',
                    'ai_money': 'professional',
                    'baby_tips': 'warm',
                    'quick_meals': 'cheerful',
                    'fitness_motivation': 'energetic'
                }
                emotion = emotion_map.get(channel_type, 'neutral')
            
            # Check if we have a cached voice for this channel type
            cache_key = f"{channel_type}_{gender if gender else 'any'}"
            if hasattr(self, 'voice_cache') and cache_key in self.voice_cache:
                voice, stored_emotion = self.voice_cache[cache_key]
                print(colored(f"Using cached voice: {voice} (emotion: {stored_emotion})", "blue"))
                return voice, stored_emotion
            
            # Initialize voice cache if it doesn't exist
            if not hasattr(self, 'voice_cache'):
                self.voice_cache = {}
            
            # Get available voices from the voice diversifier
            if hasattr(self, 'voice_diversifier') and self.voice_diversifier:
                # Select a voice based on channel type and gender preference
                voice = self.voice_diversifier.select_voice(channel_type, gender)
                
                # Cache the selected voice for future use
                self.voice_cache[cache_key] = (voice, emotion)
                
                print(colored(f"Selected Coqui voice: {voice} (emotion: {emotion})", "blue"))
                return voice, emotion
            else:
                print(colored("Voice diversifier not initialized, using default voice", "yellow"))
                return "default", emotion
                
        except Exception as e:
            print(colored(f"Error selecting Coqui voice: {str(e)}", "red"))
            return "default", "neutral"

    def enhance_audio(self, audio_segment):
        """Enhance audio quality"""
        try:
            # Normalize volume
            audio = audio_segment.normalize()
            
            # Apply subtle compression
            audio = audio.compress_dynamic_range()
            
            # Adjust speed if needed
            # audio = audio._spawn(audio.raw_data, overrides={'frame_rate': int(audio.frame_rate * speed)})
            
            return audio
            
        except Exception as e:
            print(colored(f"Error enhancing audio: {str(e)}", "red"))
            return audio_segment

    async def generate_tts_coqui(self, text, voice, emotion="neutral", speed=1.0):
        """Generate TTS using Coqui TTS"""
        try:
            # Create unique filename based on voice and emotion
            filename = f"{voice}_{emotion}_{int(time.time())}.wav"
            output_path = os.path.join(self.voiceovers_dir, filename)
            
            print(colored(f"Using Coqui TTS voice: {voice} (emotion: {emotion}, speed: {speed})", "blue"))
            
            # Generate voice using the Coqui TTS API
            result = await self.voice_diversifier.api.generate_voice(
                text=text,
                speaker=voice,
                language="en",
                emotion=emotion,
                speed=speed,
                output_path=output_path
            )
            
            if result:
                # Convert WAV to MP3 for compatibility with video generation
                mp3_path = os.path.join(self.voiceovers_dir, f"{os.path.splitext(os.path.basename(result))[0]}.mp3")
                
                # Use pydub to convert WAV to MP3
                audio = AudioSegment.from_wav(result)
                audio.export(mp3_path, format="mp3", bitrate="128k")
                
                print(colored(f"✓ Generated TTS audio with Coqui: {mp3_path}", "green"))
                return mp3_path
            else:
                raise ValueError("Failed to generate TTS with Coqui")
        except Exception as e:
            print(colored(f"Coqui TTS failed: {str(e)}", "red"))
            return None
    
    async def generate_tts_openai(self, text, voice="nova"):
        """Generate TTS using OpenAI's voice API"""
        try:
            # Ensure we have text to synthesize
            if not text.strip():
                raise ValueError("Empty script for TTS generation")
                
            # Create unique filename
            output_path = os.path.join(self.voiceovers_dir, f"openai_{voice}_{int(time.time())}.mp3")
            
            print(colored(f"Using OpenAI voice: {voice}", "blue"))
            
            # Call OpenAI API
            response = await asyncio.to_thread(
                self.openai_client.audio.speech.create,
                model="tts-1",
                voice=voice,
                input=text
            )
            
            # Save to file
            response.stream_to_file(output_path)
            
            print(colored(f"✓ Generated TTS audio with OpenAI: {output_path}", "green"))
            return output_path
            
        except Exception as e:
            print(colored(f"OpenAI TTS failed: {str(e)}", "red"))
            return None

async def generate_tts(text, voice, output_path):
    """Generate TTS using TikTok voice"""
    try:
        # Generate TTS
        tts(text=text, voice=voice, filename=output_path)
        
        # Create audio clip for duration calculation
        audio_clip = AudioFileClip(output_path)
        
        return True, [audio_clip]
    except Exception as e:
        print(colored(f"TTS Generation failed: {str(e)}", "red"))
        return False, None

class VideoGenerator:
    def __init__(self):
        """Initialize the video generator"""
        load_dotenv()
        self.validator = ContentValidator()
        self.script_generator = ScriptGenerator()
        self.quality_checker = ContentQualityChecker()
        self.audio_manager = AudioManager()
        
        # Directories
        self.output_dir = "output/videos"
        self.temp_dir = "temp"
        self.assets_dir = "assets/videos"
        
        # Background music settings
        self.use_background_music = True
        self._music_volume = 0.3  # Increased from 0.15 to 0.3 for better audibility
        self.music_fade_in = 2.0  # Fade in duration in seconds
        self.music_fade_out = 3.0  # Fade out duration in seconds
        
        print(colored(f"Background music: {'Enabled' if self.use_background_music else 'Disabled'}", "blue"))
        if self.use_background_music:
            print(colored(f"Music volume: {self._music_volume}", "blue"))
        
        # Hardware optimization settings
        self.n_threads = min(multiprocessing.cpu_count(), 16)  # Use up to 16 CPU threads
        self.use_gpu = True  # Enable GPU acceleration
        
        # Create necessary directories
        for directory in [self.output_dir, self.temp_dir, self.assets_dir]:
            os.makedirs(directory, exist_ok=True)

        self.pexels_api_key = os.getenv('PEXELS_API_KEY')

        # Create necessary directories
        self.dirs = {
            'temp': {
                'videos': ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation'],
                'tts': [],
                'subtitles': []
            },
            'output': {
                'videos': []
            },
            'assets': {
                'videos': ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation'],
                'music': ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation'],
                'fonts': []
            }
        }
        
        self._create_directories()
        
        # Flag to use Coqui TTS instead of OpenAI TTS
        self.use_coqui_tts = True

        # Initialize logger
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """Create all necessary directories"""
        try:
            for main_dir, sub_dirs in self.dirs.items():
                for sub_dir, channels in sub_dirs.items():
                    # Create base directory
                    base_path = f"{main_dir}/{sub_dir}"
                    os.makedirs(base_path, exist_ok=True)
                    
                    # Create channel-specific subdirectories if any
                    for channel in channels:
                        os.makedirs(f"{base_path}/{channel}", exist_ok=True)
                        
            print(colored("✓ Directory structure created", "green"))
            
        except Exception as e:
            print(colored(f"Error creating directories: {str(e)}", "red"))

    def generate_video_for_channel(self, channel, topic, hashtags):
        """Generate and validate video content"""
        try:
            # Generate and validate script
            success, script = self.script_generator.generate_script(topic, channel)
            if not success:
                print(colored("Failed to generate valid script", "red"))
                return None

            # Validate video length before rendering
            estimated_duration = self.validator.estimate_video_length(script)
            if not self.is_duration_valid(estimated_duration, channel):
                print(colored(f"Estimated duration ({estimated_duration}s) not suitable", "red"))
                return None

            # Generate video content
            video_data = {
                'script': script,
                'title': f"{topic} | {' '.join(hashtags[:3])}",
                'description': self.generate_description(script, hashtags),
                'tags': hashtags,
                'thumbnail_text': self.generate_thumbnail_text(topic)
            }

            # Check content completeness
            is_complete, missing = self.validator.check_content_completeness(video_data)
            if not is_complete:
                print(colored(f"Missing content elements: {', '.join(missing)}", "red"))
                return None

            # Generate the actual video
            video_path = self.render_video(video_data)
            if not video_path:
                return None

            return {
                'video_path': video_path,
                'title': video_data['title'],
                'description': video_data['description'],
                'tags': video_data['tags']
            }

        except Exception as e:
            print(colored(f"Error generating video: {str(e)}", "red"))
            return None

    def is_duration_valid(self, duration, channel):
        # Implementation of is_duration_valid method
        pass

    def generate_description(self, script, hashtags):
        # Implementation of generate_description method
        pass

    def generate_thumbnail_text(self, topic):
        # Implementation of generate_thumbnail_text method
        pass

    def render_video(self, video_data):
        # Implementation of render_video method
        pass

    def get_voice_for_channel(self, channel_type):
        """Get appropriate voice and language code for each channel"""
        voices = {
            'tech_humor': {'voice': 'en_us_006', 'lang': 'en'},     # Male energetic
            'ai_money': {'voice': 'en_us_002', 'lang': 'en'},       # Male professional
            'baby_tips': {'voice': 'en_female_ht', 'lang': 'en'},   # Female warm
            'quick_meals': {'voice': 'en_us_009', 'lang': 'en'},    # Female enthusiastic
            'fitness_motivation': {'voice': 'en_male_narration', 'lang': 'en'} # Male motivational
        }
        return voices.get(channel_type, {'voice': 'en_us_002', 'lang': 'en'})

    def get_background_music(self, channel_type):
        """Get random background music for the channel"""
        music_dir = f"assets/music/{channel_type}"
        if not os.path.exists(music_dir):
            return None
        
        music_files = [f for f in os.listdir(music_dir) if f.endswith('.mp3')]
        if not music_files:
            return None
        
        return os.path.join(music_dir, random.choice(music_files))

    async def create_video(self, script_file, channel_type, output_path=None):
        """Create a video from a script file"""
        try:
            print(colored("\n=== Video Generation Started ===", "blue"))
            
            # Create necessary directories
            self._create_directories()
            
            # Load script from file
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    try:
                        # Try to load as JSON first
                        script_data = json.load(f)
                        script = script_data.get('script', '')
                    except json.JSONDecodeError:
                        # If not JSON, reset file pointer and read as plain text
                        f.seek(0)
                        script = f.read()
            except Exception as e:
                print(colored(f"Error loading script: {str(e)}", "red"))
                return False
                
            # Check if script is empty
            if not script.strip():
                print(colored("Error: Script is empty", "red"))
                return False
                
            # Check if we have a plain text version of the script (which might be better formatted)
            txt_script_file = script_file.replace('.json', '.txt')
            if os.path.exists(txt_script_file):
                try:
                    with open(txt_script_file, 'r', encoding='utf-8') as f:
                        txt_script = f.read()
                        if txt_script.strip():
                            print(colored("Using plain text version of script for better formatting", "green"))
                            script = txt_script
                except Exception as e:
                    print(colored(f"Error loading plain text script: {str(e)}", "yellow"))
                    # Continue with the JSON script
            
            # Print the script being used
            print(colored(f"Script to be used for video generation:\n{script}", "blue"))
            
            # Get voice for channel
            voice = self.get_voice_for_channel(channel_type)
            print(colored(f"Selected voice: {voice}", "green"))
            
            # Get background music if enabled
            background_music = None
            if self.use_background_music:
                background_music = self.get_background_music(channel_type)
                if background_music:
                    print(colored(f"Selected background music for {channel_type}", "green"))
            
            # Generate TTS audio
            tts_path = await self._generate_tts(script, channel_type)
            if not tts_path:
                print(colored("Failed to generate TTS audio", "red"))
                return False
                
            # Generate subtitles
            subtitle_path = await self._generate_subtitles(script, tts_path, channel_type)
            if not subtitle_path:
                print(colored("Failed to generate subtitles", "red"))
                return False
                
            # Process background videos
            background_videos = await self._process_background_videos(channel_type, script)
            if not background_videos:
                print(colored("Failed to process background videos", "red"))
                return False
                
            # Generate video
            video_path = await self._generate_video(tts_path, subtitle_path, background_videos, channel_type, output_path)
            if not video_path:
                print(colored("Failed to generate video", "red"))
                return False
                
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Clean up video library to prevent excessive accumulation
            # Only keep 20 videos per channel, and always keep videos newer than 30 days
            self.cleanup_video_library(channel_type, max_videos=20, days_to_keep=30)
            
            print(colored("\n=== Video Generation Complete ===", "blue"))
            
            return True
            
        except Exception as e:
            print(colored(f"Error creating video: {str(e)}", "red"))
            traceback.print_exc()
            return False

    async def _generate_tts(self, script, channel_type):
        """Generate TTS using either Coqui TTS or OpenAI's voice"""
        try:
            # First, check if we have a cleaned script in the JSON file
            json_path = f"cache/scripts/{channel_type}_latest.json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        script_data = json.load(f)
                        if "cleaned_script" in script_data and script_data["cleaned_script"].strip():
                            clean_script = script_data["cleaned_script"]
                            print(colored(f"Using pre-cleaned script from JSON file", "green"))
                            print(colored(f"Script for TTS:\n{clean_script}", "blue"))
                        else:
                            # Fall back to cleaning the script ourselves
                            print(colored(f"No cleaned script found in JSON, cleaning manually", "yellow"))
                            clean_script = self._clean_script_for_tts(script)
                except Exception as e:
                    print(colored(f"Error reading JSON file: {str(e)}", "yellow"))
                    clean_script = self._clean_script_for_tts(script)
            else:
                # Clean the script manually
                clean_script = self._clean_script_for_tts(script)
            
            # Check if the script is empty after cleaning
            if not clean_script.strip():
                print(colored("Warning: Script is empty after cleaning. Cannot generate TTS.", "red"))
                return None
            
            # Use Coqui TTS for diverse voices
            if self.use_coqui_tts:
                try:
                    # Select appropriate emotion for the content type
                    # (voice selection will be handled by the API based on whether the model supports speakers)
                    voice, emotion = self.audio_manager.select_coqui_voice(channel_type)
                    
                    # Generate TTS using Coqui
                    tts_path = await self.audio_manager.generate_tts_coqui(
                        clean_script, 
                        voice=voice, 
                        emotion=emotion
                    )
                    
                    if tts_path:
                        print(colored(f"✓ Generated TTS using Coqui with voice: {voice}, emotion: {emotion}", "green"))
                        return tts_path
                    else:
                        print(colored("Failed to generate TTS with Coqui, falling back to OpenAI", "yellow"))
                except Exception as e:
                    print(colored(f"Error with Coqui TTS: {str(e)}", "yellow"))
                    print(colored("Falling back to OpenAI TTS", "yellow"))
            
            # Fall back to OpenAI TTS
            voice = self.get_voice_for_channel(channel_type)
            tts_path = await self.audio_manager.generate_tts_openai(clean_script, voice)
            
            if tts_path:
                print(colored(f"✓ Generated TTS using OpenAI with voice: {voice}", "green"))
                return tts_path
            else:
                print(colored("Failed to generate TTS", "red"))
                return None
                
        except Exception as e:
            print(colored(f"Error generating TTS: {str(e)}", "red"))
            return None
    
    def _clean_script_for_tts(self, script):
        """Clean the script for TTS generation"""
        # Clean up the script - remove line numbers and preserve emojis for display
        # but the actual TTS will have emojis removed by the _clean_text method
        clean_script = '\n'.join(
            line.strip().strip('"') 
            for line in script.split('\n') 
            if line.strip() and not line[0].isdigit()
        )
        
        # Print the script before processing
        print(colored(f"Original script for TTS:\n{clean_script}", "blue"))
        
        # Remove section headers like "**HOOK:**", "Problem/Setup:", etc.
        section_headers = [
            "**Hook:**", "**Problem/Setup:**", "**Solution/Development:**", 
            "**Result/Punchline:**", "**Call to action:**",
            "Hook:", "Problem/Setup:", "Solution/Development:", 
            "Result/Punchline:", "Call to action:",
            "**Script:**", "Script:"
        ]
        
        # Define patterns to filter out
        special_patterns = ["---", "***", "**", "##"]
        
        clean_script_lines = []
        for line in clean_script.split('\n'):
            line_to_add = line
            skip_line = False
            
            # Skip lines that only contain special characters
            if line.strip() in special_patterns or line.strip("-*#") == "":
                skip_line = True
                continue
            
            # Check if line starts with a number followed by a period (like "1.")
            if re.match(r'^\d+\.', line.strip()):
                # Remove the number prefix
                line_to_add = re.sub(r'^\d+\.\s*', '', line.strip())
            
            for header in section_headers:
                if line.strip().startswith(header):
                    # Extract content after the header
                    content = line[line.find(header) + len(header):].strip()
                    if content:  # If there's content after the header, use it
                        line_to_add = content
                    else:  # If it's just a header line, skip it entirely
                        skip_line = True
                    break
            
            if not skip_line and line_to_add.strip():
                clean_script_lines.append(line_to_add)
        
        clean_script = '\n'.join(clean_script_lines)
        
        # Print the cleaned script
        print(colored(f"Cleaned script for TTS:\n{clean_script}", "blue"))
        
        # Check if the script is empty after cleaning
        if not clean_script.strip():
            print(colored("Warning: Script is empty after cleaning. Using original script.", "yellow"))
            # Use the original script without section headers
            clean_script = script
            # Remove line numbers and section headers
            clean_script = re.sub(r'^\d+\.\s*', '', clean_script)
            for header in section_headers:
                clean_script = clean_script.replace(header, '')
            # Remove special patterns
            for pattern in special_patterns:
                clean_script = clean_script.replace(pattern, '')
            clean_script = clean_script.strip()
            print(colored(f"Fallback script for TTS:\n{clean_script}", "blue"))
        
        return clean_script

    async def _generate_subtitles(self, script: str, tts_path: str, channel_type: str) -> str:
        """Generate subtitles for the video"""
        try:
            print(colored("\n=== Generating Subtitles ===", "blue"))
            
            # Directly use the function from video.py
            subtitles_path = generate_subtitles(
                script=script,
                audio_path=tts_path,
                content_type=channel_type
            )
            
            if not subtitles_path:
                raise ValueError("Failed to generate subtitles file")
            
            print(colored("✓ Subtitles generated successfully", "green"))
            return subtitles_path
            
        except Exception as e:
            print(colored(f"Subtitles generation failed: {str(e)}", "red"))
            return None

    def _format_time(self, seconds):
        """Format time for subtitles"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _analyze_script_content(self, script):
        """Analyze script to understand main topics and context"""
        try:
            # Common words to filter out
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were', 'that', 'this', 'these', 'those'}
            
            # Clean and split script
            script_lower = script.lower()
            sentences = script_lower.split('.')
            
            # Extract main topics and context
            topics = {
                'main_subject': [],    # Primary topic (e.g., 'coffee', 'debugging')
                'actions': [],         # What's happening (e.g., 'drinking', 'coding')
                'environment': [],     # Where it happens (e.g., 'office', 'desk')
                'objects': [],         # Related objects (e.g., 'keyboard', 'monitor')
                'mood': []            # Emotional context (e.g., 'funny', 'serious')
            }
            
            # Mood indicators
            mood_words = {
                'funny': ['joke', 'humor', 'funny', 'laugh', 'twist', 'pun'],
                'serious': ['important', 'serious', 'critical', 'essential'],
                'educational': ['learn', 'understand', 'explain', 'tutorial'],
                'motivational': ['inspire', 'motivation', 'achieve', 'success']
            }
            
            # Process each sentence
            for sentence in sentences:
                words = sentence.replace('!', '').replace('?', '').replace(',', '').split()
                words = [w for w in words if w not in stop_words and len(w) > 3]
                
                # Look for word pairs that might indicate topics
                for i in range(len(words)-1):
                    pair = f"{words[i]} {words[i+1]}"
                    
                    # Check if pair indicates a topic
                    if any(term in pair for term in ['coding', 'programming', 'developer', 'software']):
                        topics['main_subject'].append(pair)
                    elif any(term in pair for term in ['drinking', 'working', 'typing']):
                        topics['actions'].append(pair)
                    elif any(term in pair for term in ['office', 'desk', 'workspace']):
                        topics['environment'].append(pair)
                
                # Check individual words
                for word in words:
                    # Check for objects
                    if any(term in word for term in ['keyboard', 'screen', 'monitor', 'coffee', 'computer']):
                        topics['objects'].append(word)
                    
                    # Check for mood
                    for mood, indicators in mood_words.items():
                        if any(indicator in word for indicator in indicators):
                            topics['mood'].append(mood)

            # Clean up and deduplicate
            for category in topics:
                topics[category] = list(set(topics[category]))

            print(colored(f"Script analysis: {topics}", "blue"))
            return topics

        except Exception as e:
            print(colored(f"Error analyzing script: {str(e)}", "red"))
            return None

    def _generate_search_terms(self, script_analysis):
        """Generate search terms based on script analysis"""
        try:
            search_terms = []
            
            if not script_analysis:
                return ['background video']
            
            # Combine main subject with actions
            for subject in script_analysis['main_subject']:
                for action in script_analysis['actions']:
                    search_terms.append(f"{subject} {action}")
            
            # Combine main subject with environment
            for subject in script_analysis['main_subject']:
                for env in script_analysis['environment']:
                    search_terms.append(f"{subject} {env}")
            
            # Add object-based searches
            for obj in script_analysis['objects']:
                search_terms.append(obj)
                # Combine with environment if available
                for env in script_analysis['environment']:
                    search_terms.append(f"{obj} {env}")
            
            # Add mood-based context
            if script_analysis['mood']:
                mood = script_analysis['mood'][0]  # Use primary mood
                for term in search_terms[:]:  # Search through copy of list
                    search_terms.append(f"{mood} {term}")
            
            # Deduplicate and limit
            search_terms = list(set(search_terms))
            print(colored(f"Generated search terms: {search_terms[:10]}", "blue"))
            return search_terms[:10]

        except Exception as e:
            print(colored(f"Error generating search terms: {str(e)}", "red"))
            return ['background video']

    async def _get_video_suggestions(self, script):
        """Get video suggestions from GPT based on script content"""
        try:
            # Estimate duration based on word count (assuming ~3 words per second)
            word_count = len(script.split())
            estimated_duration = word_count / 3  # Roughly 3 words per second

            # Determine number of video suggestions based on duration
            if estimated_duration <= 15:  # Short videos (~15 sec)
                num_suggestions = 3
            elif estimated_duration <= 60:  # Medium-length videos (~30-60 sec)
                num_suggestions = 4
            else:  # Long videos (60+ sec)
                num_suggestions = 6

            prompt = f"""
            Analyze this script and suggest {num_suggestions} specific video scenes that would match the content well.
            Focus on visual elements that enhance the story.
            Format each suggestion as a clear search term for stock videos.

            Script:
            {script}

            Provide suggestions in this format:
            1. [search term] - [brief explanation why this fits]
            2. [search term] - [brief explanation why this fits]
            etc.

            Example output:
            1. programmer drinking coffee - Shows the main subject of the story
            2. coding workspace setup - Establishes the environment
            3. typing on keyboard closeup - Shows the action
            4. coffee cup steam programming - Creates atmosphere (only if longer script)
            """

            # Use your existing GPT integration
            response = await self.script_generator.generate_with_gpt(prompt)
            
            # Parse the response to extract search terms
            suggestions = []
            for line in response.split('\n'):
                if line.strip() and line[0].isdigit():
                    # Extract the search term before the dash
                    term = line.split('-')[0].strip()
                    # Remove the number and dot from the beginning
                    term = term.split('.', 1)[1].strip()
                    suggestions.append({
                        'term': term,
                        'explanation': line.split('-')[1].strip() if '-' in line else ''
                    })
            
            print(colored("Video suggestions from GPT:", "blue"))
            for suggestion in suggestions:
                print(colored(f"• {suggestion['term']} - {suggestion['explanation']}", "cyan"))
            
            return suggestions

        except Exception as e:
            print(colored(f"Error getting video suggestions: {str(e)}", "red"))
            return None

    async def _process_background_videos(self, channel_type, script=None):
        """Process background videos with improved handling for short videos and fallbacks"""
        try:
            # Create directories if they don't exist
            os.makedirs("assets/videos/categorized", exist_ok=True)
            os.makedirs("temp/backgrounds", exist_ok=True)
            
            # Analyze script to get search terms
            if script:
                script_analysis = self._analyze_script_content(script)
                search_terms = self._generate_search_terms(script_analysis)
                print(colored(f"Generated search terms: {', '.join(search_terms)}", "blue"))
            else:
                search_terms = self._generate_generic_search_terms(channel_type)
                print(colored(f"Using generic search terms: {', '.join(search_terms)}", "blue"))
            
            # Get all available background videos
            background_videos = []
            used_videos = set()  # Track used videos to avoid repetition
            min_video_duration = 8.0  # Minimum acceptable video duration in seconds
            
            # Try each search term
            for term in search_terms:
                # Sanitize the search term for directory name
                term_dir = self._sanitize_filename(term)
                directory = f"assets/videos/categorized/{term_dir}"
                os.makedirs(directory, exist_ok=True)
                
                # Check if we already have videos for this term
                existing_videos = glob.glob(f"{directory}/*.mp4")
                
                # If we don't have enough videos, download more
                if len(existing_videos) < 3:
                    print(colored(f"Downloading more videos for '{term}'...", "blue"))
                    try:
                        # Try to download more videos from Pexels
                        await self._search_and_save_videos(term, directory, count=5)
                        # Update the list of existing videos
                        existing_videos = glob.glob(f"{directory}/*.mp4")
                    except Exception as e:
                        print(colored(f"Error downloading videos: {str(e)}", "yellow"))
                
                # Check each video for usability
                for video_path in existing_videos:
                    # Skip if we've already used this video
                    if video_path in used_videos:
                        continue
                        
                    try:
                        # Check video duration
                        video = VideoFileClip(video_path)
                        duration = video.duration
                        video.close()
                        
                        if duration >= min_video_duration:
                            background_videos.append(video_path)
                            used_videos.add(video_path)
                            print(colored(f"Added background video: {os.path.basename(video_path)} ({duration:.2f}s)", "green"))
                        else:
                            # Instead of skipping short videos, we'll use them but mark them for looping
                            background_videos.append((video_path, "short"))
                            used_videos.add(video_path)
                            print(colored(f"⚠️ Warning: Video too short ({duration:.2f}s), will loop it: {os.path.basename(video_path)}", "yellow"))
                    except Exception as e:
                        print(colored(f"Error checking video {os.path.basename(video_path)}: {str(e)}", "yellow"))
            
            # If we don't have enough usable videos, try to download more with broader terms
            if len(background_videos) < 3:
                broader_terms = [
                    "abstract background", "creative animation", "digital technology",
                    "business professional", "modern lifestyle", "nature scenery"
                ]
                
                # Select terms based on channel type
                if channel_type == "tech_humor":
                    broader_terms.extend(["technology", "computer", "coding", "digital art"])
                elif channel_type == "ai_money":
                    broader_terms.extend(["finance", "business growth", "money", "success"])
                elif channel_type == "baby_tips":
                    broader_terms.extend(["baby", "family", "parenting", "children"])
                elif channel_type == "quick_meals":
                    broader_terms.extend(["cooking", "food preparation", "kitchen", "ingredients"])
                elif channel_type == "fitness_motivation":
                    broader_terms.extend(["workout", "exercise", "fitness", "healthy lifestyle"])
                
                print(colored(f"Not enough usable videos. Trying broader terms: {', '.join(broader_terms[:3])}", "blue"))
                
                for term in broader_terms:
                    if len(background_videos) >= 5:
                        break
                        
                    term_dir = self._sanitize_filename(term)
                    directory = f"assets/videos/categorized/{term_dir}"
                    os.makedirs(directory, exist_ok=True)
                    
                    try:
                        await self._search_and_save_videos(term, directory, count=3)
                        new_videos = glob.glob(f"{directory}/*.mp4")
                        
                        for video_path in new_videos:
                            if video_path not in used_videos:
                                try:
                                    video = VideoFileClip(video_path)
                                    duration = video.duration
                                    video.close()
                                    
                                    if duration >= min_video_duration:
                                        background_videos.append(video_path)
                                        used_videos.add(video_path)
                                        print(colored(f"Added additional background video: {os.path.basename(video_path)} ({duration:.2f}s)", "green"))
                                        if len(background_videos) >= 5:
                                            break
                                    else:
                                        # Use short videos but mark them for looping
                                        background_videos.append((video_path, "short"))
                                        used_videos.add(video_path)
                                        print(colored(f"⚠️ Warning: Additional video too short ({duration:.2f}s), will loop it: {os.path.basename(video_path)}", "yellow"))
                                        if len(background_videos) >= 5:
                                            break
                                except Exception as e:
                                    print(colored(f"Error checking additional video: {str(e)}", "yellow"))
                    except Exception as e:
                        print(colored(f"Error downloading additional videos: {str(e)}", "yellow"))
            
            # If we still don't have enough videos, try to use images as fallback
            if len(background_videos) < 2:
                print(colored("Not enough videos available. Using images as fallback.", "yellow"))
                background_videos = self._get_fallback_images(channel_type, search_terms)
            
            # If we still don't have any usable backgrounds, create a default one
            if not background_videos:
                print(colored("No usable backgrounds found. Creating default background.", "yellow"))
                default_bg = self._create_default_background(channel_type)
                background_videos = [default_bg]
            
            # Shuffle the videos for variety
            random.shuffle(background_videos)
            
            return background_videos
            
        except Exception as e:
            print(colored(f"Error processing background videos: {str(e)}", "red"))
            traceback.print_exc()
            # Create and return a default background as fallback
            return [self._create_default_background(channel_type)]
    
    def _get_fallback_images(self, channel_type, search_terms):
        """Get fallback images when videos aren't available and convert them to video clips"""
        try:
            # Create directory for fallback images
            os.makedirs("assets/images/fallback", exist_ok=True)
            
            background_videos = []
            
            # Try to download images from Pexels or Pixabay
            api_keys = {
                'pexels': os.getenv('PEXELS_API_KEY'),
                'pixabay': os.getenv('PIXABAY_API_KEY')
            }
            
            # Check if we have valid API keys
            valid_apis = [api for api, key in api_keys.items() if key]
            
            if not valid_apis:
                print(colored("No valid API keys found for image sources.", "yellow"))
                return background_videos
            
            # Try each search term
            for term in search_terms[:3]:  # Limit to first 3 terms
                # Try each available API
                for api in valid_apis:
                    try:
                        # Create a unique filename for this image-based video
                        output_path = f"temp/backgrounds/image_bg_{self._sanitize_filename(term)}_{int(time.time())}.mp4"
                        
                        if api == 'pexels':
                            # Download image from Pexels
                            headers = {'Authorization': api_keys['pexels']}
                            response = requests.get(
                                f"https://api.pexels.com/v1/search?query={term}&per_page=15&orientation=portrait",
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data['photos']:
                                    # Get a random image from the results
                                    photo = random.choice(data['photos'])
                                    img_url = photo['src']['large2x']
                                    
                                    # Download the image
                                    img_response = requests.get(img_url)
                                    if img_response.status_code == 200:
                                        img_path = f"assets/images/fallback/{self._sanitize_filename(term)}_{int(time.time())}.jpg"
                                        with open(img_path, 'wb') as f:
                                            f.write(img_response.content)
                                        
                                        print(colored(f"Downloaded image from Pexels: {img_url}", "green"))
                                        
                                        # Convert image to video clip with subtle animation
                                        self._create_video_from_image(img_path, output_path, duration=15)
                                        background_videos.append(output_path)
                                        
                                        if len(background_videos) >= 3:
                                            return background_videos
                        
                        elif api == 'pixabay':
                            # Download image from Pixabay
                            response = requests.get(
                                f"https://pixabay.com/api/?key={api_keys['pixabay']}&q={term}&image_type=photo&orientation=vertical&per_page=15"
                            )
                            
                            if response.status_code == 200:
                                data = response.json()
                                if data['hits']:
                                    # Get a random image from the results
                                    photo = random.choice(data['hits'])
                                    img_url = photo['largeImageURL']
                                    
                                    # Download the image
                                    img_response = requests.get(img_url)
                                    if img_response.status_code == 200:
                                        img_path = f"assets/images/fallback/{self._sanitize_filename(term)}_{int(time.time())}.jpg"
                                        with open(img_path, 'wb') as f:
                                            f.write(img_response.content)
                                        
                                        print(colored(f"Downloaded image from Pixabay: {img_url}", "green"))
                                        
                                        # Convert image to video clip with subtle animation
                                        self._create_video_from_image(img_path, output_path, duration=15)
                                        background_videos.append(output_path)
                                        
                                        if len(background_videos) >= 3:
                                            return background_videos
                    
                    except Exception as e:
                        print(colored(f"Error getting fallback image from {api}: {str(e)}", "yellow"))
            
            return background_videos
            
        except Exception as e:
            print(colored(f"Error getting fallback images: {str(e)}", "red"))
            return []
    
    def _create_video_from_image(self, image_path, output_path, duration=15):
        """Create a video clip from an image with subtle animation"""
        try:
            # Load the image
            img = ImageClip(image_path).set_duration(duration)
            
            # Resize to vertical format (9:16)
            img = img.resize(height=1920)
            
            # Center crop to 9:16 aspect ratio if needed
            if img.w > 1080:
                # Crop from center
                x_center = img.w / 2
                img = img.crop(x1=x_center-540, y1=0, x2=x_center+540, y2=1920)
            
            # Add subtle zoom and pan animation
            def zoom_effect(t):
                # Zoom from 1.0 to 1.05 over the duration
                zoom = 1.0 + (0.05 * t / duration)
                return zoom
            
            img = img.resize(lambda t: zoom_effect(t))
            
            # Write the video file
            img.write_videofile(output_path, fps=30, codec='libx264', audio=False)
            print(colored(f"Created video from image: {output_path}", "green"))
            
            return output_path
            
        except Exception as e:
            print(colored(f"Error creating video from image: {str(e)}", "red"))
            return None

    async def _search_and_save_videos(self, term, directory, count=3):
        """Search and download videos from Pexels with improved error handling"""
        try:
            # Check if we have a Pexels API key
            api_key = os.getenv('PEXELS_API_KEY')
            if not api_key:
                print(colored("No Pexels API key found. Set the PEXELS_API_KEY environment variable.", "yellow"))
                return
            
            # Set up headers for Pexels API
            headers = {'Authorization': api_key}
            
            # Search for videos
            response = requests.get(
                f"https://api.pexels.com/videos/search?query={term}&per_page={count*2}&orientation=portrait",
                headers=headers
            )
            
            if response.status_code != 200:
                print(colored(f"Pexels API error: {response.status_code} - {response.text}", "yellow"))
                return
            
            data = response.json()
            videos = data.get('videos', [])
            
            if not videos:
                print(colored(f"No videos found for term: {term}", "yellow"))
                return
            
            # Shuffle videos for variety
            random.shuffle(videos)
            
            # Download videos
            downloaded = 0
            for video in videos:
                if downloaded >= count:
                    break
                
                # Get the video files
                video_files = video.get('video_files', [])
                
                # Filter for HD or SD quality and portrait orientation
                suitable_files = [
                    f for f in video_files 
                    if (f.get('quality') in ['hd', 'sd']) and 
                    (f.get('width', 0) < f.get('height', 0))  # Portrait orientation
                ]
                
                if not suitable_files:
                    continue
                
                # Sort by quality (HD first, then SD)
                suitable_files.sort(key=lambda x: 0 if x.get('quality') == 'hd' else 1)
                
                # Get the best quality file
                video_file = suitable_files[0]
                video_url = video_file.get('link')
                
                if not video_url:
                    continue
                
                try:
                    # Generate a unique filename
                    video_id = video.get('id', uuid.uuid4())
                    filename = f"{directory}/{video_id}.mp4"
                    
                    # Download the video if it doesn't already exist
                    if not os.path.exists(filename):
                        print(colored(f"Downloading video: {video_url}", "blue"))
                        
                        # Stream the download to handle large files
                        with requests.get(video_url, stream=True) as r:
                            r.raise_for_status()
                            with open(filename, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        
                        print(colored(f"Downloaded video: {os.path.basename(filename)}", "green"))
                        
                        # Verify the video is valid
                        try:
                            video_clip = VideoFileClip(filename)
                            duration = video_clip.duration
                            video_clip.close()
                            
                            if duration < 5:  # Too short
                                print(colored(f"Video too short ({duration:.2f}s), removing", "yellow"))
                                os.remove(filename)
                                continue
                                
                            downloaded += 1
                            
                        except Exception as e:
                            print(colored(f"Invalid video file, removing: {str(e)}", "yellow"))
                            if os.path.exists(filename):
                                os.remove(filename)
                            continue
                    else:
                        # File already exists
                        downloaded += 1
                        print(colored(f"Video already exists: {os.path.basename(filename)}", "blue"))
                
                except Exception as e:
                    print(colored(f"Error downloading video: {str(e)}", "yellow"))
                    continue
            
            print(colored(f"Downloaded {downloaded} videos for term: {term}", "green"))
            
        except Exception as e:
            print(colored(f"Error searching and saving videos: {str(e)}", "red"))
            traceback.print_exc()

    def _sanitize_filename(self, filename):
        """Sanitize a string to be used as a filename"""
        # Replace spaces with underscores and remove invalid characters
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        sanitized = ''.join(c for c in filename if c in valid_chars)
        sanitized = sanitized.replace(' ', '_').lower()
        
        # Ensure the filename is not empty
        if not sanitized:
            sanitized = "unnamed"
            
        return sanitized

    def _generate_generic_search_terms(self, channel_type):
        """Generate generic search terms based on channel type"""
        generic_terms = {
            'tech_humor': ['technology background', 'computer animation', 'digital world', 'tech abstract'],
            'ai_money': ['business growth', 'finance technology', 'digital money', 'success graph'],
            'baby_tips': ['happy baby', 'parenting moments', 'child playing', 'family time'],
            'quick_meals': ['food preparation', 'cooking ingredients', 'kitchen scene', 'healthy food'],
            'fitness_motivation': ['workout session', 'fitness training', 'exercise routine', 'active lifestyle']
        }
        
        return generic_terms.get(channel_type, ['abstract background', 'colorful motion', 'dynamic background'])

    async def _generate_video(self, tts_path, subtitles_path, background_videos, channel_type, custom_output_path=None):
        """Generate a video with the given TTS audio and subtitles"""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join("output", "videos", channel_type)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{channel_type}_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Use custom output path if provided
            if custom_output_path:
                output_path = custom_output_path
            
            # Get audio duration
            audio = AudioFileClip(tts_path)
            audio_duration = audio.duration
            audio.close()
            
            print(colored(f"Original audio duration: {audio_duration:.2f}s", "blue"))
            
            # Calculate video duration with padding
            start_padding = 1.0  # 1 second at start (already added to audio)
            end_padding = 3.0    # 3 seconds at end (increased from 2 to 3 seconds)
            target_duration = audio_duration + end_padding  # Start padding already included in audio
            print(colored(f"Video duration: {target_duration:.2f}s (with {start_padding}s start and {end_padding}s end padding)", "blue"))
            
            # Find the script path based on the TTS path
            # The script is typically stored with the same base name as the TTS file but with .txt extension
            script_path = None
            if tts_path:
                possible_script_path = tts_path.replace('.mp3', '.txt')
                if os.path.exists(possible_script_path):
                    script_path = possible_script_path
                else:
                    # Try looking in the temp/tts directory for a script with the channel name
                    possible_script_path = os.path.join('temp', 'tts', f"{channel_type}.txt")
                    if os.path.exists(possible_script_path):
                        script_path = possible_script_path
            
            # Generate video
            from video import generate_video
            video = generate_video(
                background_path=background_videos,
                audio_path=tts_path,
                subtitles_path=subtitles_path,
                content_type=channel_type,
                target_duration=target_duration,
                use_background_music=self.use_background_music,
                music_volume=self._music_volume,
                music_fade_in=self.music_fade_in,
                music_fade_out=self.music_fade_out,
                script_path=script_path
            )
            
            if not video:
                print(colored("Failed to generate video", "red"))
                return None
                
            # Write final video with specific parameters to prevent black frames
            temp_video_path = f"{output_dir.replace('/videos', '')}/temp_video_no_audio.mp4"
            temp_audio_path = f"{output_dir.replace('/videos', '')}/temp_audio.mp3"
            
            # Use custom output path if provided, otherwise use default
            final_output_path = custom_output_path if custom_output_path else output_path
            
            # Create a temporary output path for the initial video before trimming
            temp_output_path = f"{output_dir}/temp_{timestamp}.mp4"
            
            print(colored(f"\n=== 🎥 Rendering Final Video 🎥 ===", "blue"))
            print(colored(f"ℹ️ Output path: {final_output_path}", "cyan"))
            
            # Show a message about rendering time
            print(colored("⏳ Rendering final video... This may take a while", "cyan"))
            start_time = time.time()
            
            # Step 1: Extract the audio - handle both simple AudioFileClip and CompositeAudioClip
            if hasattr(video, 'audio') and video.audio is not None:
                try:
                    # Check if it's a CompositeAudioClip (which doesn't have fps)
                    if isinstance(video.audio, CompositeAudioClip):
                        # CompositeAudioClip doesn't have fps, so we need to set it
                        # Get fps from the first clip in the composite
                        if hasattr(video.audio.clips[0], 'fps') and video.audio.clips[0].fps:
                            # Use the fps from the first clip
                            fps = video.audio.clips[0].fps
                            # Create a new audio file with the same content but with fps
                            temp_mixed_path = f"{self.temp_dir}/temp_mixed_{uuid.uuid4()}.mp3"
                            video.audio.write_audiofile(temp_mixed_path, fps=44100)
                            # Now load it as a regular AudioFileClip which has fps
                            regular_audio = AudioFileClip(temp_mixed_path)
                            regular_audio.write_audiofile(temp_audio_path)
                            regular_audio.close()
                            # Clean up
                            if os.path.exists(temp_mixed_path):
                                os.remove(temp_mixed_path)
                        else:
                            # Fallback to using the original TTS file
                            print(colored("⚠️ Warning: Could not determine fps for composite audio, using original TTS", "yellow"))
                            source_audio = AudioFileClip(tts_path)
                            source_audio.write_audiofile(temp_audio_path)
                            source_audio.close()
                    else:
                        # It's a regular AudioFileClip
                        video.audio.write_audiofile(temp_audio_path)
                except Exception as e:
                    print(colored(f"Error extracting audio: {str(e)}", "red"))
                    print(colored("Falling back to original TTS audio", "yellow"))
                    # Fallback to using the original TTS file
                    source_audio = AudioFileClip(tts_path)
                    source_audio.write_audiofile(temp_audio_path)
                    source_audio.close()
            else:
                # If no audio in video, use the original TTS file
                source_audio = AudioFileClip(tts_path)
                source_audio.write_audiofile(temp_audio_path)
                source_audio.close()
                print(colored("⚠️ Warning: No audio found in video, using original TTS without music", "yellow"))
            
            # Step 2: Write the video without audio
            video.without_audio().write_videofile(
                temp_video_path,
                codec='libx264',
                fps=30,
                preset='ultrafast',
                ffmpeg_params=["-vf", "format=yuv420p"]
            )
            
            # Step 3: Combine video and audio using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_path,
                "-i", temp_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",  # Use the entire video track
                "-map", "1:a:0",  # Use the entire audio track
                temp_output_path
            ]
            subprocess.run(cmd, check=True)
            
            # Step 4: Instead of trimming, just copy the file to the final output
            print(colored("⏳ Finalizing video...", "cyan"))
            
            # Get the duration of the generated video
            probe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                temp_output_path
            ]
            
            try:
                # Get the duration of the video
                duration_output = subprocess.check_output(probe_cmd, universal_newlines=True).strip()
                video_duration = float(duration_output)
                
                # Simply copy the file instead of trimming
                shutil.copy(temp_output_path, final_output_path)
                print(colored(f"✅ Successfully created video with duration {video_duration:.2f}s", "green"))
                
                # Remove the temporary output file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                
            except Exception as copy_error:
                print(colored(f"⚠️ Warning: Error copying video: {str(copy_error)}", "yellow"))
                print(colored("Using the temporary video as final output", "yellow"))
                # If copying fails, just use the temp file as the final output
                if os.path.exists(temp_output_path) and not os.path.exists(final_output_path):
                    try:
                        shutil.copy(temp_output_path, final_output_path)
                    except Exception:
                        pass
            
            elapsed_time = time.time() - start_time
            print(colored(f"⏱️ Video rendered in {elapsed_time:.1f} seconds", "cyan"))
            
            # Clean up
            print(colored("ℹ️ Cleaning up resources...", "cyan"))
            try:
                video.close()
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                print(colored("✓ Temporary files cleaned up", "green"))
            except Exception as e:
                print(colored(f"Warning: Error cleaning up: {str(e)}", "yellow"))
            
            return final_output_path
            
        except Exception as e:
            print(colored(f"Error generating video: {str(e)}", "red"))
            traceback.print_exc()
            return None

    async def generate_video(self, channel, script_file):
        """Generate a video for the given channel using the script file"""
        try:
            # Use the existing create_video method
            output_path = f"output/videos/{channel}_latest.mp4"
            return await self.create_video(script_file, channel, output_path)
        except Exception as e:
            print(colored(f"Error in generate_video: {str(e)}", "red"))
            traceback.print_exc()
        return False

    @property
    def music_volume(self):
        """Get the current music volume setting"""
        return self._music_volume
        
    @music_volume.setter
    def music_volume(self, value):
        """Set the music volume, ensuring it's within valid range"""
        if not isinstance(value, (int, float)):
            self.logger.warning(f"Invalid music volume type: {type(value)}. Using default 0.3")
            self._music_volume = 0.3
        elif value < 0.0 or value > 1.0:
            self.logger.warning(f"Music volume {value} out of range (0.0-1.0). Clamping to valid range.")
            self._music_volume = max(0.0, min(1.0, value))
        else:
            self._music_volume = value
            print(colored(f"Music volume set to {self._music_volume}", "cyan"))

    def _cleanup_temp_files(self):
        """Clean up temporary files created during video generation"""
        try:
            print(colored("ℹ️ Cleaning up resources...", "blue"))
            temp_dirs = ["temp", "temp/segments"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        try:
                            file_path = os.path.join(temp_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                        except Exception as e:
                            print(colored(f"Warning: Error cleaning up: {str(e)}", "yellow"))
        except Exception as e:
            print(colored(f"Warning: Error during cleanup: {str(e)}", "yellow"))
            
    def cleanup_video_library(self, channel_type, max_videos=20, days_to_keep=30):
        """Clean up video library to prevent excessive accumulation
        
        Args:
            channel_type (str): The channel type to clean up
            max_videos (int): Maximum number of videos to keep per channel
            days_to_keep (int): Always keep videos newer than this many days
        """
        try:
            print(colored(f"ℹ️ Cleaning up video library for {channel_type}...", "blue"))
            
            # Define the output directory for this channel
            output_dir = os.path.join("output", "videos", channel_type)
            if not os.path.exists(output_dir):
                return
                
            # Get all video files for this channel
            video_files = []
            for file in os.listdir(output_dir):
                if file.endswith(".mp4") and not file.startswith("temp_"):
                    file_path = os.path.join(output_dir, file)
                    # Get file creation time
                    creation_time = os.path.getctime(file_path)
                    video_files.append((file_path, creation_time))
            
            # Sort by creation time (newest first)
            video_files.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate the cutoff date for keeping videos
            import time
            from datetime import datetime, timedelta
            cutoff_date = time.time() - (days_to_keep * 24 * 60 * 60)
            
            # Keep track of how many videos we're keeping
            kept_videos = 0
            
            # Process each video
            for file_path, creation_time in video_files:
                # Always keep videos newer than the cutoff date
                if creation_time >= cutoff_date:
                    kept_videos += 1
                    continue
                    
                # Keep up to max_videos
                if kept_videos < max_videos:
                    kept_videos += 1
                    continue
                    
                # Delete older videos beyond our limit
                try:
                    os.remove(file_path)
                    print(colored(f"Removed old video: {os.path.basename(file_path)}", "yellow"))
                except Exception as e:
                    print(colored(f"Error removing old video: {str(e)}", "yellow"))
                    
            print(colored(f"Kept {kept_videos} videos for {channel_type}", "green"))
            
        except Exception as e:
            print(colored(f"Error cleaning up video library: {str(e)}", "yellow"))

