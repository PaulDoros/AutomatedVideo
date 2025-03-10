from moviepy.editor import ImageClip
import pysrt
from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
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
            
            # Process script into sentences for better TTS quality
            print(colored(f"Processing {len(text.split('.'))} sentences for better TTS quality", "blue"))
            
            # Generate TTS for each sentence separately for better quality
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            audio_files = []
            
            for i, sentence in enumerate(sentences):
                # Add period back if it was removed by split
                if not sentence.endswith(('!', '?', '.')):
                    sentence += '.'
                
                # Add slight pause between sentences
                if i > 0:
                    sentence = f"<break time='0.2s'/> {sentence}"
                
                # Generate audio for this sentence
                sentence_path = f"temp/tts/sentence_{i}_{int(time.time())}.wav"
                result = await self.voice_diversifier.api.generate_voice(
                    text=sentence,
                    speaker=voice,
                    language="en",
                    emotion=emotion,
                    speed=speed,
                    output_path=sentence_path
                )
                
                if result:
                    audio_files.append(result)
                else:
                    print(colored(f"Failed to generate TTS for sentence: {sentence}", "red"))
            
            # Combine all sentence audio files
            if audio_files:
                # Concatenate audio files
                from pydub import AudioSegment
                combined = AudioSegment.empty()
                for audio_file in audio_files:
                    segment = AudioSegment.from_file(audio_file)
                    combined += segment
                
                # Export combined audio
                combined.export(output_path, format="wav")
                
                print(colored(f"✓ Generated voice file: {output_path}", "green"))
                return output_path
            
            return None
            
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
                    content = f.read().strip()
                    
                    # Try to parse as JSON first
                    try:
                        script_data = json.loads(content)
                        if isinstance(script_data, dict):
                            script = script_data.get('script', '')
                        else:
                            script = str(script_data)
                    except json.JSONDecodeError:
                        # If not valid JSON, use the content as is
                        script = content
                    
                    # Check if we have a plain text version of the script (which might be better formatted)
                    txt_script_file = script_file.replace('.json', '.txt')
                    if os.path.exists(txt_script_file):
                        try:
                            with open(txt_script_file, 'r', encoding='utf-8') as f:
                                txt_script = f.read().strip()
                                if txt_script:
                                    print(colored("Using plain text version of script for better formatting", "green"))
                                    script = txt_script
                        except Exception as e:
                            print(colored(f"Error loading plain text script: {str(e)}", "yellow"))
                            # Continue with the JSON script
                    
                    # Print the script being used
                    print(colored(f"Script to be used for video generation:\n{script}", "blue"))
                    
                    # Check if script is empty
                    if not script.strip():
                        print(colored("Error: Script is empty", "red"))
                        return False
                    
            except Exception as e:
                print(colored(f"Error loading script: {str(e)}", "red"))
                return False
            
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
                    # Select appropriate voice for the channel type
                    voice, emotion = self.audio_manager.select_coqui_voice(channel_type)
                    print(colored(f"Selected Coqui voice: {voice} (emotion: {emotion})", "green"))
                    
                    # Apply speed multiplier based on content type and emotion
                    speed_multiplier = 1.0
                    if channel_type == 'tech_humor':
                        # Faster for tech humor content
                        speed_multiplier = 1.3
                        print(colored(f"Using speed multiplier of {speed_multiplier} for {channel_type} content", "blue"))
                    elif emotion in ["energetic", "enthusiastic", "playful"]:
                        # Faster for energetic emotions
                        speed_multiplier = 1.25
                        print(colored(f"Using speed multiplier of {speed_multiplier} for {emotion} {channel_type} content", "blue"))
                    
                    # Generate TTS using Coqui
                    tts_path = await self.audio_manager.generate_tts_coqui(
                        clean_script, 
                        voice=voice, 
                        emotion=emotion,
                        speed=speed_multiplier
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
        """Process background videos for the channel - now with smart video management"""
        try:
            print(colored("\n=== Processing Background Videos ===", "blue"))
            
            # Create directory if it doesn't exist
            video_dir = os.path.join("assets", "videos", channel_type)
            os.makedirs(video_dir, exist_ok=True)
            
            # Create a directory for categorized videos
            categorized_dir = os.path.join("assets", "videos", "categorized")
            os.makedirs(categorized_dir, exist_ok=True)
            
            # Number of videos to use for a more dynamic video
            target_video_count = 4  # We want to use 4 videos for a more dynamic experience
            
            # Initialize final video list
            final_videos = []
            
            # Step 1: Analyze script to identify key themes and categories
            script_analysis = self._analyze_script_content(script) if script else {}
            
            # Extract main categories from script analysis
            categories = []
            if 'main_subject' in script_analysis:
                categories.extend(script_analysis['main_subject'])
            if 'objects' in script_analysis:
                categories.extend(script_analysis['objects'])
            if 'environment' in script_analysis:
                categories.extend(script_analysis['environment'])
            
            # Add channel type as a category
            categories.append(channel_type)
            
            # Remove duplicates and empty strings
            categories = [cat for cat in list(dict.fromkeys(categories)) if cat]
            
            print(colored(f"Identified categories: {', '.join(categories)}", "blue"))
            
            # Step 2: Check for categorized videos first
            categorized_videos = []
            for category in categories:
                category_dir = os.path.join(categorized_dir, self._sanitize_filename(category))
                if os.path.exists(category_dir):
                    category_videos = [os.path.join(category_dir, f) for f in os.listdir(category_dir) 
                                     if f.endswith(('.mp4', '.mov')) and os.path.getsize(os.path.join(category_dir, f)) > 0]
                    
                    # Verify each video is valid
                    for video_path in category_videos:
                        try:
                            video = VideoFileClip(video_path)
                            if video.duration >= 3:  # At least 3 seconds
                                categorized_videos.append(video_path)
                            video.close()
                        except Exception as e:
                            print(colored(f"Error checking video {video_path}: {str(e)}", "yellow"))
                    
            if categorized_videos:
                print(colored(f"Found {len(categorized_videos)} relevant categorized videos", "green"))
                # If we have more than needed, randomly select some
                if len(categorized_videos) > target_video_count:
                    selected = random.sample(categorized_videos, target_video_count)
                    final_videos.extend(selected)
                    print(colored(f"Selected {len(selected)} categorized videos", "green"))
                else:
                    final_videos.extend(categorized_videos)
            
            # Step 3: Check channel-specific videos if we need more
            if len(final_videos) < target_video_count:
                remaining_slots = target_video_count - len(final_videos)
                
                # Check if we have local videos for this channel
                channel_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                               if f.endswith(('.mp4', '.mov')) and os.path.getsize(os.path.join(video_dir, f)) > 0]
                
                # Verify each video is valid
                valid_channel_videos = []
                for video_path in channel_videos:
                    try:
                        video = VideoFileClip(video_path)
                        if video.duration >= 3:  # At least 3 seconds
                            valid_channel_videos.append(video_path)
                        video.close()
                    except Exception as e:
                        print(colored(f"Error checking video {video_path}: {str(e)}", "yellow"))
                
                if valid_channel_videos:
                    # If we have more than needed, randomly select some
                    if len(valid_channel_videos) > remaining_slots:
                        selected = random.sample(valid_channel_videos, remaining_slots)
                        final_videos.extend(selected)
                        print(colored(f"Added {len(selected)} channel-specific videos", "green"))
                    else:
                        final_videos.extend(valid_channel_videos)
                        print(colored(f"Added {len(valid_channel_videos)} channel-specific videos", "green"))
            
            # Step 4: Download new videos if needed
            if len(final_videos) < target_video_count and script:
                remaining_slots = target_video_count - len(final_videos)
                print(colored(f"Need {remaining_slots} more videos, downloading from Pexels", "blue"))
                
                # Get video suggestions from GPT
                suggestions = await self._get_video_suggestions(script)
                
                if suggestions:
                    # Generate search terms from suggestions and script analysis
                    search_terms = []
                    
                    # Add GPT suggestions first (they're usually better)
                    for suggestion in suggestions:
                        if isinstance(suggestion, dict) and 'term' in suggestion:
                            term = suggestion['term']
                            if term and len(term) > 3:
                                search_terms.append(term)
                        elif isinstance(suggestion, str):
                            if suggestion.startswith('**') and suggestion.endswith('**'):
                                term = suggestion.strip('*').strip()
                                if term and len(term) > 3:
                                    search_terms.append(term)
                    
                    # Add terms from script analysis as backup
                    analysis_terms = self._generate_search_terms(script_analysis)
                    search_terms.extend(analysis_terms)
                    
                    # Deduplicate terms
                    search_terms = list(dict.fromkeys(search_terms))
                    
                    # Limit to remaining slots
                    search_terms = search_terms[:remaining_slots]
                    
                    # Search for videos and save them to categorized directories
                    for term in search_terms:
                        # Create category directory
                        category_dir = os.path.join(categorized_dir, self._sanitize_filename(term))
                        os.makedirs(category_dir, exist_ok=True)
                        
                        # Download video
                        videos = await self._search_and_save_videos(term, category_dir, count=1)
                        if videos:
                            final_videos.extend(videos)
                            print(colored(f"Downloaded and categorized video for '{term}'", "green"))
                        
                        # If we have enough videos, stop downloading
                        if len(final_videos) >= target_video_count:
                            break
                    
            # Step 5: If we still don't have enough videos, use generic search terms
            if len(final_videos) < 2 and script:
                print(colored("Not enough videos, searching for more from Pexels", "yellow"))
                
                # Generate more search terms
                more_terms = self._generate_generic_search_terms(channel_type)
                
                # Search for videos
                for term in more_terms:
                    # Create category directory
                    category_dir = os.path.join(categorized_dir, self._sanitize_filename(term))
                    os.makedirs(category_dir, exist_ok=True)
                    
                    # Download video
                    videos = await self._search_and_save_videos(term, category_dir, count=1)
                    if videos:
                        final_videos.extend(videos)
                        print(colored(f"Downloaded and categorized video for '{term}'", "green"))
                    
                    # If we have enough videos, stop searching
                    if len(final_videos) >= 2:
                        break
            
            # If we have at least 2 videos, return them
            if len(final_videos) >= 2:
                print(colored(f"Using {len(final_videos)} videos for a dynamic video", "green"))
                return final_videos
            
            # If all else fails, create a default background
            print(colored("No videos found, creating default background", "yellow"))
            return self._create_default_background(channel_type)
            
        except Exception as e:
            print(colored(f"Error processing background videos: {str(e)}", "red"))
            traceback.print_exc()
            # Create a default background as fallback
            return self._create_default_background(channel_type)
    
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

    async def _search_and_save_videos(self, term, directory, count):
        """Helper function to search and save videos"""
        try:
            urls = search_for_stock_videos(
                query=term,
                api_key=self.pexels_api_key,
                it=count,
                min_dur=3
            )
            
            saved_paths = []
            for url in urls:
                saved_path = save_video(url, directory)
                if saved_path:
                    saved_paths.append(saved_path)
            
            return saved_paths
        except Exception as e:
            print(colored(f"Search failed for '{term}': {str(e)}", "yellow"))
            return []

    def _create_default_background(self, channel_type, duration=15):
        """Create a default background video if no suitable videos are found"""
        try:
            print(colored("\n=== Creating Default Background Video ===", "blue"))
            
            # Create directory if it doesn't exist
            os.makedirs("temp/backgrounds", exist_ok=True)
            
            # Generate a unique filename
            default_path = f"temp/backgrounds/default_{channel_type}_{int(time.time())}.mp4"
            
            # Create a gradient background based on channel type
            from thumbnail_generator import ThumbnailGenerator
            thumbnail_gen = ThumbnailGenerator()
            
            # Select colors based on channel type
            colors = {
                'tech_humor': ['#3a1c71', '#d76d77'],  # Purple gradient
                'ai_money': ['#4e54c8', '#8f94fb'],    # Blue gradient
                'baby_tips': ['#11998e', '#38ef7d'],   # Green gradient
                'quick_meals': ['#e1eec3', '#f05053'], # Yellow-red gradient
                'fitness_motivation': ['#355c7d', '#6c5b7b']  # Cool blue-purple gradient
            }
            
            # Get colors for this channel or use default
            gradient_colors = colors.get(channel_type, ['#2b5876', '#4e4376'])
            
            # Create gradient image
            gradient = thumbnail_gen.create_gradient(gradient_colors)
            
            # Convert to numpy array for MoviePy
            gradient_array = np.array(gradient)
            
            # Create a clip with the gradient
            clip = ImageClip(gradient_array).set_duration(duration)
            
            # Resize to vertical format (9:16)
            clip = clip.resize((1080, 1920))
            
            # Add subtle animation (zoom effect)
            # Use a simpler approach without lambda functions
            clip = clip.resize(1.05)  # 5% zoom
            
            # Add a subtle text label based on channel type
            channel_labels = {
                'tech_humor': "Tech Tips",
                'ai_money': "AI Insights",
                'baby_tips': "Parenting Tips",
                'quick_meals': "Quick Recipe",
                'fitness_motivation': "Fitness Tips"
            }
            
            label = channel_labels.get(channel_type, "Tips & Tricks")
            
            # Create text clip
            txt_clip = TextClip(
                label,
                fontsize=70,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2,
                method='label'  # Faster than 'caption'
            )
            
            # Position text at the bottom
            txt_clip = txt_clip.set_position(('center', 1700)).set_duration(duration)
            
            # Combine clips
            final_clip = CompositeVideoClip([clip, txt_clip], size=(1080, 1920))
            
            # Detect number of CPU cores for optimal threading
            cpu_count = multiprocessing.cpu_count()
            optimal_threads = max(4, min(cpu_count - 1, 8))  # Use at least 4 threads, but leave 1 core free
            
            # Write video file with optimized settings
            final_clip.write_videofile(
                default_path,
                codec='libx264',
                fps=30,
                preset='faster',  # Use faster preset for better performance
                audio=False,
                ffmpeg_params=["-shortest", "-avoid_negative_ts", "1"],  # Prevent warnings
                threads=optimal_threads
            )
            
            # Clean up
            try:
                clip.close()
                txt_clip.close()
                final_clip.close()
            except:
                pass  # Ignore errors during cleanup
            
            print(colored("Created default background video", "yellow"))
            return [default_path]
            
        except Exception as e:
            print(colored(f"Error creating default background: {str(e)}", "red"))
            return None

    def create_section_clip(self, index, title, content, total_sections):
        """Create a clip for a section (runs in parallel)"""
        try:
            print(colored(f"\nProcessing section {index}/{total_sections}", "yellow"))
            
            # Calculate duration based on content length
            duration = min(max(len(content.split()) / 2, 3), 10)
            
            # Create background (using numpy for speed)
            bg_array = np.zeros((1920, 1080, 3), dtype=np.uint8)
            bg = ColorClip(bg_array, duration=duration)
            
            # Create text clips
            clips = [bg]
            
            if title:
                title_clip = TextClip(
                    title,
                    fontsize=80,
                    color='white',
                    font='Montserrat-Bold',
                    size=(1000, None),
                    method='label'  # Faster than 'caption'
                )
                title_clip = title_clip.set_position(('center', 200)).set_duration(duration)
                clips.append(title_clip)
            
            content_clip = TextClip(
                content,
                fontsize=60,
                color='white',
                font='Montserrat-Bold',
                size=(900, None),
                method='label'  # Faster than 'caption'
            )
            content_clip = content_clip.set_position('center').set_duration(duration)
            clips.append(content_clip)
            
            # Combine clips
            scene = CompositeVideoClip(clips)
            print(colored(f"✓ Section {index} complete (duration: {duration:.1f}s)", "green"))
            
            return scene, duration
            
        except Exception as e:
            print(colored(f"Error in section {index}: {str(e)}", "red"))
            return None, 0

    def split_into_sections(self, script):
        """Split script into sections, handling different formats"""
        sections = []
        lines = script.split('\n')
        current_title = ""
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section markers
            if line.startswith('[') and line.endswith(']'):
                # Save previous section if exists
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                    current_content = []
                current_title = line.strip('[]')
            elif line.startswith('**') and line.endswith('**'):
                # Alternative section marker
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                    current_content = []
                current_title = line.strip('*')
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections

    def _cleanup_temp_files(self):
        """Clean up temporary files after video generation"""
        try:
            # Clean up temporary files
            for file in glob.glob("temp/segments/*.mp4"):
                try:
                    os.remove(file)
                except Exception as e:
                    print(colored(f"Warning: Could not remove {file}: {str(e)}", "yellow"))
            
            for file in glob.glob("temp/tts/sentence_*.wav"):
                try:
                    os.remove(file)
                except Exception as e:
                    print(colored(f"Warning: Could not remove {file}: {str(e)}", "yellow"))
                    
            print(colored("✓ Temporary files cleaned up", "green"))
            
        except Exception as e:
            print(colored(f"Warning: Error cleaning up temporary files: {str(e)}", "yellow"))

    def cleanup_video_library(self, channel_type=None, max_videos=20, days_to_keep=30):
        """
        Clean up the video library to prevent excessive accumulation of videos.
        
        Args:
            channel_type: Specific channel to clean up, or None for all channels
            max_videos: Maximum number of videos to keep per channel/category
            days_to_keep: Keep videos newer than this many days regardless of count
        """
        try:
            # Clean up channel-specific videos
            channels = [channel_type] if channel_type else ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
            
            for channel in channels:
                video_dir = os.path.join("assets", "videos", channel)
                if os.path.exists(video_dir):
                    self._cleanup_directory(video_dir, max_videos, days_to_keep)
            
            # Clean up categorized videos
            categorized_dir = os.path.join("assets", "videos", "categorized")
            if os.path.exists(categorized_dir):
                # Get all category directories
                categories = [d for d in os.listdir(categorized_dir) 
                             if os.path.isdir(os.path.join(categorized_dir, d))]
                
                for category in categories:
                    category_dir = os.path.join(categorized_dir, category)
                    self._cleanup_directory(category_dir, max_videos, days_to_keep)
                
                print(colored(f"Cleaned up {len(categories)} video categories", "green"))
                
        except Exception as e:
            print(colored(f"Warning: Error cleaning up video library: {str(e)}", "yellow"))
            traceback.print_exc()
    
    def _cleanup_directory(self, directory, max_videos=20, days_to_keep=30):
        """Helper method to clean up a specific directory of videos"""
        try:
            # Get all videos in the directory
            videos = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.endswith(('.mp4', '.mov')) and os.path.getsize(os.path.join(directory, f)) > 0]
            
            # If we have fewer videos than the maximum, no need to clean up
            if len(videos) <= max_videos:
                print(colored(f"Video library for {os.path.basename(directory)} is within limits ({len(videos)}/{max_videos})", "green"))
                return
            
            print(colored(f"Cleaning up video library for {os.path.basename(directory)} ({len(videos)} videos, keeping max {max_videos})", "blue"))
            
            # Get video info with creation time
            video_info = []
            cutoff_date = time.time() - (days_to_keep * 24 * 60 * 60)  # Convert days to seconds
            
            for video_path in videos:
                # Get creation time
                creation_time = os.path.getctime(video_path)
                # Add to list
                video_info.append({
                    'path': video_path,
                    'created': creation_time,
                    'keep': creation_time > cutoff_date  # Keep if newer than cutoff
                })
            
            # Sort by creation time (oldest first)
            video_info.sort(key=lambda x: x['created'])
            
            # Count videos to keep (those marked as keep)
            keep_count = sum(1 for v in video_info if v['keep'])
            
            # If we're keeping more than max_videos due to age, adjust max_videos
            if keep_count > max_videos:
                print(colored(f"Keeping {keep_count} videos for {os.path.basename(directory)} due to age restriction", "yellow"))
                return
            
            # Calculate how many videos to delete
            delete_count = len(videos) - max(max_videos, keep_count)
            
            if delete_count <= 0:
                print(colored(f"No videos need to be deleted for {os.path.basename(directory)}", "green"))
                return
            
            # Delete oldest videos that aren't marked to keep
            deleted = 0
            for video in video_info:
                if deleted >= delete_count:
                    break
                    
                if not video['keep']:
                    try:
                        os.remove(video['path'])
                        deleted += 1
                        print(colored(f"Deleted old video: {os.path.basename(video['path'])}", "yellow"))
                    except Exception as e:
                        print(colored(f"Warning: Could not remove {video['path']}: {str(e)}", "yellow"))
            
            print(colored(f"Cleaned up {deleted} videos for {os.path.basename(directory)}", "green"))
            
        except Exception as e:
            print(colored(f"Warning: Error cleaning up directory {directory}: {str(e)}", "yellow"))

    async def _generate_delayed_subtitles(self, script, audio_path, channel_type, delay=1.0):
        """Generate subtitles with a delay for padding"""
        try:
            print(colored(f"\n=== Generating Subtitles with {delay}s Delay ===", "blue"))
            
            # First generate normal subtitles
            temp_subs_path = generate_subtitles(
                script=script,
                audio_path=audio_path,
                content_type=channel_type
            )
            
            if not temp_subs_path:
                raise ValueError("Failed to generate base subtitles")
            
            # Now shift all subtitles by the delay
            delayed_subs_path = f"temp/subtitles/delayed_{uuid.uuid4()}.srt"
            
            # Read original subtitles
            subs = pysrt.open(temp_subs_path)
            
            # Shift all subtitles by the delay amount
            for sub in subs:
                sub.start.seconds += delay
                sub.end.seconds += delay
            
            # Save shifted subtitles
            subs.save(delayed_subs_path, encoding='utf-8-sig')
            
            print(colored(f"✓ Generated subtitles with {delay}s delay", "green"))
            return delayed_subs_path
            
        except Exception as e:
            print(colored(f"Delayed subtitles generation failed: {str(e)}", "red"))
            return None

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
            
            # Step 3: Combine video and audio using ffmpeg directly
            # IMPORTANT: Removed the -shortest flag to ensure the full video duration is preserved
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_path,
                "-i", temp_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",  # Use the entire video track
                "-map", "1:a:0",  # Use the entire audio track
                "-af", "afade=t=out:st=" + str(audio_duration - 0.5) + ":d=0.5",  # Add fade out to audio
                temp_output_path
            ]
            subprocess.run(cmd, check=True)
            
            # Step 4: Trim the last second from the video
            print(colored("⏳ Trimming the last second from the video...", "cyan"))
            
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
                
                # Calculate the new duration (trim the last second)
                new_duration = max(1.0, video_duration - 1.0)
                
                # Trim the video using ffmpeg
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_output_path,
                    "-t", str(new_duration),
                    "-c:v", "copy",
                    "-c:a", "copy",
                    final_output_path
                ]
                
                subprocess.run(trim_cmd, check=True)
                print(colored(f"✅ Successfully trimmed video from {video_duration:.2f}s to {new_duration:.2f}s", "green"))
                
                # Remove the temporary output file
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                
            except Exception as trim_error:
                print(colored(f"⚠️ Warning: Error trimming video: {str(trim_error)}", "yellow"))
                print(colored("Using the original video without trimming", "yellow"))
                
                # If trimming fails, just use the original output
                if os.path.exists(temp_output_path):
                    # Rename the temp file to the final output path
                    if os.path.exists(final_output_path):
                        os.remove(final_output_path)
                    os.rename(temp_output_path, final_output_path)
            
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

