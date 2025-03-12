from moviepy.editor import ImageClip
import pysrt
from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
import random
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
        """Initialize the audio manager"""
        from video import log_info, log_success, log_error, log_warning
        
        # TTS models
        self.tts_model = None
        self.tts_synthesizer = None
        
        # Store speaker information once initialized
        self.english_speakers = []
        self.male_speakers = []
        self.female_speakers = []
        self.all_speakers = []
        self.available_languages = []
        self.tts_initialized = False
        
        # Voice configuration
        self.voices = {
            'default': {
                'style': 'neutral',
                'voices': {
                    'nova': {'weight': 1},
                    'alloy': {'weight': 1},
                    'echo': {'weight': 1},
                    'fable': {'weight': 1},
                    'onyx': {'weight': 1},
                    'shimmer': {'weight': 1}
                }
            },
            'tech_humor': {
                'style': 'humorous',
                'voices': {
                    'echo': {'weight': 3},
                    'onyx': {'weight': 2},
                    'alloy': {'weight': 1}
                }
            },
            'ai_money': {
                'style': 'professional',
                'voices': {
                    'onyx': {'weight': 3},
                    'alloy': {'weight': 2},
                    'echo': {'weight': 1}
                }
            },
            'baby_tips': {
                'style': 'warm',
                'voices': {
                    'nova': {'weight': 3},
                    'shimmer': {'weight': 2},
                    'fable': {'weight': 1}
                }
            },
            'quick_meals': {
                'style': 'enthusiastic',
                'voices': {
                    'shimmer': {'weight': 3},
                    'nova': {'weight': 2},
                    'fable': {'weight': 1}
                }
            },
            'fitness_motivation': {
                'style': 'motivational',
                'voices': {
                    'echo': {'weight': 3},
                    'onyx': {'weight': 2},
                    'alloy': {'weight': 1}
                }
            }
        }
        
        # Emotion mapping for Coqui voices
        self.emotion_map = {
            'tech_humor': ['humorous', 'cheerful', 'excited', 'surprised'],
            'ai_money': ['professional', 'confident', 'serious', 'energetic'],
            'baby_tips': ['warm', 'friendly', 'calm', 'cheerful'],
            'quick_meals': ['energetic', 'cheerful', 'excited', 'friendly'],
            'fitness_motivation': ['energetic', 'excited', 'professional', 'confident'],
            'default': ['neutral', 'professional', 'friendly']
        }
        
        # Initialize voice diversification system
        self.voice_language_map = {
            # Default language mapping for voices
            "default": "en"
        }
        
        # Initialize OpenAI client if API key is available
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Try to initialize TTS model if possible
        try:
            self._initialize_tts_model()
        except Exception as e:
            print(colored(f"⚠️ Could not initialize TTS model during startup: {str(e)}", "yellow"))
            print(colored(f"ℹ️ TTS model will be initialized on first use", "blue"))
    
    def _initialize_tts_model(self):
        """Initialize the TTS model and classify speakers (only once)"""
        from video import log_info, log_success, log_error, log_warning
        
        # Skip if already initialized
        if self.tts_initialized:
            return True
            
        try:
            log_info("Initializing Coqui TTS (may take a moment)")
            from TTS.api import TTS
            import torch
            
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log_info(f"Using device: {device} for TTS")
            
            # Use a faster model for better performance
            self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self.tts_synthesizer = self.tts_model
            
            # Store all available speakers
            if hasattr(self.tts_model, "speakers") and self.tts_model.speakers:
                self.all_speakers = self.tts_model.speakers
                log_info(f"Found {len(self.all_speakers)} available speakers")
                
                # Identify English speakers (based on our analysis)
                # First, look for speakers with English in their name
                for speaker in self.all_speakers:
                    if any(en_indicator in speaker.lower() for en_indicator in ["en_", "en-", "english"]):
                        self.english_speakers.append(speaker)
                
                # If no obvious English speakers found, look for ones with "en" language
                if not self.english_speakers and hasattr(self.tts_model, "languages") and "en" in self.tts_model.languages:
                    # Based on our analysis, try these specific speakers that seem to handle English well
                    english_candidates = [
                        "Ana Florence", "Brenda Stern", "Henriette Usha", "Sofia Hellen",
                        "Damien Black", "Viktor Menelaos", "Zofija Kendrick", "Baldur Sanjin"
                    ]
                    
                    for candidate in english_candidates:
                        if candidate in self.all_speakers:
                            self.english_speakers.append(candidate)
                
                # If we still don't have English speakers, treat all speakers as potential English speakers
                # since the model supports English language
                if not self.english_speakers and "en" in self.tts_model.languages:
                    log_info("No specific English speakers found, using any speaker with English language")
                    self.english_speakers = self.all_speakers
                
                log_info(f"Found {len(self.english_speakers)} suitable English speakers")
                
                # Simple gender classification - better than before but still basic
                for speaker in self.all_speakers:
                    speaker_lower = speaker.lower()
                    if "female" in speaker_lower or "woman" in speaker_lower:
                        self.female_speakers.append(speaker)
                    elif "male" in speaker_lower or "man" in speaker_lower:
                        self.male_speakers.append(speaker)
                    # For remaining speakers, use analysis of typical names
                    elif any(female_name in speaker_lower for female_name in ["ana", "brenda", "daisy", "alison", "sofia", "henriette", "zofija"]):
                        self.female_speakers.append(speaker)
                    elif any(male_name in speaker_lower for male_name in ["damien", "viktor", "eugenio", "ferran", "baldur"]):
                        self.male_speakers.append(speaker)
                
                log_info(f"Classified {len(self.male_speakers)} male voices and {len(self.female_speakers)} female voices")
                log_info(f"Total available voices: {len(self.all_speakers)}")
            
            # Store available languages
            if hasattr(self.tts_model, "languages") and self.tts_model.languages:
                self.available_languages = self.tts_model.languages
                log_info(f"Available languages: {self.available_languages}")
            
            # Mark as initialized
            self.tts_initialized = True
            log_success("Coqui TTS model initialized successfully")
            return True
            
        except Exception as e:
            log_error(f"Failed to initialize Coqui TTS: {str(e)}")
            return False

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
        """Select a Coqui voice based on channel type and gender preference"""
        from video import log_info, log_warning, log_error
        
        # Make sure TTS is initialized
        if not self.tts_initialized:
            self._initialize_tts_model()
        
        # Get list of emotions for this channel type
        emotion_options = self.emotion_map.get(channel_type, self.emotion_map['default'])
        
        # Select a random emotion from the options for variety
        emotion = random.choice(emotion_options)
        
        # Determine gender preference based on channel type if not specified
        if gender is None:
            if channel_type in ['ai_money', 'tech_humor', 'fitness_motivation']:
                gender = 'male'  # These channels typically use male voices
            elif channel_type in ['baby_tips', 'quick_meals']:
                gender = 'female'  # These channels typically use female voices
            else:
                gender = random.choice(['male', 'female'])  # Random for other channels
        
        # Select a voice based on gender preference
        voice_candidates = []
        if gender == 'male' and self.male_speakers:
            # Try to find English male speakers first
            english_male = [s for s in self.male_speakers if s in self.english_speakers]
            if english_male:
                voice_candidates = english_male
                log_info(f"Using English male voice pool ({len(voice_candidates)} voices)")
            else:
                voice_candidates = self.male_speakers
                log_info(f"Using any male voice pool ({len(voice_candidates)} voices)")
        elif gender == 'female' and self.female_speakers:
            # Try to find English female speakers first
            english_female = [s for s in self.female_speakers if s in self.english_speakers]
            if english_female:
                voice_candidates = english_female
                log_info(f"Using English female voice pool ({len(voice_candidates)} voices)")
            else:
                voice_candidates = self.female_speakers
                log_info(f"Using any female voice pool ({len(voice_candidates)} voices)")
        elif self.english_speakers:
            # No gender preference or no voices for that gender, use any English speaker
            voice_candidates = self.english_speakers
            log_info(f"Using any English voice pool ({len(voice_candidates)} voices)")
        else:
            # Fallback to any available speaker
            voice_candidates = self.all_speakers
            log_info(f"Using any available voice pool ({len(voice_candidates)} voices)")
        
        # Select a random voice from the candidates
        if voice_candidates:
            voice = random.choice(voice_candidates)
            log_info(f"Selected Coqui voice: {voice} (emotion: {emotion})")
            return voice, emotion
        else:
            log_warning("No suitable voices found, using default voice")
            return "Baldur Sanjin", emotion  # Default voice as fallback

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
        from video import log_info, log_success, log_error, log_step, log_progress, log_warning
        import random
        
        try:
            # Initialize TTS model if not already done
            if not self.tts_initialized:
                if not self._initialize_tts_model():
                    log_error("Failed to initialize TTS model")
                    return None
            
            output_path = f"temp/tts/{voice}_{emotion}_{int(time.time())}.wav"
            os.makedirs("temp/tts", exist_ok=True)
            
            # Extract global style if present
            global_style = emotion
            style_match = re.search(r'<style="([^"]+)">(.*?)</style>', text)
            if style_match:
                global_style = style_match.group(1)
                text = style_match.group(2)
                log_info(f"Using global style: {global_style}")
            
            # Process text in smaller chunks for better quality
            # First, extract sentences with their emotion tags
            tagged_sentences = []
            
            # Extract tagged sentences
            tag_pattern = r'<(question|emphasis|excited|sad|serious)>(.*?)</\1>'
            last_end = 0
            for match in re.finditer(tag_pattern, text):
                # If there's untagged text before this match, add it
                if match.start() > last_end:
                    untagged = text[last_end:match.start()].strip()
                    if untagged:
                        tagged_sentences.append((untagged, global_style))
                
                # Add the tagged sentence with its emotion
                tag_type = match.group(1)
                sentence = match.group(2).strip()
                
                # Map tag types to emotions
                emotion_map = {
                    'question': 'curious',
                    'emphasis': 'energetic',
                    'excited': 'excited',
                    'sad': 'sad',
                    'serious': 'serious'
                }
                
                sentence_emotion = emotion_map.get(tag_type, global_style)
                tagged_sentences.append((sentence, sentence_emotion))
                
                last_end = match.end()
            
            # Add any remaining text
            if last_end < len(text):
                remaining = text[last_end:].strip()
                if remaining:
                    tagged_sentences.append((remaining, global_style))
            
            # If no tagged sentences were found, use the original text
            if not tagged_sentences:
                # Split by sentence endings
                sentences = re.split(r'(?<=[.!?])\s+', text)
                tagged_sentences = [(s.strip(), global_style) for s in sentences if s.strip()]
            
            log_info(f"Processing {len(tagged_sentences)} sentences with emotion for better TTS quality")
            
            # Generate audio for each chunk
            chunk_audios = []
            for i, (sentence, sentence_emotion) in enumerate(tagged_sentences):
                log_step(i+1, len(tagged_sentences), f"Generating speech chunk {i+1}/{len(tagged_sentences)} with emotion: {sentence_emotion}")
                
                # Add emotion-specific enhancements to the text
                enhanced_text = self._enhance_text_for_emotion(sentence, sentence_emotion)
                
                # Adjust speed based on emotion and content type
                current_speed = speed
                if sentence_emotion == "excited" or sentence_emotion == "energetic":
                    log_info("Using faster speed for energetic/excited emotion")
                    current_speed *= 1.15
                elif sentence_emotion == "sad" or sentence_emotion == "serious":
                    log_info("Using slower speed for sad/serious emotion")
                    current_speed *= 0.9
                elif sentence_emotion == "curious":
                    # For questions, slightly slower with rising intonation
                    current_speed *= 0.95
                
                # Generate TTS
                chunk_path = f"temp/tts/sentence_{i}_{int(time.time())}.wav"
                
                try:
                    # Use the voice parameter directly - it should be a valid speaker name
                    self.tts_synthesizer.tts_to_file(
                        text=enhanced_text,
                        file_path=chunk_path,
                        speaker=voice,
                        language="en",
                        speed=current_speed
                    )
                    log_success(f"Generated voice file: {chunk_path}")
                    chunk_audios.append(chunk_path)
                except Exception as chunk_error:
                    log_error(f"Failed to generate TTS for chunk {i+1}: {str(chunk_error)}")
                    # Try with simpler parameters as fallback
                    try:
                        log_info("Trying fallback TTS method")
                        self.tts_synthesizer.tts_to_file(
                            text=enhanced_text,
                            file_path=chunk_path,
                            speaker=voice,
                            language="en"
                        )
                        log_success(f"Generated voice file with fallback method: {chunk_path}")
                        chunk_audios.append(chunk_path)
                    except Exception as fallback_error:
                        log_error(f"Fallback TTS also failed: {str(fallback_error)}")
                        continue
            
            # Combine all chunks into a single audio file
            if chunk_audios:
                combined_audio = None
                for audio_path in chunk_audios:
                    audio = AudioSegment.from_file(audio_path)
                    if combined_audio is None:
                        combined_audio = audio
                    else:
                        combined_audio += audio
                
                if combined_audio:
                    combined_audio.export(output_path, format="wav")
                    
                    # Clean up temporary files
                    for path in chunk_audios:
                        try:
                            os.remove(path)
                        except:
                            pass
                    
                    # Convert to MP3 for better compatibility
                    mp3_path = output_path.replace(".wav", ".mp3")
                    combined_audio.export(mp3_path, format="mp3", bitrate="192k")
                    log_success(f"Generated voice file: {mp3_path}")
                return mp3_path
            
            return None
            
        except Exception as e:
            log_error(f"Coqui TTS failed: {str(e)}")
            return None
    
    def _enhance_text_for_emotion(self, text, emotion):
        """Enhance text with emotion-specific markers and phrasing"""
        from video import log_info
        
        # Define enhancement patterns for different emotions
        emotion_patterns = {
            # Basic emotions
            "neutral": {
                "prefix": "",
                "emphasis": "",
                "pacing": "normal",
                "description": "in a calm, balanced tone"
            },
            "professional": {
                "prefix": "",
                "emphasis": "clearly and confidently",
                "pacing": "measured",
                "description": "in a professional, authoritative manner"
            },
            "cheerful": {
                "prefix": "",
                "emphasis": "enthusiastically",
                "pacing": "upbeat",
                "description": "with a cheerful, upbeat tone"
            },
            "friendly": {
                "prefix": "",
                "emphasis": "warmly",
                "pacing": "relaxed",
                "description": "in a friendly, approachable way"
            },
            "serious": {
                "prefix": "",
                "emphasis": "seriously",
                "pacing": "deliberate",
                "description": "with a serious, thoughtful tone"
            },
            
            # Extended emotions
            "excited": {
                "prefix": "",
                "emphasis": "excitedly",
                "pacing": "fast",
                "description": "with excitement and enthusiasm"
            },
            "calm": {
                "prefix": "",
                "emphasis": "calmly",
                "pacing": "slow",
                "description": "with a calming, soothing tone"
            },
            "sad": {
                "prefix": "",
                "emphasis": "sadly",
                "pacing": "slow",
                "description": "with a somber, reflective tone"
            },
            "curious": {
                "prefix": "",
                "emphasis": "with curiosity",
                "pacing": "varied",
                "description": "with a rising, inquisitive tone"
            },
            "energetic": {
                "prefix": "",
                "emphasis": "energetically",
                "pacing": "fast",
                "description": "with high energy and emphasis"
            },
            "helpful": {
                "prefix": "",
                "emphasis": "helpfully",
                "pacing": "measured",
                "description": "in a helpful, instructive manner"
            }
        }
        
        # Get the pattern for the specified emotion, or use neutral as default
        pattern = emotion_patterns.get(emotion, emotion_patterns["neutral"])
        
        # Apply text enhancements based on the emotion pattern
        enhanced_text = text
        
        # Add SSML-like markers for better emotion expression
        if emotion == "excited" or emotion == "energetic":
            # Add emphasis to key words
            words = enhanced_text.split()
            for i, word in enumerate(words):
                if len(word) > 4 and random.random() < 0.3:  # Randomly emphasize some longer words
                    words[i] = f"<emphasis>{word}</emphasis>"
            enhanced_text = " ".join(words)
            
        elif emotion == "curious":
            # For questions, ensure rising intonation
            if not enhanced_text.endswith("?"):
                enhanced_text += "?"
                
        elif emotion == "sad":
            # Add slight pauses for sad emotion
            enhanced_text = enhanced_text.replace(", ", ", <break time='300ms'/> ")
            
        elif emotion == "serious":
            # Add emphasis to important words
            for important_word in ["important", "critical", "essential", "must", "key"]:
                if important_word in enhanced_text.lower():
                    enhanced_text = enhanced_text.replace(
                        important_word, 
                        f"<emphasis>{important_word}</emphasis>"
                    )
        
        # Add emotion description as a prefix if needed for the TTS system
        log_info(f"Enhancing text {pattern['description']}")
        
        return enhanced_text
    
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
        """Get appropriate voice for each channel (OpenAI format)"""
        # OpenAI voices: 'nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral'
        voices = {
            'tech_humor': 'echo',     # Male energetic
            'ai_money': 'onyx',       # Male professional
            'baby_tips': 'nova',      # Female warm
            'quick_meals': 'shimmer', # Female enthusiastic
            'fitness_motivation': 'echo'  # Male motivational
        }
        return voices.get(channel_type, 'nova')  # Default to nova if channel not found

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
        """Clean the script for TTS generation by removing emojis and special characters"""
        # First, preserve the original script for display purposes
        display_script = script
        
        # Remove emojis from TTS script using emoji library
        import emoji
        tts_script = emoji.replace_emoji(script, '')
        
        # Remove trailing '.']' that causes strange sounds
        tts_script = re.sub(r"\.\'\]$", "", tts_script)
        tts_script = re.sub(r"\.\'\]\s*$", "", tts_script)  # Also catch if there's whitespace after
        tts_script = re.sub(r"\[|\]|\'", "", tts_script)    # Remove any remaining brackets and quotes
        
        # Clean up any double spaces or weird whitespace
        tts_script = ' '.join(tts_script.split())
        
        # Remove any remaining special characters except basic punctuation
        tts_script = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', tts_script)
        
        # Clean up any multiple periods or spaces
        tts_script = re.sub(r'\.+', '.', tts_script)
        tts_script = re.sub(r'\s+', ' ', tts_script)
        
        # Remove trailing period that causes strange sounds in TTS
        tts_script = tts_script.rstrip('.')
        
        # Enhance script with emotion and intonation markers
        tts_script = self._enhance_script_with_emotion(tts_script)
        
        # Log the cleaning process
        print(colored("Original script:", "cyan"))
        print(script)
        print(colored("\nCleaned script for TTS:", "green"))
        print(tts_script)
        
        return tts_script.strip()
        
    def _enhance_script_with_emotion(self, script):
        """Add emotion and intonation markers to the script for better TTS"""
        # Don't modify if script is empty
        if not script:
            return script
            
        # Split into sentences for better processing
        sentences = re.split(r'([.!?])', script)
        sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2] + [''] * (len(sentences[0::2]) - len(sentences[1::2])))]
        sentences = [s.strip() for s in sentences if s.strip()]
        
        enhanced_sentences = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Detect sentence type and add appropriate markers
            if sentence.endswith('?'):
                # Questions - add rising intonation
                enhanced = f"<question>{sentence}</question>"
            elif sentence.endswith('!'):
                # Exclamations - add emphasis
                enhanced = f"<emphasis>{sentence}</emphasis>"
            elif any(keyword in sentence.lower() for keyword in ['amazing', 'awesome', 'incredible', 'wow']):
                # Enthusiastic statements
                enhanced = f"<excited>{sentence}</excited>"
            elif any(keyword in sentence.lower() for keyword in ['sad', 'unfortunately', 'sorry']):
                # Sad statements
                enhanced = f"<sad>{sentence}</sad>"
            elif any(keyword in sentence.lower() for keyword in ['important', 'remember', 'key', 'crucial']):
                # Important information
                enhanced = f"<serious>{sentence}</serious>"
            else:
                # Regular statements - no special markers
                enhanced = sentence
                
            enhanced_sentences.append(enhanced)
            
        # Join enhanced sentences
        enhanced_script = ' '.join(enhanced_sentences)
        
        # Add global emotion based on content
        if '😂' in script or '🤣' in script or 'funny' in script.lower() or 'joke' in script.lower():
            enhanced_script = f"<style=\"cheerful\">{enhanced_script}</style>"
        elif '💡' in script or 'tip' in script.lower() or 'hack' in script.lower():
            enhanced_script = f"<style=\"helpful\">{enhanced_script}</style>"
        elif '🔥' in script or 'amazing' in script.lower() or 'awesome' in script.lower():
            enhanced_script = f"<style=\"excited\">{enhanced_script}</style>"
            
        return enhanced_script

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
            except Exception as e:
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
        """Clean up the video library to avoid excessive disk usage"""
        from video import log_section, log_info, log_success, log_warning, log_error
        
        try:
            log_section("Video Library Cleanup", "🧹")
            
            # Get base video directory
            base_dir = "assets/videos/categorized"
            
            # Track total videos cleaned
            total_cleaned = 0
            categories_cleaned = 0
            
            # Clean specific channel type if provided
            if channel_type:
                # Find directories for this channel type
                channel_dirs = []
                for subdir in os.listdir(base_dir):
                    if channel_type.lower() in subdir.lower():
                        channel_dirs.append(os.path.join(base_dir, subdir))
                
                if not channel_dirs:
                    log_warning(f"No video directories found for {channel_type}")
                    return
                
                # Clean each directory
                for directory in channel_dirs:
                    cleaned = self._cleanup_directory(directory, max_videos, days_to_keep)
                    if cleaned > 0:
                        total_cleaned += cleaned
                        categories_cleaned += 1
                
                log_success(f"Cleaned up {total_cleaned} videos from {categories_cleaned} categories for {channel_type}")
                return
            
            # If no channel type provided, clean all directories
            categories = []
            for subdir in os.listdir(base_dir):
                path = os.path.join(base_dir, subdir)
                if os.path.isdir(path):
                    categories.append(path)
            
            # Clean each category directory
            for directory in categories:
                if os.path.isdir(directory):
                    try:
                        cleaned = self._cleanup_directory(directory, max_videos, days_to_keep)
                        if cleaned > 0:
                            total_cleaned += cleaned
                            categories_cleaned += 1
                    except Exception as e:
                        log_warning(f"Error cleaning {os.path.basename(directory)}: {str(e)}")
            
            if total_cleaned > 0:
                log_success(f"Cleaned up {total_cleaned} videos from {categories_cleaned} categories")
            else:
                log_info("No videos needed cleaning")
                
        except Exception as e:
            log_error(f"Error cleaning up video library: {str(e)}")

    def _cleanup_directory(self, directory, max_videos=20, days_to_keep=30):
        """Clean up a specific directory of videos"""
        from video import log_info, log_success, log_warning
        
        try:
            videos = []
            # Get list of videos with creation time
            for file in os.listdir(directory):
                if file.endswith(".mp4"):
                    file_path = os.path.join(directory, file)
                    created = os.path.getctime(file_path)
                    videos.append({
                        'path': file_path,
                        'created': created,
                        'age': (time.time() - created) / (60 * 60 * 24),  # Age in days
                        'keep': False  # Default to not keeping
                    })
            
            # Check if cleaning is needed
            if len(videos) <= max_videos:
                return 0
            
            # Sort by creation time (oldest first)
            videos.sort(key=lambda x: x['created'])
            
            # Mark videos to keep or delete
            to_keep = []
            to_delete = []
            
            # First, keep the most recent videos up to max_videos
            for i, video in enumerate(reversed(videos)):
                if i < max_videos:
                    video['keep'] = True
                    to_keep.append(video)
                else:
                    # For older videos, keep them if they're within days_to_keep
                    if video['age'] <= days_to_keep:
                        video['keep'] = True
                        to_keep.append(video)
                    else:
                        to_delete.append(video)
            
            # If nothing to delete, we're done
            if not to_delete:
                return 0
                
            # Delete videos marked for deletion
            deleted_count = 0
            for video in to_delete:
                try:
                    os.remove(video['path'])
                    deleted_count += 1
                except Exception as e:
                    log_warning(f"Could not remove {os.path.basename(video['path'])}: {str(e)}")
            
            if deleted_count > 0:
                return deleted_count
            else:
                return 0
            
        except Exception as e:
            log_warning(f"Error cleaning up directory {os.path.basename(directory)}: {str(e)}")
            return 0

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
        from video import log_section, log_info, log_success, log_warning, log_error, log_highlight, log_result, log_step, log_separator
        
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
                
            # Final output path (for clear display at the end)
            final_output_path = custom_output_path if custom_output_path else output_path
            
            # Get audio duration
            audio = AudioFileClip(tts_path)
            audio_duration = audio.duration
            audio.close()
            
            log_section("Video Generation", "🎬")
            log_info(f"Target duration: {audio_duration:.1f}s")
            
            # Calculate video duration with padding
            start_padding = 1.0  # 1 second at start
            end_padding = 3.0    # 3 seconds at end
            target_duration = audio_duration + end_padding
            
            # Find the script path based on the TTS path
            script_path = None
            if tts_path:
                possible_script_path = tts_path.replace('.mp3', '.txt')
                if os.path.exists(possible_script_path):
                    script_path = possible_script_path
                else:
                    possible_script_path = os.path.join('temp', 'tts', f"{channel_type}.txt")
                    if os.path.exists(possible_script_path):
                        script_path = possible_script_path
            
            # Generate video
            from video import generate_video
            
            # Suppress MoviePy warnings during video generation
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.video.io")
            
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
                log_error("Failed to generate video")
                return None
            
            # Create temporary paths for processing
            temp_video_path = os.path.join("temp", "temp_video.mp4")
            temp_audio_path = os.path.join("temp", "temp_audio.mp3")
            
            step1_start = time.time()
            log_step(1, 4, "Extracting audio track", start_time=step1_start)
            
            # Extract audio from video
            if hasattr(video, 'audio') and video.audio is not None:
                try:
                    # If we have background music, mix it with TTS
                    if self.use_background_music:
                        # Find an appropriate music file
                        music_path = self.get_background_music(channel_type)
                        if music_path and os.path.exists(music_path):
                            log_info(f"Mixing audio with background music")
                            try:
                                # Extract video audio
                                source_audio_path = os.path.join("temp", "source_audio_temp.mp3")
                                video.audio.write_audiofile(source_audio_path, 
                                                         logger=None, verbose=False)
                                source_audio = AudioFileClip(source_audio_path)
                                # Mix with music
                                from video import mix_audio
                                mix_audio(tts_path, music_path, temp_audio_path, 
                                        music_volume=self.music_volume,
                                        fade_in=2.0, fade_out=3.0)
                                source_audio.close()
                            except Exception as mix_error:
                                log_warning(f"Error mixing audio: {str(mix_error)}")
                                log_warning("Using original TTS without music")
                                source_audio = AudioFileClip(tts_path)
                                source_audio.write_audiofile(temp_audio_path, 
                                                         logger=None, verbose=False)
                                source_audio.close()
                        else:
                            video.audio.write_audiofile(temp_audio_path, 
                                                     logger=None, verbose=False)
                    else:
                        video.audio.write_audiofile(temp_audio_path, 
                                                 logger=None, verbose=False)
                except Exception as e:
                    log_warning(f"Error extracting audio: {str(e)}")
                    log_warning("Falling back to original TTS audio")
                    source_audio = AudioFileClip(tts_path)
                    source_audio.write_audiofile(temp_audio_path, 
                                              logger=None, verbose=False)
                    source_audio.close()
            else:
                source_audio = AudioFileClip(tts_path)
                source_audio.write_audiofile(temp_audio_path, 
                                          logger=None, verbose=False)
                source_audio.close()
                log_warning("No audio found in video, using original TTS without music")
            
            step1_end = time.time()
            log_step(1, 4, "Extracting audio track", end_time=step1_end, start_time=step1_start)
            
            # Step 2: Write the video without audio
            step2_start = time.time()
            log_step(2, 4, "Processing video frames", start_time=step2_start)
            
            # Less verbose video writing
            video.without_audio().write_videofile(
                temp_video_path,
                codec='libx264',
                fps=30,
                preset='ultrafast',
                ffmpeg_params=["-vf", "format=yuv420p"],
                logger=None,
                verbose=False
            )
            
            step2_end = time.time()
            log_step(2, 4, "Processing video frames", end_time=step2_end, start_time=step2_start)
            
            # Step 3: Combine video and audio using ffmpeg directly
            step3_start = time.time()
            log_step(3, 4, "Combining video and audio tracks", start_time=step3_start)
            
            import subprocess
            
            # Run FFmpeg with output suppressed
            with open(os.devnull, 'w') as devnull:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_video_path,
                    "-i", temp_audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-shortest",
                    output_path
                ]
                subprocess.call(cmd, stdout=devnull, stderr=devnull)
            
            step3_end = time.time()
            log_step(3, 4, "Combining video and audio tracks", end_time=step3_end, start_time=step3_start)
            
            # Step 4: Clean up and finalize
            step4_start = time.time()
            log_step(4, 4, "Finalizing video", start_time=step4_start)
            
            # Check if output file exists and has a reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                # Clean up temporary files
                for temp_file in [temp_video_path, temp_audio_path]:
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                
                # Calculate processing times
                processing_time = time.time() - step1_start
                log_step(4, 4, "Finalizing video", end_time=time.time(), start_time=step4_start)
                
                # Display final output info
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                log_success(f"Video generated successfully in {processing_time:.1f}s")
                log_info(f"Output: {final_output_path}")
                log_info(f"File size: {file_size_mb:.1f} MB")
                
                # Get absolute path for easier access
                abs_path = os.path.abspath(final_output_path)
                log_info(f"Absolute path: {abs_path}")
                
                return final_output_path
            else:
                log_error("Failed to generate final video file")
                return None
            
        except Exception as e:
            log_error(f"Error generating video: {str(e)}")
            log_error(traceback.format_exc())
            return None

    async def generate_video(self, channel, script_file):
        """Generate a video for a channel"""
        from video import log_info, log_success, log_error, log_warning, log_result
        
        try:
            # Create a standardized output path
            output_path = f"output/videos/{channel}_latest.mp4"
            
            # Create the video
            success = await self.create_video(script_file, channel, output_path)
            
            if success:
                log_success(f"Video generation completed successfully")
                log_result("Final video", output_path)
                return output_path
            else:
                log_error("Video generation failed")
                return None
                
        except Exception as e:
            log_error(f"Error generating video: {str(e)}")
            return None

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

