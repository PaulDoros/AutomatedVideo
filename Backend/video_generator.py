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
        self.validator = ContentValidator()
        self.script_generator = ScriptGenerator()
        self.output_dir = "output/videos"
        self.temp_dir = "temp"
        self.assets_dir = "assets/videos"
        
        # Hardware optimization settings
        self.n_threads = min(multiprocessing.cpu_count(), 16)  # Use up to 16 CPU threads
        self.use_gpu = True  # Enable GPU acceleration
        
        # Create necessary directories
        for directory in [self.output_dir, self.temp_dir, self.assets_dir]:
            os.makedirs(directory, exist_ok=True)

        load_dotenv()
        self.pexels_api_key = os.getenv('PEXELS_API_KEY')
        self.content_checker = ContentQualityChecker()

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

        # Initialize audio manager for voice selection
        self.audio_manager = AudioManager()
        
        # Flag to use Coqui TTS instead of OpenAI TTS
        self.use_coqui_tts = True

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

    async def create_video(self, script_file, channel_type):
        try:
            print(colored("\n=== Video Generation Started ===", "blue"))
            
            # Load script
            with open(script_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                script = data.get('script', '')
            
            if not script:
                raise ValueError("Script is empty")
                
            # Create necessary directories
            print(colored("✓ Directory structure created", "green"))
            
            # Generate TTS audio
            tts_path = await self._generate_tts(script, channel_type)
            if not tts_path:
                raise ValueError("Failed to generate TTS audio")
            
            # Trim the end of the audio to remove strange sounds (0.3 seconds from the end)
            # This helps prevent the audio looping issue
            print(colored("Trimming 0.3 seconds from the end of audio to remove artifacts and prevent looping", "blue"))
            original_audio = AudioFileClip(tts_path)
            
            # Ensure we don't trim too much if the audio is short
            trim_amount = min(0.3, original_audio.duration * 0.02)  # Either 0.3s or 2% of duration, whichever is smaller
            trimmed_audio = original_audio.subclip(0, original_audio.duration - trim_amount)
            
            # Add 1-second silence at the beginning of the audio to match subtitle delay
            print(colored("Adding 1-second silence at the beginning of audio to match subtitle delay", "blue"))
            
            # Create 1 second of silence with the same parameters as the original audio
            silence_duration = 1.0
            silence = AudioClip(lambda t: 0, duration=silence_duration)
            
            # Add a small silence at the end to prevent audio loop issues
            end_silence = AudioClip(lambda t: 0, duration=0.5)
            
            # Concatenate silence at beginning, trimmed audio, and silence at end
            delayed_audio = concatenate_audioclips([silence, trimmed_audio, end_silence])
            
            # Save the delayed audio to a temporary file
            delayed_audio_path = f"temp/tts/{channel_type}_delayed.mp3"
            delayed_audio.write_audiofile(delayed_audio_path)
            
            # Close the audio clips to free resources
            original_audio.close()
            trimmed_audio.close()
            delayed_audio.close()
            
            # Use the delayed audio for the rest of the process
            tts_path = delayed_audio_path
                
            # Get audio duration for video length calculation
            audio_clip = AudioFileClip(tts_path)
            audio_duration = audio_clip.duration
            audio_clip.close()
            
            print(colored(f"Original audio duration: {audio_duration:.2f}s", "blue"))
            
            # Calculate target duration with padding
            start_padding = 1.0  # 1 second at start (already added to audio)
            end_padding = 3.0    # 3 seconds at end (increased from 2 to 3 seconds)
            target_duration = audio_duration + end_padding  # Start padding already included in audio
            print(colored(f"Video duration: {target_duration:.2f}s (with {start_padding}s start and {end_padding}s end padding)", "blue"))
            
            # Generate subtitles with proper timing
            subtitles_path = await self._generate_subtitles(script, tts_path, channel_type)
            if not subtitles_path:
                raise ValueError("Failed to generate subtitles")
                
            # Process background videos
            background_videos = await self._process_background_videos(channel_type, script)
            
            # Generate video
            from video import generate_video
            video = generate_video(
                background_path=background_videos,
                audio_path=tts_path,
                subtitles_path=subtitles_path,
                content_type=channel_type,
                target_duration=target_duration
            )
            
            if not video:
                raise ValueError("Failed to generate video")
                
            # Write final video with specific parameters to prevent black frames
            temp_video_path = "temp/temp_video_no_audio.mp4"
            temp_audio_path = "temp/temp_audio.mp3"
            output_path = "temp_output.mp4"
            
            print(colored(f"\n=== 🎥 Rendering Final Video 🎥 ===", "blue"))
            print(colored(f"ℹ️ Output path: {output_path}", "cyan"))
            
            # Show a message about rendering time
            print(colored("⏳ Rendering final video... This may take a while", "cyan"))
            start_time = time.time()
            
            # Step 1: Extract the audio - handle both simple AudioFileClip and CompositeAudioClip
            if hasattr(video, 'audio') and video.audio is not None:
                # Check if it's a CompositeAudioClip (which doesn't have fps)
                if isinstance(video.audio, CompositeAudioClip):
                    # Create a new audio file directly from the source
                    source_audio = AudioFileClip(tts_path)
                    source_audio.write_audiofile(temp_audio_path)
                    source_audio.close()
                else:
                    # It's a regular AudioFileClip
                    video.audio.write_audiofile(temp_audio_path)
            else:
                # If no audio in video, use the original TTS file
                shutil.copy2(tts_path, temp_audio_path)
            
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
                output_path
            ]
            subprocess.run(cmd, check=True)
            
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
            except Exception as e:
                print(colored(f"Warning: Error during cleanup: {str(e)}", "yellow"))
            
            # Create timestamped output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = f"output/videos/{channel_type}_{timestamp}.mp4"
            
            # Create latest output path
            latest_path = f"output/videos/{channel_type}_latest.mp4"
            
            # Copy to output directory
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
            shutil.copy2(output_path, final_output_path)
            shutil.copy2(output_path, latest_path)
            
            print(colored("\n=== Video Generation Complete ===", "green"))
            return True
            
        except Exception as e:
            print(colored(f"\n=== Video Generation Failed ===\nError: {str(e)}", "red"))
            return False

    async def _generate_tts(self, script, channel_type):
        """Generate TTS using either Coqui TTS or OpenAI's voice"""
        try:
            # Clean up the script - remove line numbers and preserve emojis for display
            # but the actual TTS will have emojis removed by the _clean_text method
            clean_script = '\n'.join(
                line.strip().strip('"') 
                for line in script.split('\n') 
                if line.strip() and not line[0].isdigit()
            )
            
            # Use Coqui TTS for diverse voices
            if self.use_coqui_tts:
                try:
                    # Select appropriate emotion for the content type
                    # (voice selection will be handled by the API based on whether the model supports speakers)
                    voice, emotion = self.audio_manager.select_coqui_voice(channel_type)
                    
                    # Store the selected voice to ensure consistency
                    print(colored(f"Selected voice: {voice} (emotion: {emotion})", "blue"))
                    
                    # Create unique filename based on channel type and emotion
                    filename = f"{channel_type}_{emotion}_{int(time.time())}.wav"
                    output_path = f"temp/tts/{filename}"
                    
                    # Check if the model supports multiple speakers
                    has_speaker_support = hasattr(self.audio_manager.voice_diversifier.api, 'speakers') and self.audio_manager.voice_diversifier.api.speakers
                    
                    # Set speed based on content type and emotion
                    # Dynamically adjust speed based on emotion and content type
                    speed_multiplier = 1.0  # Default speed
                    
                    # Adjust speed based on emotion
                    if emotion in ["humorous", "witty", "sarcastic"]:
                        # Slightly faster for joke delivery
                        speed_multiplier = 1.05
                    elif emotion in ["energetic", "enthusiastic"]:
                        speed_multiplier = 1.1
                    elif emotion in ["playful"]:
                        speed_multiplier = 1.03
                    
                    # Further adjust based on content type
                    if channel_type == "tech_humor":
                        # For tech humor, adjust speed based on emotion
                        if emotion in ["humorous", "witty", "sarcastic"]:
                            # Jokes need good timing - not too fast
                            speed_multiplier = 1.05
                        else:
                            # For other emotions in tech humor
                            speed_multiplier = 1.08
                    
                    print(colored(f"Using speed multiplier of {speed_multiplier} for {emotion} {channel_type} content", "blue"))
                    
                    # Confirm we're using XTTS v2
                    print(colored("Confirming TTS model: Using XTTS v2 for high-quality voice generation", "blue"))
                    
                    # Split script into sentences for better TTS quality
                    sentences = []
                    for line in clean_script.split('\n'):
                        # Split by common sentence terminators but preserve them
                        parts = re.split(r'([.!?])', line)
                        for i in range(0, len(parts)-1, 2):
                            if parts[i].strip():
                                sentence = parts[i] + (parts[i+1] if i+1 < len(parts) else '')
                                sentences.append(sentence.strip())
                    
                    # If no sentences were found, use the whole script
                    if not sentences:
                        sentences = [clean_script]
                    
                    print(colored(f"Processing {len(sentences)} sentences for better TTS quality", "blue"))
                    
                    if has_speaker_support:
                        # Generate voice using the Coqui TTS API with speaker - process sentence by sentence
                        print(colored(f"Using Coqui TTS voice: {voice} (emotion: {emotion}, speed: {speed_multiplier})", "blue"))
                        
                        # Generate each sentence separately for better prosody
                        sentence_audio_files = []
                        for i, sentence in enumerate(sentences):
                            if not sentence.strip():
                                continue
                                
                            # Add breaks between sentences for natural pauses
                            if i > 0:
                                sentence = f"<break time='0.2s'/> {sentence}"
                            
                            # Generate temporary file for this sentence
                            sentence_path = f"temp/tts/sentence_{i}_{int(time.time())}.wav"
                            
                            # Generate voice for this sentence
                            result = await self.audio_manager.voice_diversifier.api.generate_voice(
                                text=sentence,
                                speaker=voice,  # Use the same voice for all sentences
                                language="en",
                                emotion=emotion,
                                speed=speed_multiplier,
                                output_path=sentence_path
                            )
                            
                            if result:
                                sentence_audio_files.append(result)
                        
                        # Combine all sentence audio files
                        if sentence_audio_files:
                            # Use pydub to concatenate WAV files
                            combined = AudioSegment.empty()
                            for audio_file in sentence_audio_files:
                                segment = AudioSegment.from_wav(audio_file)
                                combined += segment
                            
                            # Add a small silence at the end to prevent audio loop issues
                            silence = AudioSegment.silent(duration=300)  # 300ms silence
                            combined = combined + silence
                            
                            # Save combined audio
                            combined.export(output_path, format="wav")
                            
                            # Clean up temporary files
                            for audio_file in sentence_audio_files:
                                try:
                                    os.remove(audio_file)
                                except:
                                    pass
                            
                            print(colored(f"✓ Generated voice file: {output_path}", "green"))
                            result = output_path
                        else:
                            # Fallback to generating the whole script at once
                            result = await self.audio_manager.voice_diversifier.api.generate_voice(
                                text=clean_script,
                                speaker=voice,
                                language="en",
                                emotion=emotion,
                                speed=speed_multiplier,
                                output_path=output_path
                            )
                    else:
                        # Use default voice with emotion if the model doesn't support multiple speakers
                        print(colored(f"Using Coqui TTS default voice (emotion: {emotion}, speed: {speed_multiplier})", "blue"))
                        
                        # Generate voice using the Coqui TTS API without speaker
                        result = await self.audio_manager.voice_diversifier.api.generate_voice(
                            text=clean_script,
                            language="en",
                            emotion=emotion,
                            speed=speed_multiplier,
                            output_path=output_path
                        )
                    
                    if result:
                        # Convert WAV to MP3 for compatibility with video generation
                        final_path = f"temp/tts/{channel_type}_latest.mp3"
                        
                        # Use moviepy to convert WAV to MP3 without any processing
                        audio_clip = AudioFileClip(result)
                        audio_clip.write_audiofile(final_path)
                        audio_clip.close()
                        
                        print(colored(f"✓ Generated TTS audio with Coqui: {final_path}", "green"))
                        return final_path
                    else:
                        raise ValueError("Failed to generate TTS with Coqui")
                except Exception as e:
                    print(colored(f"Coqui TTS failed: {str(e)}", "red"))
                    print(colored("Falling back to OpenAI TTS", "yellow"))
            
            # Fall back to OpenAI TTS
            return await self._generate_openai_tts(clean_script, channel_type)
            
        except Exception as e:
            print(colored(f"TTS Generation failed: {str(e)}", "red"))
            return None

    async def _generate_openai_tts(self, clean_script, channel_type):
        """Generate TTS using OpenAI's voice API"""
        try:
            # Get appropriate voice for the channel
            voice_config = self.get_voice_for_channel(channel_type)
            voice = voice_config.get('voice', 'nova')
            style = voice_config.get('style', 'neutral')
            
            print(colored(f"Using OpenAI voice: {voice} (style: {style})", "blue"))
            
            # Create client
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Create output path
            os.makedirs("temp/tts", exist_ok=True)
            output_path = f"temp/tts/{channel_type}_latest.mp3"
            
            # Generate speech
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=clean_script,
                speed=1.2 if channel_type == "tech_humor" else 1.0  # Faster for tech humor
            )
            
            # Save to file
            response.stream_to_file(output_path)
            
            print(colored(f"✓ Generated TTS audio with OpenAI and trimmed 0.1s from end", "green"))
            return output_path
            
        except Exception as e:
            print(colored(f"OpenAI TTS failed: {str(e)}", "red"))
            return None

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
        """Process background videos for the channel - now with multiple video support"""
        try:
            print(colored("\n=== Processing Background Videos ===", "blue"))
            
            # Create directory if it doesn't exist
            video_dir = os.path.join("assets", "videos", channel_type)
            os.makedirs(video_dir, exist_ok=True)
            
            # Check if we have local videos first
            local_videos = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                           if f.endswith(('.mp4', '.mov')) and os.path.getsize(os.path.join(video_dir, f)) > 0]
            
            # Number of videos to use for a more dynamic video
            target_video_count = 4  # We want to use 4 videos for a more dynamic experience
            
            if local_videos:
                # Use existing videos if we have enough
                if len(local_videos) >= target_video_count:
                    print(colored(f"Using {target_video_count} local videos from {video_dir} for a dynamic video", "green"))
                    
                    # Select a subset of videos to use
                    selected_videos = random.sample(local_videos, target_video_count)
                    
                    # Verify each video is valid
                    valid_videos = []
                    for video_path in selected_videos:
                        try:
                            # Check if the video is valid
                            video = VideoFileClip(video_path)
                            if video.duration >= 3:  # At least 3 seconds
                                valid_videos.append(video_path)
                            video.close()
                        except Exception as e:
                            print(colored(f"Error checking video {video_path}: {str(e)}", "yellow"))
                    
                    # If we have at least 2 valid videos, return them
                    if len(valid_videos) >= 2:
                        print(colored(f"Using {len(valid_videos)} valid local videos for a dynamic video", "green"))
                        return valid_videos
                    
                # If we don't have enough valid videos, use what we have
                if local_videos:
                    print(colored(f"Using {len(local_videos)} local videos", "green"))
                    return local_videos
            
            # If we don't have local videos, try to get suggestions from GPT
            if script:
                print(colored("No local videos found, getting suggestions from GPT", "yellow"))
                
                # Analyze script content
                script_analysis = self._analyze_script_content(script)
                
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
                    
                    # Limit to target_video_count terms for variety
                    search_terms = search_terms[:target_video_count]
                    
                    # Search for videos
                    final_videos = []
                    for term in search_terms:
                        videos = await self._search_and_save_videos(term, video_dir, count=1)
                        if videos:
                            final_videos.extend(videos)
                    
                    if len(final_videos) >= 2:
                        print(colored(f"Found {len(final_videos)} videos from search for a dynamic video", "green"))
                        return final_videos
                    elif final_videos:
                        print(colored(f"Found {len(final_videos)} videos from search", "green"))
                        return final_videos
            
            # If all else fails, create a default background
            print(colored("No videos found, creating default background", "yellow"))
            return self._create_default_background(channel_type)
            
        except Exception as e:
            print(colored(f"Error processing background videos: {str(e)}", "red"))
            # Create a default background as fallback
            return self._create_default_background(channel_type)

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

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(colored(f"Error deleting {file_path}: {str(e)}", "yellow"))
        except Exception as e:
            print(colored(f"Error during cleanup: {str(e)}", "yellow"))

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

    async def create_tts_audio(self, script, channel_type, voice=None):
        """Create TTS audio from script"""
        try:
            # Create temp directory
            os.makedirs("temp/tts", exist_ok=True)
            
            # Get voice for channel
            if not voice and self.use_coqui_tts:
                voice, emotion = self.select_coqui_voice(channel_type)
                print(colored(f"Selected voice: {voice} (emotion: {emotion})", "blue"))
                
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
                
                # Generate TTS audio with Coqui
                from coqui_integration import CoquiTTSAPI
                
                coqui = CoquiTTSAPI()
                
                # Process script into sentences for better TTS quality
                print(colored(f"Processing {len(script.split('.'))} sentences for better TTS quality", "blue"))
                
                # Generate TTS for each sentence separately for better quality
                sentences = [s.strip() for s in script.split('.') if s.strip()]
                audio_files = []
                
                print(colored(f"Using Coqui TTS voice: {voice} (emotion: {emotion}, speed: {speed_multiplier})", "blue"))
                
                for i, sentence in enumerate(sentences):
                    # Add period back if it was removed by split
                    if not sentence.endswith(('!', '?', '.')):
                        sentence += '.'
                    
                    # Add slight pause between sentences
                    if i > 0:
                        sentence = f"<break time='0.2s'/> {sentence}"
                    
                    # Generate audio for this sentence
                    output_path = f"temp/tts/sentence_{i}_{int(time.time())}.wav"
                    result = await coqui.generate_voice(
                        text=sentence,
                        speaker=voice,
                        language="en",
                        emotion=emotion,
                        speed=speed_multiplier,
                        output_path=output_path
                    )
                    
                    if result:
                        audio_files.append(result)
                    else:
                        print(colored(f"Failed to generate TTS for sentence: {sentence}", "red"))
                
                # Combine all sentence audio files
                if audio_files:
                    # Create output path
                    output_path = f"temp/tts/{channel_type}_{emotion}_{int(time.time())}.wav"
                    
                    # Concatenate audio files
                    from pydub import AudioSegment
                    combined = AudioSegment.empty()
                    for audio_file in audio_files:
                        segment = AudioSegment.from_file(audio_file)
                        combined += segment
                    
                    # Export combined audio
                    combined.export(output_path, format="wav")
                    
                    # Convert to MP3 for smaller file size
                    mp3_path = f"temp/tts/{channel_type}_latest.mp3"
                    combined.export(mp3_path, format="mp3", bitrate="128k")
                    
                    print(colored(f"✓ Generated voice file: {output_path}", "green"))
                    return mp3_path
            
            # If we get here, either no Coqui TTS or no audio files were generated
            return None
            
        except Exception as e:
            print(colored(f"Error creating TTS audio: {str(e)}", "red"))
            return None
        finally:
            # Clean up temporary files if needed
            pass

def generate_video_for_channel(channel, topic, hashtags):
    """Generate and upload a video for a specific channel"""
    try:
        # Video generation parameters
        params = {
            'videoSubject': topic,
            'aiModel': 'gpt-4',  # or your preferred model
            'voice': 'en_us_001',
            'paragraphNumber': 1,
            'automateYoutubeUpload': True,
            'automateTikTokUpload': True,
            'youtubeAccount': channel,
            'tiktokAccount': channel,
            'useMusic': True,
            'threads': 2
        }
        
        # Generate video
        video_path = generate_video(params)
        
        if video_path:
            # Upload to YouTube
            youtube_response = upload_video(
                video_path=video_path,
                title=f"{topic} #{' #'.join(hashtags)}",
                description=f"Auto-generated content about {topic}\n\n#{' #'.join(hashtags)}",
                category="28",  # Tech
                keywords=",".join(hashtags),
                privacy_status="public",
                channel=channel
            )
            
            # Upload to TikTok
            tiktok_session = os.getenv(f"TIKTOK_SESSION_ID_{channel.upper()}")
            if tiktok_session:
                uploader = TikTokUploader(tiktok_session)
                tiktok_response = uploader.upload_video(
                    video_path=video_path,
                    title=f"{topic} #{' #'.join(hashtags)}"[:150],
                    tags=hashtags
                )
            
            print(colored(f"[+] Successfully generated and uploaded video for {channel}: {topic}", "green"))
            return True
            
    except Exception as e:
        print(colored(f"[-] Error generating video for {channel}: {str(e)}", "red"))
        return False

