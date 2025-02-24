from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
from termcolor import colored
from content_validator import ContentValidator, ScriptGenerator
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip, concatenate_audioclips
from moviepy.video.tools.subtitles import SubtitlesClip
from tiktokvoice import tts
from video import generate_video, generate_subtitles, combine_videos, save_video
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

# Third-party imports
try:
    from TTS.api import TTS
except ImportError:
    print("Please install TTS: pip install TTS")
    TTS = None

try:
    import torch
except ImportError:
    print("Please install PyTorch: pip install torch")
    torch = None

try:
    from scipy.io import wavfile
except ImportError:
    print("Please install SciPy: pip install scipy")
    wavfile = None

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

        # Check required packages
        if None in (TTS, torch, wavfile):
            raise ImportError(
                "Missing required packages. Please install:\n"
                "pip install TTS torch scipy"
            )

        # Initialize Coqui TTS only once
        tts_model_path = "assets/tts/model"
        os.makedirs(os.path.dirname(tts_model_path), exist_ok=True)

        # Fix PyTorch "weights only" issue
        torch.serialization.default_load_weights_only = False
        torch.serialization.add_safe_globals(["numpy.core.multiarray.scalar"])

        # Define voice model
        MODEL_NAME = "tts_models/en/vctk/vits"
        
        print(colored("Loading TTS model...", "blue"))
        self.tts = TTS(
            model_name=MODEL_NAME,
            progress_bar=False,
            gpu=torch.cuda.is_available()
        )
        
        # Set speaker for multi-speaker model
        if hasattr(self.tts, 'speakers') and len(self.tts.speakers) > 0:
            self.tts.speaker = "p273"  # Male voice with good clarity

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

            # Generate components
            tts_path = await self._generate_tts(script, channel_type)
            if not tts_path:
                raise ValueError("Failed to generate TTS audio")
            
            subtitles_path = await self._generate_subtitles(script, tts_path, channel_type)
            if not subtitles_path:
                raise ValueError("Failed to generate subtitles")
            
            background_paths = await self._process_background_videos(channel_type)
            if not background_paths:
                raise ValueError("Failed to get background videos")
            
            # Ensure we have a list of background paths
            if isinstance(background_paths, str):
                background_paths = [background_paths]
            
            # Generate final video
            output_path = generate_video(
                background_paths[0] if len(background_paths) == 1 else background_paths,
                tts_path,
                subtitles_path
            )
            
            if not output_path:
                raise ValueError("Failed to generate final video")
            
            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = f"output/videos/{channel_type}_{timestamp}.mp4"
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            shutil.copy2(output_path, final_path)
            
            print(colored("\n=== Video Generation Complete ===", "green"))
            return True
            
        except Exception as e:
            print(colored(f"\n=== Video Generation Failed ===\nError: {str(e)}", "red"))
            return False

    async def _generate_tts(self, script, channel_type):
        """Generate TTS using Coqui"""
        try:
            tts_path = f"temp/tts/{channel_type}_latest.mp3"
            os.makedirs(os.path.dirname(tts_path), exist_ok=True)
            
            sentences = []
            for line in script.split('\n'):
                clean_line = re.sub(r'[^\x00-\x7F]+', '', line).strip()
                if clean_line:
                    sentences.append(clean_line)
            
            audio_clips = []
            timings = []
            current_time = 0
            
            for sentence in sentences:
                temp_path = f"temp/tts/temp_{uuid.uuid4()}.wav"
                try:
                    # Generate audio
                    if hasattr(self.tts, 'speakers') and len(self.tts.speakers) > 0:
                        self.tts.tts_to_file(text=sentence, file_path=temp_path, speaker=self.tts.speaker)
                    else:
                        self.tts.tts_to_file(text=sentence, file_path=temp_path)
                    
                    # Wait a bit to ensure file is released
                    await asyncio.sleep(0.1)
                    
                    # Load audio only after ensuring file is written
                    if os.path.exists(temp_path):
                        audio_clip = AudioFileClip(temp_path)
                        audio_clips.append(audio_clip)
                        
                        duration = audio_clip.duration
                        timings.append({
                            'text': sentence,
                            'start': current_time,
                            'end': current_time + duration
                        })
                        current_time += duration
                except Exception as e:
                    print(colored(f"Warning: Failed to process sentence: {e}", "yellow"))
                finally:
                    # Clean up temp file with retry
                    for _ in range(3):
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            break
                        except:
                            await asyncio.sleep(0.1)
            
            if not audio_clips:
                raise ValueError("No audio clips were generated")
            
            # Combine all audio clips
            final_audio = concatenate_audioclips(audio_clips)
            final_audio.write_audiofile(tts_path, fps=44100)
            
            # Clean up
            for clip in audio_clips:
                clip.close()
            
            self.sentence_timings = timings
            return tts_path
            
        except Exception as e:
            print(colored(f"TTS generation failed: {str(e)}", "red"))
            return None

    async def _generate_subtitles(self, script, tts_path, channel_type):
        """Generate subtitles using stored timings"""
        try:
            if not tts_path or not os.path.exists(tts_path):
                raise ValueError("Invalid TTS audio path")
            
            if not hasattr(self, 'sentence_timings'):
                raise ValueError("No timing information available")
            
            subtitles_path = f"temp/subtitles/{channel_type}_latest.srt"
            os.makedirs(os.path.dirname(subtitles_path), exist_ok=True)
            
            # Use utf-8 encoding for writing subtitles
            with open(subtitles_path, 'w', encoding='utf-8') as f:
                for i, timing in enumerate(self.sentence_timings, 1):
                    f.write(f"{i}\n")
                    f.write(f"{self._format_time(timing['start'])} --> {self._format_time(timing['end'])}\n")
                    f.write(f"{timing['text']}\n\n")
            
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

    async def _process_background_videos(self, channel_type):
        """Process background videos for the channel"""
        try:
            # First check for local background videos
            local_path = os.path.join("assets", "videos", channel_type)
            if os.path.exists(local_path):
                videos = [f for f in os.listdir(local_path) if f.endswith(('.mp4', '.mov'))]
                if videos:
                    # Convert to absolute paths
                    video_paths = [os.path.abspath(os.path.join(local_path, v)) for v in videos]
                    print(colored(f"Using {len(video_paths)} local videos from {local_path}", "green"))
                    return video_paths

            # Then try to get videos from Pexels
            search_terms = {
                'tech_humor': ['programming', 'coding', 'computer', 'technology'],
                'ai_money': ['business', 'success', 'technology', 'future'],
                'baby_tips': ['baby', 'parenting', 'family', 'children'],
                'quick_meals': ['cooking', 'food', 'kitchen', 'recipe'],
                'fitness_motivation': ['fitness', 'workout', 'exercise', 'gym']
            }
            
            terms = search_terms.get(channel_type, ['background', 'abstract'])
            video_urls = []
            
            # Create channel directory if it doesn't exist
            channel_dir = os.path.join("assets", "videos", channel_type)
            os.makedirs(channel_dir, exist_ok=True)
            
            for term in terms:
                try:
                    urls = search_for_stock_videos(
                        query=term,
                        api_key=self.pexels_api_key,
                        it=2,
                        min_dur=3
                    )
                    if urls:
                        for url in urls:
                            # Save directly to channel directory
                            saved_path = save_video(url, channel_dir)
                            if saved_path:
                                video_urls.append(saved_path)
                    
                    if len(video_urls) >= 4:
                        break
                    
                except Exception as e:
                    print(colored(f"Warning: Search failed for '{term}': {str(e)}", "yellow"))
                    continue
            
            if video_urls:
                return video_urls
            
            # If no videos found, create and use a default background
            default_path = os.path.join("assets", "videos", "default_background.mp4")
            if not os.path.exists(default_path):
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                from moviepy.editor import ColorClip
                clip = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=60)
                clip.write_videofile(default_path, fps=30)
            return [default_path]
            
        except Exception as e:
            print(colored(f"Background video processing failed: {str(e)}", "red"))
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