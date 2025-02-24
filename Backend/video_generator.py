from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
from termcolor import colored
from content_validator import ContentValidator, ScriptGenerator
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
from tiktokvoice import tts  # Change this import
from video import generate_video, generate_subtitles, combine_videos, save_video
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
            
            # Create all necessary directories
            os.makedirs("temp/tts", exist_ok=True)
            os.makedirs("temp/subtitles", exist_ok=True)
            os.makedirs("temp/videos", exist_ok=True)
            os.makedirs("output/videos", exist_ok=True)
            
            # Load script
            with open(script_file, 'r') as f:
                data = json.load(f)
                script = data.get('script', '')
                preview = data.get('preview', '')
            
            if not script:
                raise ValueError("Script is empty")
            
            # Define paths
            tts_path = f"temp/tts/{channel_type}_latest.mp3"
            subtitles_path = f"temp/subtitles/{channel_type}_latest.srt"
            combined_video_path = f"temp/videos/{channel_type}_combined.mp4"
            
            # 1. Generate TTS audio
            print(colored("\nGenerating TTS audio...", "blue"))
            voice_config = self.get_voice_for_channel(channel_type)
            success, audio_clips = await generate_tts(
                text=script if not preview else preview,
                voice=voice_config['voice'],
                output_path=tts_path
            )
            if not success:
                raise Exception("TTS generation failed")
            print(colored("✓ TTS generated", "green"))
            
            # 2. Generate subtitles
            print(colored("\nGenerating subtitles...", "blue"))
            sentences = (script if not preview else preview).split('\n')
            subtitles = generate_subtitles(
                audio_path=tts_path,
                sentences=sentences,
                audio_clips=audio_clips,
                voice=voice_config['lang']
            )
            if not subtitles:
                raise Exception("Subtitles generation failed")
            
            # Copy subtitles to expected location
            shutil.copy2(subtitles, subtitles_path)
            print(colored("✓ Subtitles generated", "green"))

            # 3. Download and combine background videos
            print(colored("\nDownloading background videos...", "blue"))
            # Get video duration from audio
            audio_clip = AudioFileClip(tts_path)
            video_duration = audio_clip.duration
            audio_clip.close()

            # Search for relevant background videos
            search_terms = {
                'tech_humor': ['programming', 'coding', 'computer', 'technology'],
                'ai_money': ['business', 'computer', 'technology', 'success'],
                'baby_tips': ['baby', 'parenting', 'family', 'children'],
                'quick_meals': ['cooking', 'food', 'kitchen', 'healthy'],
                'fitness_motivation': ['workout', 'fitness', 'exercise', 'gym']
            }

            terms = search_terms.get(channel_type, ['background', 'abstract'])
            video_urls = []
            for term in terms:
                urls = search_for_stock_videos(
                    query=term,
                    api_key=self.pexels_api_key,
                    it=2,
                    min_dur=3
                )
                video_urls.extend(urls)
                if len(video_urls) >= 4:
                    break

            if not video_urls:
                raise Exception("No background videos found")

            # Download videos to temp/videos directory
            video_paths = []
            for url in video_urls:
                video_path = save_video(url, "temp/videos")
                video_paths.append(video_path)

            print(colored(f"✓ Downloaded {len(video_paths)} background videos", "green"))

            # Combine videos
            print(colored("\nCombining background videos...", "blue"))
            combined_video = combine_videos(
                video_paths=video_paths,
                max_duration=video_duration,
                max_clip_duration=10,
                threads=8
            )
            
            if not combined_video:
                raise Exception("Failed to combine background videos")
            
            # Move combined video to expected location
            shutil.move(combined_video, combined_video_path)
            print(colored("✓ Background videos combined", "green"))
            
            # 4. Generate final video
            print(colored("\nGenerating final video...", "blue"))
            params = {
                'combined_video_path': combined_video_path,
                'tts_path': tts_path,
                'subtitles_path': subtitles_path,
                'threads': 8,
                'subtitles_position': 'center,bottom',
                'text_color': 'white'
            }
            
            video_path = generate_video(**params)
            
            if not video_path:
                raise Exception("Video generation failed")
            
            # Generate unique filename and save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{channel_type}_{timestamp}.mp4"
            output_path = os.path.join("output/videos", output_filename)
            latest_path = os.path.join("output/videos", f"{channel_type}_latest.mp4")
            
            # Move video to output directory
            shutil.move(os.path.join("temp", video_path), output_path)
            
            # Update latest version
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.link(output_path, latest_path)
            
            print(colored("\n=== Video Generation Complete ===", "green"))
            print(colored(f"✓ Saved as: {output_filename}", "green"))
            return True
            
        except Exception as e:
            print(colored(f"\n=== Video Generation Failed ===\nError: {str(e)}", "red"))
            return False

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