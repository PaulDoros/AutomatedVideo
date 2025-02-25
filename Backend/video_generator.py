from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
from termcolor import colored
from content_validator import ContentValidator, ScriptGenerator
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, AudioFileClip, concatenate_audioclips, AudioClip
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
from pydub import AudioSegment
import openai

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

class AudioManager:
    def __init__(self):
        self.voices = {
            'tech_humor': {
                'voices': {
                    'nova': {'weight': 3, 'description': 'Energetic female, perfect for tech humor'},
                    'echo': {'weight': 3, 'description': 'Dynamic male, great for casual tech content'}
                },
                'style': 'humorous'
            },
            'ai_money': {
                'voices': {
                    'onyx': {'weight': 3, 'description': 'Professional male voice'},
                    'shimmer': {'weight': 2, 'description': 'Clear female voice'}
                },
                'style': 'professional'
            },
            'default': {
                'voices': {
                    'echo': {'weight': 2, 'description': 'Balanced male voice'},
                    'nova': {'weight': 2, 'description': 'Engaging female voice'}
                },
                'style': 'casual'
            }
        }
        
        self.voiceovers_dir = "temp/tts"
        os.makedirs(self.voiceovers_dir, exist_ok=True)

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

        # Define available voices for variety
        self.voice_options = {
            'male_professional': ['p226', 'p227', 'p232'],
            'male_casual': ['p273', 'p274', 'p276'],
            'female_professional': ['p238', 'p243', 'p244'],
            'female_casual': ['p248', 'p251', 'p294'],
        }
        
        # Map channels to voice types
        self.channel_voices = {
            'tech_humor': ['male_casual', 'female_casual'],
            'ai_money': ['male_professional', 'female_professional'],
            'baby_tips': ['female_casual', 'female_professional'],
            'quick_meals': ['female_casual', 'male_casual'],
            'fitness_motivation': ['male_professional', 'male_casual']
        }

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
            
            # Get background videos using script content
            background_paths = await self._process_background_videos(channel_type, script)
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
        """Generate TTS using OpenAI's enhanced voices with cost optimization"""
        try:
            audio_manager = AudioManager()
            voice, style = audio_manager.select_voice(channel_type)
            
            print(colored(f"Using OpenAI voice: {voice} (style: {style})", "blue"))
            
            # Process script with better sentence splitting
            sentences = []
            for line in script.split('\n'):
                # Clean and split into natural phrases
                phrases = re.split(r'([.!?]+)', line.strip())
                for i in range(0, len(phrases)-1, 2):
                    phrase = (phrases[i] + phrases[i+1]).strip()
                    if phrase:
                        sentences.append(phrase)
            
            audio_clips = []
            timings = []
            current_time = 0
            
            # Use TTS HD for better quality at reasonable cost
            client = openai.OpenAI()
            
            for sentence in sentences:
                try:
                    temp_path = f"temp/tts/temp_{uuid.uuid4()}.mp3"
                    
                    # Generate TTS with OpenAI
                    response = client.audio.speech.create(
                        model="tts-1-hd",  # Use HD model for better quality
                        voice=voice,
                        input=sentence,
                        speed=1.1 if style == 'humorous' else 1.0
                    )
                    
                    # Save and process audio
                    with open(temp_path, "wb") as f:
                        response.stream_to_file(temp_path)
                    
                    # Load and enhance audio
                    audio = AudioSegment.from_mp3(temp_path)
                    enhanced_audio = audio_manager.enhance_audio(audio)
                    
                    # Add natural pauses
                    pause_duration = 200  # Base pause
                    if sentence.endswith('?'):
                        pause_duration = 300
                    elif sentence.endswith('!'):
                        pause_duration = 250
                    
                    silence = AudioSegment.silent(duration=pause_duration)
                    final_audio = enhanced_audio + silence
                    
                    # Save enhanced audio
                    final_path = f"temp/tts/enhanced_{uuid.uuid4()}.mp3"
                    final_audio.export(final_path, format="mp3", bitrate="192k")
                    
                    # Create MoviePy audio clip
                    audio_clip = AudioFileClip(final_path)
                    audio_clips.append(audio_clip)
                    
                    # Store timing information
                    duration = len(final_audio) / 1000.0
                    timings.append({
                        'text': sentence,
                        'start': current_time,
                        'end': current_time + duration
                    })
                    current_time += duration
                    
                    # Clean up temp files
                    os.remove(temp_path)
                    
                except Exception as e:
                    print(colored(f"Warning: Failed to process sentence: {e}", "yellow"))
                    continue
            
            # Combine all clips
            final_path = f"temp/tts/{channel_type}_latest.mp3"
            final_audio = concatenate_audioclips(audio_clips)
            final_audio.write_audiofile(final_path, fps=44100, bitrate="192k")
            
            self.sentence_timings = timings
            return final_path
            
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
            prompt = f"""
            Analyze this script and suggest 4-6 specific video scenes that would match the content well.
            Focus on visual elements that would enhance the story.
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
            4. coffee cup steam programming - Creates atmosphere
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
        """Process background videos with AI suggestions and content analysis"""
        try:
            # First check for local videos
            local_path = os.path.join("assets", "videos", channel_type)
            if os.path.exists(local_path):
                videos = [f for f in os.listdir(local_path) if f.endswith(('.mp4', '.mov'))]
                if videos:
                    video_paths = [os.path.abspath(os.path.join(local_path, v)) for v in videos]
                    print(colored(f"Using {len(video_paths)} local videos from {local_path}", "green"))
                    return video_paths

            # Get video suggestions from GPT
            gpt_suggestions = await self._get_video_suggestions(script)
            
            # Also get our content analysis
            script_analysis = self._analyze_script_content(script)
            analysis_terms = self._generate_search_terms(script_analysis)
            
            # Combine and prioritize search terms
            search_terms = []
            
            # First add GPT suggestions as they're more specific
            if gpt_suggestions:
                search_terms.extend([s['term'] for s in gpt_suggestions])
            
            # Then add our analyzed terms as backup
            search_terms.extend(analysis_terms)
            
            # Deduplicate while preserving order
            search_terms = list(dict.fromkeys(search_terms))
            
            video_urls = []
            max_videos = 8
            
            # Allocate more videos for GPT suggestions
            videos_per_term = 2 if len(search_terms) > 4 else 3
            
            # Create channel directory
            channel_dir = os.path.join("assets", "videos", channel_type)
            os.makedirs(channel_dir, exist_ok=True)
            
            # Search for each term
            for term in search_terms:
                if len(video_urls) >= max_videos:
                    break
                
                print(colored(f"Searching for videos matching: {term}", "blue"))
                urls = await self._search_and_save_videos(term, channel_dir, videos_per_term)
                
                if urls:
                    video_urls.extend(urls)
                    print(colored(f"Found {len(urls)} videos for '{term}'", "green"))
                else:
                    print(colored(f"No videos found for '{term}'", "yellow"))
            
            if video_urls:
                # Randomize but ensure we have variety
                grouped_videos = {}
                for url in video_urls:
                    term = next((t for t in search_terms if t.lower() in url.lower()), 'other')
                    grouped_videos.setdefault(term, []).append(url)
                
                # Take videos from each group to ensure variety
                final_videos = []
                while len(final_videos) < max_videos and grouped_videos:
                    for term in list(grouped_videos.keys()):
                        if grouped_videos[term]:
                            final_videos.append(grouped_videos[term].pop(0))
                            if not grouped_videos[term]:
                                del grouped_videos[term]
                        if len(final_videos) >= max_videos:
                            break
                
                return final_videos
            
            return self._create_default_background()

        except Exception as e:
            print(colored(f"Background video processing failed: {str(e)}", "red"))
            return None

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

    def _create_default_background(self):
        """Create a default black background video"""
        try:
            # Create default background path
            default_path = os.path.join("assets", "videos", "default_background.mp4")
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            
            # Create a black background clip
            clip = ColorClip(
                size=(1080, 1920),  # Vertical format (9:16)
                color=(0, 0, 0),
                duration=60
            )
            
            # Write the video file
            clip.write_videofile(
                default_path,
                fps=30,
                codec='libx264',
                audio=False,
                threads=4
            )
            
            # Clean up
            clip.close()
            
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

def combine_videos(video_paths, audio_duration, target_duration, n_threads=2):
    """
    Combine videos by clipping segments from each video to match audio timing.
    Each video will contribute a portion to the final video.
    """
    try:
        clips = []
        
        # Get audio clip to analyze timing
        audio_clip = AudioFileClip("temp/tts/tech_humor_latest.mp3")
        total_duration = audio_clip.duration
        
        print(colored("\n=== Video Combination Debug Info ===", "blue"))
        print(colored(f"Audio duration: {total_duration:.2f}s", "cyan"))
        print(colored(f"Number of videos: {len(video_paths)}", "cyan"))
        print(colored(f"Target duration: {target_duration:.2f}s", "cyan"))
        
        # Calculate segment duration for each video
        num_videos = len(video_paths)
        segment_duration = total_duration / num_videos
        
        print(colored("\n=== Segment Calculations ===", "blue"))
        print(colored(f"Segment duration per video: {segment_duration:.2f}s", "cyan"))
        
        # Track our position in the timeline
        current_time = 0
        
        for i, video_path in enumerate(video_paths):
            try:
                print(colored(f"\n=== Processing Video {i+1}/{num_videos} ===", "blue"))
                print(colored(f"Video path: {video_path}", "cyan"))
                
                # Load video
                video = VideoFileClip(video_path)
                print(colored(f"Original video duration: {video.duration:.2f}s", "cyan"))
                print(colored(f"Original video size: {video.size}", "cyan"))
                
                # Calculate this segment's duration
                if i == num_videos - 1:
                    this_segment_duration = total_duration - current_time
                else:
                    this_segment_duration = segment_duration
                
                print(colored(f"Target segment duration: {this_segment_duration:.2f}s", "cyan"))
                
                # Select portion of video to use
                if video.duration > this_segment_duration:
                    max_start = video.duration - this_segment_duration
                    video_start = random.uniform(0, max_start)
                    print(colored(f"Video longer than needed:", "yellow"))
                    print(colored(f"- Max start time: {max_start:.2f}s", "yellow"))
                    print(colored(f"- Selected start: {video_start:.2f}s", "yellow"))
                    print(colored(f"- Will use: {video_start:.2f}s to {video_start + this_segment_duration:.2f}s", "yellow"))
                    
                    # Create clip from portion of video
                    clip = (video
                           .subclip(video_start, video_start + this_segment_duration)
                           .resize(width=1080)
                           .set_position(("center", "center")))
                else:
                    print(colored(f"Video shorter than needed - will loop", "yellow"))
                    print(colored(f"- Original duration: {video.duration:.2f}s", "yellow"))
                    print(colored(f"- Need duration: {this_segment_duration:.2f}s", "yellow"))
                    
                    # Create looped clip
                    clip = (video
                           .loop(duration=this_segment_duration)
                           .resize(width=1080)
                           .set_position(("center", "center")))
                
                print(colored(f"Created clip duration: {clip.duration:.2f}s", "green"))
                
                # Add transition effects
                if i > 0:
                    print(colored("Adding entrance crossfade", "cyan"))
                    clip = clip.crossfadein(0.3)
                if i < num_videos - 1:
                    print(colored("Adding exit crossfade", "cyan"))
                    clip = clip.crossfadeout(0.3)
                
                clips.append(clip)
                current_time += this_segment_duration
                print(colored(f"Current total time: {current_time:.2f}s / {total_duration:.2f}s", "green"))
                
            except Exception as e:
                print(colored(f"Error processing video {i+1}:", "red"))
                print(colored(f"Error details: {str(e)}", "red"))
                print(colored("Creating fallback black clip", "yellow"))
                color_clip = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=this_segment_duration)
                clips.append(color_clip)
                current_time += this_segment_duration
        
        print(colored("\n=== Finalizing Video ===", "blue"))
        print(colored(f"Number of clips to combine: {len(clips)}", "cyan"))
        
        # Combine all clips
        final_clip = concatenate_videoclips(clips, method="compose")
        print(colored(f"Final clip duration: {final_clip.duration:.2f}s", "cyan"))
        
        # Ensure exact duration match
        if abs(final_clip.duration - total_duration) > 0.1:
            print(colored(f"Duration mismatch of {abs(final_clip.duration - total_duration):.2f}s - adjusting...", "yellow"))
            final_clip = final_clip.set_duration(total_duration)
        
        # Save combined video
        output_path = "temp_combined.mp4"
        print(colored(f"\nSaving to: {output_path}", "blue"))
        
        final_clip.write_videofile(
            output_path,
            threads=n_threads,
            codec='libx264',
            audio=False,
            fps=30
        )
        
        # Clean up
        print(colored("\nCleaning up resources...", "blue"))
        audio_clip.close()
        for clip in clips:
            clip.close()
        final_clip.close()
        
        return output_path
        
    except Exception as e:
        print(colored("\n=== Error in combine_videos ===", "red"))
        print(colored(f"Error type: {type(e).__name__}", "red"))
        print(colored(f"Error details: {str(e)}", "red"))
        return None 