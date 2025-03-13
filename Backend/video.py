import os
import uuid
import requests
from typing import List
from moviepy.editor import *
from termcolor import colored, cprint
from datetime import timedelta
from moviepy.video.fx.all import crop
from moviepy.video.tools.subtitles import SubtitlesClip
import numpy as np
from urllib.parse import quote
import random
from pathlib import Path
from moviepy.config import change_settings
from openai import OpenAI
import codecs
import json
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from datetime import datetime
import html
import traceback
import pysrt
import emoji
from moviepy.video.VideoClip import ImageClip
from emoji_data_python import emoji_data
from io import BytesIO
import re
import math
import time
import shutil
import subprocess
import asyncio

from music_provider import MusicProvider, music_provider
import concurrent.futures

# First activate your virtual environment and install pysrt:
# python -m pip install pysrt --no-cache-dir

# ===== LOGGING FUNCTIONS =====
def log_section(title, emoji="✨"):
    """Print a section header with emoji"""
    print("\n")
    cprint(f"=== {emoji} {title} {emoji} ===", "blue", attrs=["bold"])

def log_error(message, emoji="❌"):
    """Log an error message in red with emoji"""
    cprint(f"{emoji} {message}", "red")

def log_warning(message, emoji="⚠️"):
    """Log a warning message in yellow with emoji"""
    cprint(f"{emoji} {message}", "yellow")

def log_info(message, emoji="ℹ️"):
    """Log an info message in cyan with emoji"""
    cprint(f"{emoji} {message}", "cyan")

def log_success(message, emoji="✅"):
    """Log a success message in green with emoji"""
    cprint(f"{emoji} {message}", "green")

def log_processing(message, emoji="⏳"):
    """Log a processing message in magenta with emoji"""
    cprint(f"{emoji} {message}", "magenta")

def log_progress(current, total, prefix="", suffix="", length=30):
    """Show a progress bar with percentage"""
    percent = int(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if current == total:
        print()

def log_debug(message, emoji="🔍"):
    """Log a debug message - only shown when DEBUG is enabled"""
    if os.environ.get("DEBUG_LOGS", "0") == "1":
        cprint(f"{emoji} {message}", "white")

def log_step(step_num, total_steps, message, emoji="🔄", start_time=None, end_time=None):
    """Log a step in a multi-step process"""
    if start_time is not None:
        message = f"{message} (Started)"
    elif end_time is not None and start_time is not None:
        duration = end_time - start_time
        message = f"{message} (Completed in {duration:.1f}s)"
    elif end_time is not None:
        message = f"{message} (Completed)"
    
    cprint(f"{emoji} Step {step_num}/{total_steps}: {message}", "cyan")

def log_separator(char="=", length=50):
    """Print a separator line"""
    print(char * length)

def log_highlight(message, emoji="🔆"):
    """Highlight important information"""
    log_separator()
    cprint(f"{emoji} {message} {emoji}", "green", attrs=["bold"])
    log_separator()

def log_result(title, path, emoji="📁"):
    """Log file output result with absolute path"""
    log_separator()
    cprint(f"{emoji} {title}:", "green", attrs=["bold"])
    cprint(f"{os.path.abspath(path)}", "cyan")
    log_separator()

# ===== END LOGGING FUNCTIONS ====

def save_video(video_url: str, directory: str) -> str:
    """Saves a video from URL to specified directory"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp4"
        video_path = os.path.join(directory, filename)
        
        # Download and save the video
        response = requests.get(video_url)
        response.raise_for_status()
        
        with open(video_path, "wb") as f:
            f.write(response.content)
        
        # Verify file exists and has content
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            print(colored(f"✓ Saved video to {video_path}", "green"))
            return video_path
        else:
            raise ValueError("Video file not saved correctly")
            
    except Exception as e:
        print(colored(f"Error saving video: {str(e)}", "red"))
        return None

def format_time(seconds):
    """Format time in seconds to SRT format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def generate_subtitles(script: str, audio_path: str, content_type: str = None) -> str:
    """Generate SRT subtitles from script with improved timing for better synchronization"""
    try:
        from video import log_info, log_success, log_warning, log_error, log_step
        
        log_info("Generating synchronized subtitles")
        
        # First, check if we have a cleaned script in the JSON file
        json_path = f"cache/scripts/{content_type}_latest.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    script_data = json.load(f)
                    if "cleaned_script" in script_data and script_data["cleaned_script"].strip():
                        script = script_data["cleaned_script"]
                        log_success("Using pre-cleaned script from JSON file for subtitles")
            except Exception as e:
                log_warning(f"Error reading JSON file: {str(e)}")
                # Continue with the provided script
        
        # Get audio duration
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        audio.close()
        
        # Create temp directory
        os.makedirs("temp/subtitles", exist_ok=True)
        subtitles_path = "temp/subtitles/generated_subtitles.srt"
        
        # Process script to create better subtitles
        log_info("Processing script for enhanced subtitle synchronization")
        
        # Process script line by line to preserve original emoji positioning
        lines = script.replace('"', '').split('\n')
        raw_sentences = []
        
        # Define section headers to remove (both with and without asterisks)
        section_headers = [
            "**Hook:**", "**Problem/Setup:**", "**Solution/Development:**", 
            "**Result/Punchline:**", "**Call to action:**",
            "Hook:", "Problem/Setup:", "Solution/Development:", 
            "Result/Punchline:", "Call to action:",
            "**Script:**", "Script:"
        ]
        
        # Define patterns to filter out
        special_patterns = ["---", "***", "**", "##"]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip lines that only contain special characters
            if line in special_patterns or line.strip("-*#") == "":
                continue
                
            # Remove any numeric prefixes (like "1.")
            line = re.sub(r'^\d+\.\s*', '', line)
            
            # Remove section headers
            skip_line = False
            for header in section_headers:
                if line.strip().startswith(header):
                    # Extract content after the header
                    content = line[line.find(header) + len(header):].strip()
                    if content:  # If there's content after the header, use it
                        line = content
                    else:  # If it's just a header line, skip it entirely
                        skip_line = True
                    break
            
            if not skip_line and line.strip():
                raw_sentences.append(line)
        
        # Split into more granular sentences for better sync
        sentences = []
        max_chars_per_subtitle = 70  # Shorter for better readability and sync
        
        # More sophisticated sentence splitting for better sync with audio
        for sentence in raw_sentences:
            # Split by natural pause points first
            chunks = re.split(r'([.!?:;,])', sentence)
            rebuilt_chunks = []
            
            current = ""
            for i in range(0, len(chunks) - 1, 2):
                if i < len(chunks):
                    part = chunks[i]
                    punct = chunks[i+1] if i+1 < len(chunks) else ""
                    
                    if len(current) + len(part) + 1 <= max_chars_per_subtitle:
                        current += part + punct
                    else:
                        if current:
                            rebuilt_chunks.append(current.strip())
                        current = part + punct
            
            if current:
                rebuilt_chunks.append(current.strip())
                
            # If no chunks were created, handle the sentence as a whole
            if not rebuilt_chunks:
                # Split by commas or spaces for better synchronization
                if len(sentence) > max_chars_per_subtitle:
                    words = sentence.split()
                    current = ""
                    for word in words:
                        if len(current) + len(word) + 1 <= max_chars_per_subtitle:
                            current += " " + word if current else word
                        else:
                            if current:
                                rebuilt_chunks.append(current.strip())
                            current = word
                    if current:
                        rebuilt_chunks.append(current.strip())
                else:
                    rebuilt_chunks.append(sentence)
            
            sentences.extend([chunk for chunk in rebuilt_chunks if chunk.strip()])
        
        # Now create the subtitle timing based on audio duration
        subtitle_count = len(sentences)
        log_info(f"Creating {subtitle_count} subtitle segments for precise synchronization")
        
        # Calculate timing based on audio duration and sentence length
        # Improved algorithm for more natural timing
        if subtitle_count > 0:
            # Calculate pauses between sentences to create natural rhythm
            base_duration = total_duration / subtitle_count
            
            # For shorter segments, allocate more time to each segment
            segment_durations = []
            
            # First pass: allocate duration based on text length
            total_chars = sum(len(s) for s in sentences)
            for sentence in sentences:
                # Proportion of total characters, with minimum time
                char_proportion = len(sentence) / total_chars if total_chars > 0 else 1/subtitle_count
                duration = max(1.0, char_proportion * total_duration)
                segment_durations.append(duration)
            
            # Adjust to match total duration
            scale_factor = total_duration / sum(segment_durations)
            segment_durations = [d * scale_factor for d in segment_durations]
            
            # Create subtitle timings
            subtitle_timings = []
            current_time = 0.0
            
            for i, (sentence, duration) in enumerate(zip(sentences, segment_durations)):
                start_time = current_time
                end_time = start_time + duration
                subtitle_timings.append((start_time, end_time, sentence))
                current_time = end_time
            
            # Generate the actual subtitle file with improved timing
            with open(subtitles_path, "w", encoding="utf-8") as f:
                for i, (start_time, end_time, sentence) in enumerate(subtitle_timings):
                    # Format times for SRT
                    start_formatted = format_time(start_time)
                    end_formatted = format_time(end_time)
                    
                    # Write subtitle entry
                    f.write(f"{i+1}\n")
                    f.write(f"{start_formatted} --> {end_formatted}\n")
                    f.write(f"{sentence.strip()}\n\n")
            
            log_success(f"Created {subtitle_count} precisely timed subtitle segments")
            return subtitles_path
        else:
            log_warning("No valid subtitle content found")
            return None
    
    except Exception as e:
        log_error(f"Error generating subtitles: {str(e)}")
        traceback.print_exc()
        return None

def combine_videos(video_paths, audio_duration, target_duration, n_threads=4):
    """Combine multiple videos with smooth transitions for YouTube Shorts"""
    from video import log_info, log_success, log_warning, log_error, log_step
    
    try:
        log_section("Video Combination", "🔄")
        
        # Validate input
        if not video_paths:
            log_error("No videos provided")
            return None
        
        log_info(f"Audio duration: {audio_duration:.2f}s")
        log_info(f"Target duration: {target_duration:.2f}s")
        
        # Adjust target duration to avoid black frames at the end
        adjusted_duration = target_duration - 0.05
        log_info(f"Adjusted to {adjusted_duration:.2f}s to prevent black end frame")
        
        # If only one video is provided, use it directly
        if len(video_paths) == 1:
            log_info("Only one video provided, using it directly")
            video_path = video_paths[0]
            if isinstance(video_path, dict):
                video_path = video_path.get('path')
                
            if not os.path.exists(video_path):
                log_error(f"Video file does not exist: {video_path}")
                return None
                
            try:
                # Load the video
                video = VideoFileClip(video_path, fps_source="fps")
                
                # Resize to vertical format
                video = resize_to_vertical(video)
                
                # Adjust duration if needed
                if video.duration < adjusted_duration:
                    log_info(f"Video too short ({video.duration:.2f}s), looping")
                    video = vfx.loop(video, duration=adjusted_duration)
                elif video.duration > adjusted_duration:
                    log_info(f"Video too long ({video.duration:.2f}s), trimming")
                    video = video.subclip(0, adjusted_duration)
                
                # Write to file
                output_path = "temp_combined.mp4"
                video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio=False,
                    fps=30,
                    preset="medium",
                    threads=n_threads
                )
                video.close()
                return output_path
            except Exception as e:
                log_error(f"Error processing single video: {str(e)}")
                return None
        
        # For multiple videos, we'll use a different approach
        # Calculate segment duration
        num_videos = len(video_paths)
        segment_duration = adjusted_duration / num_videos
        log_info(f"Number of videos: {num_videos}")
        log_info(f"Segment duration per video: {segment_duration:.2f}s")
        
        # Create a temporary directory for segment files
        temp_dir = "temp/segments"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process each video
        valid_segments = []
        
        for i, video_path in enumerate(video_paths):
            log_step(i+1, num_videos, f"Processing video {i+1}/{num_videos}")
            
            if isinstance(video_path, dict):
                video_path = video_path.get('path')
            
            log_info(f"Video path: {video_path}")
            
            # Check if the video file exists
            if not os.path.exists(video_path):
                log_warning(f"Video file doesn't exist, skipping: {video_path}")
                continue
                
            try:
                # Load video
                video = VideoFileClip(video_path, fps_source="fps")
                
                # Handle short videos by looping them instead of skipping
                if video.duration < segment_duration:
                    log_info(f"Video shorter than segment duration ({video.duration:.2f}s < {segment_duration:.2f}s), looping")
                    loop_count = math.ceil(segment_duration / video.duration)
                    # Create looped version using MoviePy's loop method
                    video = vfx.loop(video, duration=segment_duration)
                elif video.duration > segment_duration:
                    # Take a random segment if the video is longer than needed
                    max_start = max(0, video.duration - segment_duration)
                    start_time = random.uniform(0, max_start)
                    video = video.subclip(start_time, start_time + segment_duration)
                    log_info(f"Using segment from {start_time:.2f}s to {start_time + segment_duration:.2f}s")
                
                # Resize to vertical format for YouTube Shorts
                video = resize_to_vertical(video)
                
                # Set segment path
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
                
                # Write segment file without audio
                video.without_audio().write_videofile(
                    segment_path,
                    codec="libx264",
                    fps=30,
                    preset="medium",
                    threads=n_threads
                )
                
                video.close()
                valid_segments.append(segment_path)
                log_success(f"Created segment {i+1}: {segment_path}")
                
            except Exception as e:
                log_warning(f"Error processing video {i+1}: {str(e)}")
                # Try to continue with other videos
                continue
        
        # Check if we have any valid segments
        if not valid_segments:
            log_error("No valid segments created")
            return None
            
        log_info(f"Created {len(valid_segments)} video segments")
        
        # If we have more than one segment, concatenate them
        if len(valid_segments) > 1:
            # Create a list file for ffmpeg
            segments_file = os.path.join(temp_dir, "segments.txt")
            with open(segments_file, "w") as f:
                for segment in valid_segments:
                    # Use absolute paths to avoid issues
                    abs_path = os.path.abspath(segment)
                    f.write(f"file '{abs_path}'\n")
            
            # Concatenate segments using ffmpeg
            output_path = "temp_combined.mp4"
            log_info("Concatenating segments with ffmpeg")
            
            try:
                subprocess.run([
                    "ffmpeg", 
                    "-y", 
                    "-f", "concat", 
                    "-safe", "0", 
                    "-i", segments_file, 
                    "-c", "copy", 
                    output_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                log_success("Successfully concatenated segments")
                return output_path
            except subprocess.CalledProcessError as e:
                log_error(f"Error concatenating segments: {str(e)}")
                # Fallback to the first valid segment if concatenation fails
                if valid_segments:
                    log_warning("Using first segment as fallback")
                    return valid_segments[0]
                return None
        else:
            # If only one segment was created, use it
            log_info("Only one valid segment created, using it directly")
            return valid_segments[0]
            
    except Exception as e:
        log_error(f"Error combining videos: {str(e)}")
        return None

def get_background_music(content_type: str, duration: float = None) -> str:
    """Get background music based on content type"""
    try:
        # Use the new music provider to get music
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, use create_task or run_coroutine_threadsafe
            if loop.is_running():
                # Use a synchronous approach instead
                music_dir = f"assets/music/{content_type}"
                if os.path.exists(music_dir):
                    music_files = [f for f in os.listdir(music_dir) if f.endswith('.mp3')]
                    if music_files:
                        music_path = os.path.join(music_dir, random.choice(music_files))
                        print(colored(f"✓ Using default music for {content_type}", "green"))
                        return music_path
            else:
                # Loop exists but not running
                music_path = loop.run_until_complete(music_provider.get_music_for_channel(content_type, duration))
        except RuntimeError:
            # No event loop, safe to use asyncio.run()
            music_path = asyncio.run(music_provider.get_music_for_channel(content_type, duration))
        
        if music_path and os.path.exists(music_path):
            print(colored(f"✓ Using background music for {content_type}", "green"))
            return music_path
        
        # Fallback to the old method if the new provider fails
        # Load Pixabay API key from env
        pixabay_api_key = os.getenv('PIXABAY_API_KEY')
        if not pixabay_api_key:
            print(colored("Warning: No Pixabay API key found, skipping music", "yellow"))
            return ""  # Return empty string instead of None

        # Map content types to music search terms
        music_moods = {
            'tech_humor': ['funny', 'upbeat', 'quirky'],
            'ai_money': ['corporate', 'technology', 'success'],
            'baby_tips': ['gentle', 'peaceful', 'soft'],
            'quick_meals': ['cooking', 'upbeat', 'positive'],
            'fitness_motivation': ['energetic', 'workout', 'motivation']
        }

        # Get search terms for content type
        search_terms = music_moods.get(content_type, ['background', 'ambient'])
        
        # Create music directory if it doesn't exist
        music_dir = os.path.join("assets", "music", content_type)
        os.makedirs(music_dir, exist_ok=True)

        # Try each search term until we find suitable music
        for term in search_terms:
            try:
                # Search Pixabay API
                url = f"https://pixabay.com/api/videos/music/?key={pixabay_api_key}&q={quote(term)}&category=music"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()

                if not data.get('hits'):
                    continue

                # Filter tracks by duration if specified
                tracks = data['hits']
                if duration:
                    tracks = [t for t in tracks if abs(float(t['duration']) - duration) < 30]

                if not tracks:
                    continue

                # Select a random track
                track = random.choice(tracks)
                music_url = track['audio']
                
                # Download the music
                music_path = os.path.join(music_dir, f"{uuid.uuid4()}.mp3")
                response = requests.get(music_url)
                response.raise_for_status()

                with open(music_path, 'wb') as f:
                    f.write(response.content)

                print(colored(f"✓ Downloaded background music: {term}", "green"))
                return music_path

            except Exception as e:
                print(colored(f"Warning: Failed to get music for term '{term}': {str(e)}", "yellow"))
                continue

        print(colored("Could not find suitable background music", "yellow"))
        return ""  # Return empty string instead of None

    except Exception as e:
        print(colored(f"Error getting background music: {str(e)}", "red"))
        return ""  # Return empty string instead of None

def mix_audio(voice_path: str, music_path: str, output_path: str, music_volume: float = 0.3, 
             fade_in: float = 2.0, fade_out: float = 3.0) -> str:
    """
    Mix voice audio with background music with enhanced quality and balance.
    
    Args:
        voice_path: Path to voice audio file
        music_path: Path to background music file
        output_path: Path to save mixed audio
        music_volume: Volume of background music (0.0 to 1.0)
        fade_in: Duration of fade in for music (seconds)
        fade_out: Duration of fade out for music (seconds)
        
    Returns:
        Path to mixed audio file
    """
    try:
        log_info(f"Mixing audio with music volume: {music_volume} (fade in: {fade_in}s, fade out: {fade_out}s)")
        
        # Check if voice path exists
        if not os.path.exists(voice_path):
            log_error(f"Voice file not found: {voice_path}")
            return voice_path
            
        # Check if music path exists and is valid
        if not music_path or not os.path.exists(music_path):
            log_warning(f"No valid music path provided: {music_path}")
            log_info("Using voice audio only")
            # Just copy the voice file to the output path
            shutil.copy(voice_path, output_path)
            return output_path
        
        # Simple approach using pydub
        try:
            from pydub import AudioSegment
            
            # Load audio files
            log_info("Loading voice audio file...")
            voice = AudioSegment.from_file(voice_path)
            
            log_info("Loading music audio file...")
            music = AudioSegment.from_file(music_path)
            
            # Get durations
            voice_duration_ms = len(voice)
            music_duration_ms = len(music)
            
            log_info(f"Voice duration: {voice_duration_ms/1000:.2f}s")
            log_info(f"Music duration: {music_duration_ms/1000:.2f}s")
            
            # Adjust music volume
            music = music - (20 - (music_volume * 20))  # Convert 0.0-1.0 to 0-20 dB reduction
            
            # Apply fade effects
            if fade_in > 0:
                fade_in_ms = int(fade_in * 1000)
                music = music.fade_in(fade_in_ms)
            
            if fade_out > 0:
                fade_out_ms = int(fade_out * 1000)
                music = music.fade_out(fade_out_ms)
            
            # Trim or loop music to match voice duration
            if music_duration_ms > voice_duration_ms:
                log_info(f"Music longer than voice, trimming...")
                music = music[:voice_duration_ms]
            elif music_duration_ms < voice_duration_ms:
                log_info(f"Music shorter than voice, looping...")
                # Calculate how many times to loop
                loops_needed = int(voice_duration_ms / music_duration_ms) + 1
                looped_music = music
                for _ in range(loops_needed - 1):
                    looped_music += music
                music = looped_music[:voice_duration_ms]
            
            # Overlay music with voice
            log_info("Mixing voice and music...")
            mixed = voice.overlay(music, position=0)
            
            # Export mixed audio
            log_info(f"Exporting mixed audio to {output_path}...")
            mixed.export(output_path, format="mp3", bitrate="192k")
            
            log_success(f"Successfully mixed voice and music using pydub")
            return output_path
            
        except ImportError:
            log_warning("Pydub not available, falling back to MoviePy")
            
            # Fallback to MoviePy
            try:
                # Load audio clips
                voice = AudioFileClip(voice_path)
                music = AudioFileClip(music_path)
                
                voice_duration = voice.duration
                music_duration = music.duration
                
                log_info(f"Voice duration: {voice_duration:.2f}s")
                log_info(f"Music duration: {music_duration:.2f}s")
                
                # Apply volume adjustment to music
                music = music.volumex(music_volume)
                
                # Apply fades to music
                if fade_in > 0:
                    music = music.audio_fadein(fade_in)
                
                if fade_out > 0:
                    music = music.audio_fadeout(fade_out)
                
                # Handle music duration relative to voice
                if music_duration < voice_duration:
                    log_info(f"Music shorter than voice, looping...")
                    # Loop music to match voice duration
                    music = afx.audio_loop(music, duration=voice_duration)
                elif music_duration > voice_duration:
                    log_info(f"Music longer than voice, trimming...")
                    # Trim music to match voice duration
                    music = music.subclip(0, voice_duration)
                
                # Create a list of audio clips
                audio_clips = [music, voice]
                
                # Write each clip to a temporary file
                temp_files = []
                for i, clip in enumerate(audio_clips):
                    temp_file = f"temp/audio_part_{i}_{uuid.uuid4()}.wav"
                    clip.write_audiofile(temp_file, fps=44100, nbytes=2, codec='pcm_s16le')
                    temp_files.append(temp_file)
                
                # Load the clips back as AudioSegment objects
                from pydub import AudioSegment
                segments = [AudioSegment.from_file(f) for f in temp_files]
                
                # Mix the segments
                mixed = segments[0]
                for segment in segments[1:]:
                    mixed = mixed.overlay(segment)
                
                # Export the mixed audio
                mixed.export(output_path, format="mp3", bitrate="192k")
                
                # Clean up temp files
                for f in temp_files:
                    try:
                        os.remove(f)
                    except:
                        pass
                
                log_success(f"Successfully mixed voice and music using MoviePy and pydub")
                return output_path
                
            except Exception as moviepy_error:
                log_warning(f"Error in MoviePy mixing: {str(moviepy_error)}")
                
                # Last resort: just use the voice audio
                log_info("Using voice audio only as last resort")
                shutil.copy(voice_path, output_path)
                return output_path
    
    except Exception as e:
        log_error(f"Error mixing audio: {str(e)}")
        traceback.print_exc()
        log_warning(f"Falling back to voice audio only")
        
        # Make sure we actually copy the voice audio to the output path
        try:
            # Copy the voice audio to the output path
            shutil.copy(voice_path, output_path)
            log_info(f"Copied voice audio to {output_path}")
            return output_path
        except Exception as copy_error:
            log_error(f"Error copying voice audio: {str(copy_error)}")
            # If we can't even copy the file, return the original voice path
            return voice_path

def resize_to_vertical(clip):
    """Resize video clip to vertical format"""
    try:
        # Ensure we're working with RGB video
        if clip.ismask:
            clip = clip.to_RGB()
        
        # Convert dimensions to integers
        target_width = 1080
        target_height = 1920
        clip_w = int(clip.w)
        clip_h = int(clip.h)
        
        # Calculate aspect ratios
        target_ratio = float(target_width) / float(target_height)
        clip_ratio = float(clip_w) / float(clip_h)
        
        if clip_ratio > target_ratio:  # Too wide
            new_height = clip_h
            new_width = int(clip_h * target_ratio)
            x_center = (clip_w - new_width) // 2
            
            # Crop width to match aspect ratio
            clip = clip.crop(
                x1=int(x_center), 
                x2=int(x_center + new_width),
                y1=0,
                y2=clip_h
            )
        else:  # Too tall
            new_width = clip_w
            new_height = int(clip_w / target_ratio)
            y_center = (clip_h - new_height) // 2
            
            # Crop height to match aspect ratio
            clip = clip.crop(
                x1=0,
                x2=clip_w,
                y1=int(y_center),
                y2=int(y_center + new_height)
            )
        
        # Ensure all dimensions are integers
        clip = clip.resize((int(target_width), int(target_height)))
        
        # Force RGB mode
        if hasattr(clip, 'to_RGB'):
            clip = clip.to_RGB()
            
        return clip
    except Exception as e:
        print(colored(f"Error in resize_to_vertical: {str(e)}", "red"))
        return None

def get_emoji_image(emoji_char, size=120):
    """Get colored emoji image using Twitter's emoji CDN"""
    try:
        # Create cache directory if it doesn't exist
        cache_dir = "../temp/emoji_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # More comprehensive emoji cleaning to prevent square characters
        # Remove variation selectors, zero-width joiners, and other special characters
        special_chars = [
            (0xFE00, 0xFE0F),  # Variation Selectors
            (0x200D, 0x200D),  # Zero Width Joiner
            (0x20D0, 0x20FF),  # Combining Diacritical Marks for Symbols
            (0x1F3FB, 0x1F3FF),  # Emoji Modifiers (skin tones)
            (0x200B, 0x200F),  # Zero Width Space, Zero Width Non-Joiner, etc.
            (0x2060, 0x206F)   # Word Joiner, Invisible Times, etc.
        ]
        
        # Clean the emoji by removing all special characters
        cleaned_emoji = ""
        for c in emoji_char:
            should_keep = True
            for start, end in special_chars:
                if start <= ord(c) <= end:
                    should_keep = False
                    break
            if should_keep:
                cleaned_emoji += c
        
        # If cleaning removed everything, use the original
        if not cleaned_emoji:
            cleaned_emoji = emoji_char
        
        # Convert emoji to unicode code points
        emoji_code = "-".join(
            format(ord(c), 'x').lower()
            for c in cleaned_emoji
        )
        
        # Check cache first
        cache_path = f"{cache_dir}/{emoji_code}.png"
        if os.path.exists(cache_path):
            emoji_img = Image.open(cache_path).convert('RGBA')
            emoji_img = emoji_img.resize((size, size), Image.Resampling.LANCZOS)
            return emoji_img

        # Use Twitter's Twemoji CDN which is reliable and open source
        url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{emoji_code}.png"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Successfully got the emoji
                emoji_img = Image.open(BytesIO(response.content)).convert('RGBA')
                
                # Resize emoji
                emoji_img = emoji_img.resize((size, size), Image.Resampling.LANCZOS)
                
                # Cache the image
                emoji_img.save(cache_path, 'PNG')
                
                return emoji_img
            else:
                log_warning(f"Failed to download emoji: {cleaned_emoji} (HTTP {response.status_code})")
                # Try with just the first character if it's a multi-character emoji
                if len(cleaned_emoji) > 1:
                    first_char = cleaned_emoji[0]
                    first_code = format(ord(first_char), 'x').lower()
                    url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{first_code}.png"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        emoji_img = Image.open(BytesIO(response.content)).convert('RGBA')
                        emoji_img = emoji_img.resize((size, size), Image.Resampling.LANCZOS)
                        emoji_img.save(cache_path, 'PNG')
                        return emoji_img
                
                return create_fallback_emoji(size, cleaned_emoji)
        except Exception as e:
            log_warning(f"Error downloading emoji: {str(e)}")
            return create_fallback_emoji(size, cleaned_emoji)
            
    except Exception as e:
        log_warning(f"Error getting emoji image for {emoji_char}: {str(e)}")
        return create_fallback_emoji(size, emoji_char)

def create_fallback_emoji(size=120, emoji_char=None):
    """Create a fallback colored symbol for emoji"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a colorful circle with the emoji character if available
    draw.ellipse([(10, 10), (size-10, size-10)], fill=(255, 200, 0, 255))
    
    # If we have emoji character, try to draw it in the center
    if emoji_char:
        try:
            # Try to find a font that can display emojis
            try:
                font = ImageFont.truetype('seguiemj.ttf', size//2)
            except:
                font = ImageFont.load_default()
                
            # Draw the emoji character in black
            text_bbox = draw.textbbox((0, 0), emoji_char, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (size - text_width) // 2
            y = (size - text_height) // 2
            
            draw.text((x, y), emoji_char, fill=(0, 0, 0, 255), font=font)
        except:
            pass  # Just use the circle if we can't render the text
    
    return img

def create_text_with_emoji(txt, size=(1080, 800)):
    """Create text image with emoji support"""
    try:
        # Create a transparent background
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Load font - REDUCED SIZE for better fit on YouTube Shorts
        try:
            font_path = "assets/fonts/Montserrat-Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "C:/Windows/Fonts/Arial.ttf"  # Fallback for Windows
            font = ImageFont.truetype(font_path, 70)  # Increased from 60 to 70 for better visibility
        except Exception:
            font = ImageFont.load_default()
        
        # Instead of using emoji.get_emoji_regexp() which is causing errors,
        # we'll use a simpler approach and just render the text directly
        
        # Calculate text size for centering
        text_bbox = draw.textbbox((0, 0), txt, font=font)
        text_width = text_bbox[2]
        text_height = text_bbox[3]
        
        # Calculate position to center text
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2 - 50  # Move a bit higher to be more centered in frame
        
        # Draw background for better readability - semi-transparent black background
        padding = 20
        bg_x = x - padding
        bg_y = y - padding
        bg_width = text_width + (padding * 2)
        bg_height = text_height + (padding * 2)
        
        # Draw rounded rectangle background
        radius = 20
        draw.rounded_rectangle(
            [(bg_x, bg_y), (bg_x + bg_width, bg_y + bg_height)],
            radius=radius,
            fill=(0, 0, 0, 180)  # Semi-transparent black
        )
        
        # Draw text in white
        draw.text((x, y), txt, fill=(255, 255, 255, 255), font=font)
        
        return img
    except Exception as e:
        log_error(f"Error creating text: {str(e)}")
        # Create a simple fallback text image
        return create_fallback_text(txt, size)

def create_fallback_text(txt, size=(1080, 800)):
    """Create a simple fallback text image without emojis"""
    try:
        # Create a transparent background
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Use default font
        font = ImageFont.load_default()
        
        # Calculate text size for centering
        text_bbox = draw.textbbox((0, 0), txt, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate position to center text
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        # Draw background
        padding = 10
        draw.rectangle(
            [(x - padding, y - padding), 
             (x + text_width + padding, y + text_height + padding)],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text
        draw.text((x, y), txt, fill=(255, 255, 255, 255), font=font)
        
        return img
    except:
        # Last resort - just create a black image
        return Image.new('RGBA', size, (0, 0, 0, 180))

def create_subtitle_bg(txt, style=None, is_last=False, total_duration=None):
    """Create dynamic centered subtitles with emoji support"""
    try:
        if not txt:
            return None
            
        # Clean text
        clean_txt = txt if isinstance(txt, str) else str(txt)
        clean_txt = clean_txt.strip().strip('"')
        if not clean_txt:
            return None
        
        # Create image with text and emojis - OPTIMIZED FOR SHORTS
        # Use a taller image size to accommodate multiple lines
        text_image = create_text_with_emoji(clean_txt, size=(1080, 900))
        if text_image is None:
            return None
                
        # Convert PIL image to MoviePy clip
        txt_clip = ImageClip(np.array(text_image))
        
        # Set duration based on content length
        if is_last and total_duration is not None:
            duration = max(1.5, total_duration)
        else:
            # Calculate duration based on word count
            word_count = len(clean_txt.split())
            duration = max(1.8, min(4.0, word_count * 0.35))
        
        # Add effects with smoother crossfades
        final_clip = (txt_clip
            .set_duration(duration)
            .set_position(('center', 'center'))  # Center in the middle of the screen
            .crossfadein(0.2))  # Slightly longer fade in
        
        # Add fade out based on position
        if is_last:
            final_clip = final_clip.crossfadeout(0.5)
        else:
            final_clip = final_clip.crossfadeout(0.3)  # Longer fade out for better transitions
        
        return final_clip
        
    except Exception as e:
        log_error(f"Subtitle error: {str(e)}")
        log_error(f"Text content: {txt}")
        return None

def trim_audio_file(audio_path, output_path=None, trim_end=0.15):
    """Trim silence and artifacts from audio file"""
    try:
        print(colored(f"Trimming audio file: {audio_path}", "cyan"))
        
        # Load audio
        audio = AudioFileClip(audio_path)
        
        # Increase trim amount to remove strange sounds at the end
        trim_end = 0.3  # Increase from 0.15 to 0.3 seconds
        
        # Calculate new duration
        new_duration = max(0.1, audio.duration - trim_end)
        
        # Trim the audio
        trimmed_audio = audio.subclip(0, new_duration)
        
        # Add a gentle fade out
        trimmed_audio = trimmed_audio.audio_fadeout(0.2)
        
        # Save to the specified output path or the same path if not provided
        output_path = output_path if output_path else audio_path
        trimmed_audio.write_audiofile(output_path, fps=44100)
        
        # Clean up
        audio.close()
        trimmed_audio.close()
        
        print(colored(f"✓ Trimmed {trim_end}s from end of audio", "green"))
        return output_path
        
    except Exception as e:
        print(colored(f"Error trimming audio: {str(e)}", "red"))
        return audio_path

def enhance_tts_prompt(script, content_type):
    """Create an enhanced prompt for TTS based on content type"""
    
    base_prompt = ""
    
    if content_type == "tech_humor":
        base_prompt = "Speak in an enthusiastic tech presenter voice with good comedic timing. " \
                      "Emphasize the punchlines and add subtle pauses before key jokes. " \
                      "Sound like you're sharing an insider tech joke with the audience."
    
    elif content_type == "life_hack":
        base_prompt = "Speak in a friendly, helpful tone like you're sharing a valuable secret. " \
                      "Emphasize the transformative benefits of the hack with excitement. " \
                      "Use a slightly faster pace for setup and slow down for the key instructions."
    
    elif content_type == "coding_tips":
        base_prompt = "Speak like an experienced programmer sharing wisdom. " \
                      "Use a confident, knowledgeable tone with strategic pauses to emphasize key points. " \
                      "Sound slightly amused when mentioning common coding mistakes."
    
    elif content_type == "food_recipe":
        base_prompt = "Speak in a warm, appetizing tone that makes the food sound delicious. " \
                      "Use a moderately slow pace with emphasis on flavor descriptions. " \
                      "Sound excited about the final result to build anticipation."
    
    else:  # Default for general content
        base_prompt = "Speak in an engaging, naturally enthusiastic voice with good dynamic range. " \
                      "Use appropriate pauses for emphasis and let the funny moments land naturally. " \
                      "End with an upbeat call to action."
    
    # Add script-specific instructions based on emojis present
    if "😂" in script or "🤣" in script:
        base_prompt += " Include light chuckles at the funny parts."
    
    if "💡" in script:
        base_prompt += " Sound genuinely impressed when revealing the key insight."
    
    if "👍" in script or "❤️" in script:
        base_prompt += " End with an enthusiastic tone for the call to action."
    
    return base_prompt

def process_subtitles(subs_path, base_video, start_padding=0.0):
    """Process subtitles and create clips to overlay on video with typing effect that syncs with voiceover"""
    try:
        from video import log_info, log_success, log_warning, log_error
        
        # Check if subtitles file exists
        if not os.path.exists(subs_path):
            log_warning(f"Subtitles file not found: {subs_path}")
            return base_video
            
        # Load subtitles
        try:
            subs = pysrt.open(subs_path)
            log_info(f"Processing {len(subs)} subtitles")
        except Exception as e:
            log_warning(f"Error loading subtitles: {str(e)}")
            return base_video
            
        # Get video dimensions
        video_width, video_height = base_video.size
        
        # Define font sizes based on video dimensions
        large_font_size = 120
        medium_font_size = 100
        small_font_size = 80
        
        # Load fonts
        try:
            # Use a more readable font with better contrast
            font_path = "assets/fonts/Montserrat-Bold.ttf"
            if not os.path.exists(font_path):
                font_path = "assets/fonts/Arial.ttf"
                if not os.path.exists(font_path):
                    # Fallback to default
                    large_font = None
                    medium_font = None
                    small_font = None
                else:
                    large_font = font_path
                    medium_font = font_path
                    small_font = font_path
            else:
                large_font = font_path
                medium_font = font_path
                small_font = font_path
        except Exception as e:
            log_warning(f"Error loading fonts: {str(e)}")
            large_font = None
            medium_font = None
            small_font = None
            
        # Create subtitle clips
        subtitle_clips = []
        
        # Verify video dimensions
        log_info(f"Video dimensions: {video_width}x{video_height}")
        
        # Calculate the vertical position for subtitles (middle of the screen)
        vertical_position = 0.5  # Middle of the screen
        
        # Apply start padding to all subtitles if specified
        if start_padding > 0:
            log_info(f"Applying {start_padding}s start padding to all subtitles")
            for sub in subs:
                sub.start.seconds += start_padding
                sub.end.seconds += start_padding
        
        # Check if the last subtitle ends before the video ends
        if subs and len(subs) > 0:
            last_sub = subs[-1]
            last_end_time = last_sub.end.ordinal / 1000.0  # Convert to seconds
            video_duration = base_video.duration
            
            # Make the last subtitle stay until the end of the video
            if last_end_time < video_duration - 0.5:
                log_info(f"Extending last subtitle to match video duration")
                last_sub.end.seconds = video_duration - 0.5  # End 0.5 second before video end
        
        # Process each subtitle
        for i, sub in enumerate(subs):
            try:
                # Skip empty subtitles
                if not sub.text or not sub.text.strip():
                    continue
                    
                # Calculate timing with precise millisecond accuracy
                start_time = sub.start.ordinal / 1000.0  # Convert to seconds
                end_time = sub.end.ordinal / 1000.0
                duration = end_time - start_time
                
                # Add a small buffer to ensure subtitles don't cut off too quickly
                duration += 0.2  # Add 200ms buffer
                end_time += 0.2
                
                # Skip if duration is invalid or too short
                if duration <= 0.1:  # Minimum duration of 0.1 seconds
                    log_warning(f"Skipping subtitle {i+1} with invalid duration: {duration}s")
                    continue
                    
                # Clean the text but preserve emojis
                text = html.unescape(sub.text)
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                text = re.sub(r"\.'\]$", "", text)   # Remove trailing .']
                text = text.strip()
                
                if not text:
                    continue
                
                # Choose font size based on text length
                if len(text) < 20:
                    font = large_font
                    font_size = large_font_size
                elif len(text) < 40:
                    font = medium_font
                    font_size = medium_font_size
                else:
                    font = small_font
                    font_size = small_font_size
                
                # Create text clip with typing effect
                try:
                    # Calculate text width to determine if we need to split into multiple lines
                    max_width = video_width * 0.85  # Use 85% of video width
                    
                    # Split text into multiple lines if it's too long
                    words = text.split()
                    lines = []
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        # Estimate text width (this is approximate)
                        estimated_width = len(test_line) * (font_size * 0.6)  # Rough estimate
                        
                        if estimated_width <= max_width:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Join lines with newlines
                    multiline_text = "\n".join(lines)
                    
                    # Create a list of clips for the typing effect
                    typing_clips = []
                    
                    # Calculate typing speed based on duration and text length
                    total_chars = len(multiline_text)
                    
                    # Improved typing speed calculation:
                    # - For short durations, ensure all text appears with enough time to read
                    # - For longer durations, create a more natural typing effect
                    if duration < 2.0:
                        # For very short subtitles, type faster to ensure all text appears
                        chars_per_second = total_chars / (duration * 0.7)  # Use 70% of duration for typing
                    elif duration < 4.0:
                        # For medium length subtitles, use a balanced approach
                        chars_per_second = total_chars / (duration * 0.8)  # Use 80% of duration for typing
                    else:
                        # For longer subtitles, use a more natural typing speed
                        chars_per_second = total_chars / (duration * 0.9)  # Use 90% of duration for typing
                    
                    # Ensure minimum typing speed for readability
                    min_chars_per_second = 15  # Minimum 15 characters per second
                    chars_per_second = max(chars_per_second, min_chars_per_second)
                    
                    # Log the typing speed for debugging
                    if i == 0 or i == len(subs) - 1:
                        log_info(f"Subtitle {i+1} typing speed: {chars_per_second:.1f} chars/sec (duration: {duration:.2f}s, chars: {total_chars})")
                    
                    # Create a clip for each character to simulate typing
                    for j in range(1, total_chars + 1):
                        partial_text = multiline_text[:j]
                        
                        # Create text clip for this partial text
                        txt_clip = TextClip(
                            partial_text,
                            fontsize=font_size,
                            font=font,
                            color='white',
                            align='center',
                            method='caption',
                            size=(int(video_width * 0.9), None)
                        )
                        
                        # Calculate when this character should appear
                        char_time = start_time + (j / chars_per_second)
                        
                        # Set duration for this character's clip (until the next character appears)
                        if j < total_chars:
                            next_char_time = start_time + ((j + 1) / chars_per_second)
                            char_duration = next_char_time - char_time
                        else:
                            # Last character stays until the end of the subtitle
                            char_duration = end_time - char_time
                            
                            # For the last subtitle, make the last character stay until the end of the video
                            if i == len(subs) - 1:
                                char_duration = base_video.duration - char_time
                        
                        # Set timing and position
                        txt_clip = txt_clip.set_start(char_time).set_duration(char_duration)
                        txt_clip = txt_clip.set_position(('center', vertical_position), relative=True)
                        
                        typing_clips.append(txt_clip)
                    
                    # Add all typing clips to the subtitle clips list
                    subtitle_clips.extend(typing_clips)
                    
                    # Log progress
                    if i == 0 or i == len(subs) - 1 or i % max(5, len(subs) // 5) == 0:
                        log_info(f"Created subtitle {i+1}/{len(subs)} with typing effect (Time: {start_time:.2f}s - {end_time:.2f}s)")
                
                except Exception as img_error:
                    log_warning(f"Error creating subtitle image for subtitle {i+1}: {str(img_error)}")
                    traceback.print_exc()
                    
                    # Fallback to simple text without typing effect
                    try:
                        txt_clip = TextClip(
                            multiline_text,
                            fontsize=font_size,
                            font=font,
                            color='white',
                            align='center',
                            method='caption',
                            size=(int(video_width * 0.9), None)
                        )
                        
                        # Set timing and position
                        txt_clip = txt_clip.set_start(start_time).set_duration(duration)
                        txt_clip = txt_clip.set_position(('center', vertical_position), relative=True)
                        
                        # For the last subtitle, make it stay until the end of the video
                        if i == len(subs) - 1:
                            txt_clip = txt_clip.set_duration(base_video.duration - start_time)
                        
                        # Add fade effects
                        fade_duration = min(0.2, duration / 5)
                        txt_clip = txt_clip.crossfadein(fade_duration).crossfadeout(fade_duration)
                        
                        subtitle_clips.append(txt_clip)
                        log_warning(f"Using fallback simple text for subtitle {i+1}")
                    except Exception as fallback_error:
                        log_warning(f"Fallback also failed for subtitle {i+1}: {str(fallback_error)}")
                        continue
                    
            except Exception as e:
                log_warning(f"Error processing subtitle {i+1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Check if we created any subtitle clips
        if not subtitle_clips:
            log_warning("No valid subtitle clips were created")
            return base_video
            
        # Add all subtitle clips to the video
        log_info(f"Adding {len(subtitle_clips)} subtitle clips with typing effect to video")
        
        # Create final video with subtitles
        final_video = CompositeVideoClip([base_video] + subtitle_clips)
        
        return final_video
        
    except Exception as e:
        log_error(f"Error processing subtitles: {str(e)}")
        traceback.print_exc()
        return base_video

def generate_video(background_path, audio_path, subtitles_path=None, content_type=None, target_duration=None, 
                  use_background_music=True, music_volume=0.3, music_fade_in=2.0, music_fade_out=3.0, script_path=None):
    """Generate a video with audio and subtitles with 1s start padding and 2s end padding"""
    try:
        from video import log_info, log_success, log_warning, log_error, log_step
        
        # Handle different background path formats
        if isinstance(background_path, list):
            # If we have a list of background videos, use combine_videos function
            if len(background_path) > 1:
                log_info(f"Combining {len(background_path)} videos for a dynamic background")
                try:
                    # Load audio to get duration
                    audio = AudioFileClip(audio_path)
                    audio_duration = audio.duration
                    audio.close()
                    
                    # Define padding
                    start_padding = 1.0  # 1 second at the start
                    end_padding = 2.0    # 2 seconds at the end
                    
                    # Calculate total video duration with padding
                    total_duration = audio_duration + start_padding + end_padding
                    
                    # Use the combine_videos function to create a dynamic background
                    combined_video_path = combine_videos(background_path, audio_duration, total_duration)
                    if combined_video_path and os.path.exists(combined_video_path):
                        background_path = combined_video_path
                        log_success("Successfully combined multiple videos for a dynamic background")
                    else:
                        # Fallback to using the first video if combination fails
                        log_warning("Video combination failed, using first video as fallback")
                        background_path = background_path[0]
                        if isinstance(background_path, dict):
                            background_path = background_path.get('path')
                except Exception as e:
                    log_error(f"Error in video combination: {str(e)}")
                    # Fallback to using the first video
                    log_warning("Using first video as fallback due to combination error")
                    background_path = background_path[0]
                    if isinstance(background_path, dict):
                        background_path = background_path.get('path')
            else:
                # Just one video in the list
                log_info(f"Using single video from list")
                background_path = background_path[0]
                if isinstance(background_path, dict):
                    background_path = background_path.get('path')
        
        # Check if files exist
        if not os.path.exists(background_path):
            log_error(f"Background video not found: {background_path}")
            return None
            
        if not os.path.exists(audio_path):
            log_error(f"Audio file not found: {audio_path}")
            return None
            
        # Load background video
        background_video = VideoFileClip(background_path)
        
        # Load audio
        audio = AudioFileClip(audio_path)
        
        # Get audio duration
        audio_duration = audio.duration
        
        # Define padding
        start_padding = 1.0  # 1 second at the start
        end_padding = 2.0    # 2 seconds at the end
        
        # Calculate total video duration with padding
        total_duration = audio_duration + start_padding + end_padding
        log_info(f"Audio duration: {audio_duration:.2f}s")
        log_info(f"Video duration: {total_duration:.2f}s (with {start_padding}s start and {end_padding}s end padding)")
        
        # If target duration is provided, use it
        if target_duration:
            log_info(f"Using provided target duration: {target_duration}s")
            # Ensure target duration is not shorter than audio duration + padding
            if target_duration < total_duration:
                log_warning(f"Target duration {target_duration}s is shorter than audio duration + padding {total_duration}s")
                log_info(f"Adjusting target duration to {total_duration}s")
                target_duration = total_duration
        else:
            target_duration = total_duration
            
        # Adjust duration slightly to prevent black end frame
        target_duration -= 0.05
        log_info(f"Adjusted to {target_duration:.2f}s to prevent black end frame")
        
        # Resize and trim background video
        try:
            # Resize to vertical format
            log_processing("Resizing video to vertical format")
            background_video = resize_to_vertical(background_video)
            
            # Trim or loop background video to match target duration
            log_processing(f"Trimming background video from {background_video.duration:.2f}s to {target_duration:.2f}s")
            
            if background_video.duration < target_duration:
                # Loop the video if it's too short
                n_loops = math.ceil(target_duration / background_video.duration)
                clips = [background_video] * n_loops
                background_video = concatenate_videoclips(clips)
                
            # Trim to exact duration
            background_video = background_video.subclip(0, target_duration)
            
            # Add audio to video with delay
            log_processing("Adding audio to video with delay")
            
            # If background music is enabled, mix it with the audio
            if use_background_music and content_type:
                try:
                    # Get background music
                    music_path = get_background_music(content_type, audio_duration)
                    
                    if music_path and os.path.exists(music_path):
                        # Get file size in MB
                        file_size = os.path.getsize(music_path) / (1024 * 1024)  # MB
                        file_name = os.path.basename(music_path)
                        
                        log_info(f"Adding background music: {file_name}")
                        log_info(f"Music file size: {file_size:.2f} MB")
                        log_info(f"Music volume: {music_volume} (fade in: {music_fade_in}s, fade out: {music_fade_out}s)")
                        
                        # Create a temporary file for the mixed audio
                        mixed_audio_path = f"temp/mixed_audio_{uuid.uuid4()}.mp3"
                        
                        # Set the audio to start after the start_padding
                        delayed_audio_path = f"temp/delayed_audio_{uuid.uuid4()}.mp3"
                        delayed_audio = audio.set_start(start_padding)
                        delayed_audio.write_audiofile(delayed_audio_path)
                        
                        # Check if we already have a mixed version (to avoid duplicate mixing)
                        mixed_flag_file = delayed_audio_path + ".mixed"
                        if os.path.exists(mixed_flag_file):
                            log_info("Using pre-mixed audio (already processed)")
                            # Just use the existing mixed audio
                            if os.path.exists(mixed_audio_path):
                                try:
                                    final_audio = AudioFileClip(mixed_audio_path)
                                    video = background_video.set_audio(final_audio)
                                except Exception as e:
                                    log_warning(f"Error loading pre-mixed audio: {str(e)}")
                                    # Fallback to original audio
                                    delayed_audio = audio.set_start(start_padding)
                                    video = background_video.set_audio(delayed_audio)
                            else:
                                # If mixed file doesn't exist, use original
                                delayed_audio = audio.set_start(start_padding)
                                video = background_video.set_audio(delayed_audio)
                        else:
                            # Mix the audio using our enhanced mix_audio function
                            mixed_audio_path = mix_audio(
                                voice_path=delayed_audio_path,
                                music_path=music_path,
                                output_path=mixed_audio_path,
                                music_volume=music_volume,
                                fade_in=music_fade_in,
                                fade_out=music_fade_out
                            )
                            
                            if mixed_audio_path and os.path.exists(mixed_audio_path):
                                try:
                                    # Load the mixed audio
                                    final_audio = AudioFileClip(mixed_audio_path)
                                    
                                    # Set audio to video
                                    video = background_video.set_audio(final_audio)
                                    log_success("Added background music to video using enhanced mixing")
                                    log_info(f"Music: {os.path.basename(music_path)} | Volume: {music_volume}")
                                except Exception as audio_error:
                                    log_warning(f"Error setting mixed audio to video: {str(audio_error)}")
                                    # Fallback to direct audio setting without using CompositeAudioClip
                                    log_info("Using fallback method for audio mixing")
                                    delayed_audio = audio.set_start(start_padding)
                                    video = background_video.set_audio(delayed_audio)
                            else:
                                # No music found, just use voice audio
                                log_warning(f"Failed to mix audio, using voice audio only")
                                delayed_audio = audio.set_start(start_padding)
                                video = background_video.set_audio(delayed_audio)
                                
                            # Clean up temporary files
                            try:
                                if os.path.exists(delayed_audio_path):
                                    os.remove(delayed_audio_path)
                                if mixed_audio_path and os.path.exists(mixed_audio_path):
                                    os.remove(mixed_audio_path)
                            except Exception as e:
                                log_warning(f"Error cleaning up temporary audio files: {str(e)}")
                    else:
                        # No music found, just use voice audio
                        log_warning(f"No background music found for {content_type}, using voice audio only")
                        delayed_audio = audio.set_start(start_padding)
                        video = background_video.set_audio(delayed_audio)
                except Exception as e:
                    log_warning(f"Error adding background music: {str(e)}")
                    # No music found, just use voice audio
                    delayed_audio = audio.set_start(start_padding)
                    video = background_video.set_audio(delayed_audio)
            else:
                # Background music disabled, just use voice audio
                log_info("Background music disabled, using voice audio only")
                delayed_audio = audio.set_start(start_padding)
                video = background_video.set_audio(delayed_audio)
            
            # Process subtitles if provided
            if subtitles_path and os.path.exists(subtitles_path):
                log_info(f"Adding subtitles to video")
                # Process subtitles with padding offset
                video = process_subtitles(subtitles_path, video, start_padding)
            
            return video
            
        except Exception as e:
            log_error(f"Error processing video: {str(e)}")
            log_error(traceback.format_exc())
            return None
            
    except Exception as e:
        log_error(f"Error generating video: {str(e)}")
        log_error(traceback.format_exc())
        return None

def generate_tts_audio(script: str, voice: str = "en_us_001", content_type: str = None) -> str:
    """Generate TTS audio from script"""
    try:
        # Check if using DeepSeek voice
        if voice.startswith("deepseek_"):
            try:
                from deepseek_integration import DeepSeekAPI
                import asyncio
                
                deepseek = DeepSeekAPI()
                voice_id = voice.replace("deepseek_", "")  # Extract DeepSeek voice ID
                audio_path = asyncio.run(deepseek.generate_voice(script, voice_id))
                
                if audio_path:
                    print(colored("✓ Generated TTS audio using DeepSeek", "green"))
                    return audio_path
                else:
                    print(colored("[-] DeepSeek TTS failed, falling back to Coqui TTS", "yellow"))
            except Exception as e:
                print(colored(f"[-] Error with DeepSeek TTS: {str(e)}, falling back to Coqui TTS", "yellow"))

        # Try Coqui TTS first
        try:
            from coqui_integration import CoquiTTSAPI
            import asyncio
            
            coqui = CoquiTTSAPI()
            
            # Map TikTok voices to Coqui speakers/languages
            voice_mapping = {
                "en_us_001": ("xtts_v2", "en"),  # Default English
                "en_us_002": ("jenny", "en"),     # Jenny voice
                "en_male_funny": ("vits", "en"),  # VITS male voice
                "en_male_narration": ("xtts_v2", "en"),  # XTTS male voice
                "en_female_emotional": ("xtts_v2", "en")  # XTTS female voice
            }
            
            # Get model and language from mapping or use defaults
            model, language = voice_mapping.get(voice, ("xtts_v2", "en"))
            
            # Determine emotion from content type
            emotion_mapping = {
                "tech_humor": "cheerful",
                "life_hack": "friendly",
                "coding_tips": "professional",
                "food_recipe": "warm",
                "fitness_motivation": "energetic"
            }
            emotion = emotion_mapping.get(content_type, "neutral")
            
            # Generate audio with Coqui TTS
            audio_path = asyncio.run(coqui.generate_voice(
                text=script,
                speaker=None,  # Let the model choose best speaker
                language=language,
                emotion=emotion
            ))
            
            if audio_path:
                print(colored("✓ Generated TTS audio using Coqui", "green"))
                return audio_path
            else:
                print(colored("[-] Coqui TTS failed, falling back to TikTok TTS", "yellow"))
                
        except Exception as e:
            print(colored(f"[-] Error with Coqui TTS: {str(e)}, falling back to TikTok TTS", "yellow"))

        # Create temp directory if it doesn't exist
        os.makedirs("temp/tts", exist_ok=True)
        output_path = f"temp/tts/{content_type}_latest.mp3"

        # Fallback to TikTok TTS
        success = tts(script, voice, output_path)
        
        if success:
            print(colored("✓ Generated TTS audio using TikTok", "green"))
            return output_path
        else:
            print(colored("[-] Failed to generate TTS audio", "red"))
            return None

    except Exception as e:
        print(colored(f"[-] Error generating TTS audio: {str(e)}", "red"))
        return None

def read_subtitles_file(filename):
    """Read subtitles file with proper UTF-8 encoding"""
    times_texts = []
    current_times = None
    current_text = ""
    
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.readlines()
            
        for line in content:
            line = line.strip()
            if not line:
                if current_times and current_text:
                    times_texts.append((current_times, current_text.strip()))
                current_times, current_text = None, ""
            elif '-->' in line:
                # Parse timecodes
                start, end = line.split('-->')
                start = start.strip().replace(',', '.')
                end = end.strip().replace(',', '.')
                current_times = [float(sum(float(x) * 60 ** i 
                               for i, x in enumerate(reversed(t.split(':'))))) 
                               for t in [start, end]]
            elif line.isdigit():
                # Skip subtitle numbers
                continue
            elif current_times is not None:
                current_text += line + "\n"
        
        return times_texts
        
    except Exception as e:
        print(colored(f"Error reading subtitles file: {str(e)}", "red"))
        return None

def test_single_subtitle():
    """Test function for single subtitle with emojis"""
    try:
        log_section("Subtitle Test Mode", "🧪")
        
        # Create a test video clip (dark gray background)
        log_processing("Creating test background...")
        test_video = ColorClip(size=(1080, 1920), color=(40, 40, 40)).set_duration(5)
        
        # Create test subtitles file with just one subtitle
        log_info("Creating test subtitle with emoji...")
        test_srt = """1
00:00:00,000 --> 00:00:05,000
This is a centered subtitle with emoji! 🚀 It should be visible on any background. 🎬"""

        # Save test subtitles
        test_srt_path = "../temp/test_single_sub.srt"
        with open(test_srt_path, "w", encoding="utf-8-sig") as f:
            f.write(test_srt)
        log_success(f"Saved test subtitle to {test_srt_path}")

        # Process subtitles
        log_processing("Processing test subtitle...")
        final_video = process_subtitles(test_srt_path, test_video)
        
        # Save test video
        output_path = "../temp/centered_subtitle_test.mp4"
        log_processing(f"Rendering test video to {output_path}...")
        start_time = time.time()
        
        final_video.write_videofile(output_path, fps=30)
        
        elapsed_time = time.time() - start_time
        log_success(f"Test video rendered in {elapsed_time:.1f} seconds", "⏱️")
        log_success(f"Test complete! Video saved to: {output_path}", "🎉")
        
        # Print instructions for viewing
        log_info("Open the test video to see the centered subtitle with text outline", "👁️")
        log_info("The subtitle should be clearly visible in the center of the screen", "📱")
        
    except Exception as e:
        log_error(f"Error in subtitle test: {str(e)}")
        log_error(traceback.format_exc())

def generate_video_thumbnail(script, content_type):
    """Generate a thumbnail for the video using script content"""
    try:
        # Create thumbnail directory
        os.makedirs("temp/thumbnails", exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"temp/thumbnails/{content_type}_{timestamp}.jpg"
        
        # Initialize thumbnail generator
        from thumbnail_generator import ThumbnailGenerator
        generator = ThumbnailGenerator()
        
        # Clean script text to remove problematic characters
        clean_script = re.sub(r'[\\/:*?"<>|]', '', script)  # Remove characters not allowed in filenames
        
        # Extract main text from script (first line usually)
        lines = clean_script.strip().split('\n')
        main_text = lines[0].replace('"', '').strip()
        if len(main_text) > 30:
            main_text = main_text[:27] + "..."
            
        # Get subtitle (second line if available)
        subtitle = ""
        if len(lines) > 1:
            subtitle = lines[1].replace('"', '').strip()
            if len(subtitle) > 20:
                subtitle = subtitle[:17] + "..."
        
        # Extract emojis for thumbnail
        emojis = []
        for c in clean_script:
            try:
                if emoji.is_emoji(c):
                    emojis.append(c)
            except:
                pass
        
        primary_emoji = emojis[0] if emojis else ""
        
        # Configure thumbnail based on content type
        templates = {
            'tech_humor': {
                'bg_color': '#1E1E1E',
                'gradient': ['#FF4D4D', '#1E1E1E'],
                'text': main_text,
                'subtitle': subtitle or 'Tech Humor',
                'icon': primary_emoji or '🔥',
                'tech_element': '💻',
                'font_size': 120,
                'subtitle_size': 72,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay', 'blur']
            },
            'coding_tips': {
                'bg_color': '#0D2538',
                'gradient': ['#1A4B6D', '#0D2538'],
                'text': main_text,
                'subtitle': subtitle or 'Coding Tips',
                'icon': primary_emoji or '💡',
                'tech_element': '⌨️',
                'font_size': 110,
                'subtitle_size': 70,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay', 'blur']
            },
            'life_hack': {
                'bg_color': '#2B580C',
                'gradient': ['#639A67', '#2B580C'],
                'text': main_text,
                'subtitle': subtitle or 'Life Hack',
                'icon': primary_emoji or '✨',
                'tech_element': '🔧',
                'font_size': 110,
                'subtitle_size': 68,
                'text_color': '#FFFFFF',
                'shadow_color': '#000000',
                'effects': ['gradient', 'overlay']
            },
        }
        
        # Get template based on content type or use default
        template = templates.get(content_type, templates['tech_humor'])
        
        # Create thumbnail image
        img = Image.new('RGB', (1080, 1920), template['bg_color'])
        
        # Apply gradient
        if 'gradient' in template['effects']:
            gradient = generator.create_gradient(template['gradient'], size=(1080, 1920))
            img = Image.blend(img, gradient, 0.8)
            
        # Apply overlay
        if 'overlay' in template['effects']:
            overlay = Image.new('RGB', (1080, 1920), template['bg_color'])
            img = Image.blend(img, overlay, 0.4)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype(generator.montserrat_bold, template['font_size'])
            subtitle_font = ImageFont.truetype(generator.roboto_regular, template['subtitle_size'])
        except Exception as e:
            log_warning(f"Error loading fonts: {str(e)}")
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            
        draw = ImageDraw.Draw(img)
        
        # Instead of directly using the template text, save it to a temporary file first
        temp_text_file = f"temp/thumbnails/text_{timestamp}.txt"
        with open(temp_text_file, "w", encoding="utf-8") as f:
            f.write(template['text'])
        
        # Read it back to ensure proper encoding
        with open(temp_text_file, "r", encoding="utf-8") as f:
            safe_text = f.read().strip()
        
        # Add text with enhanced shadow
        generator.draw_enhanced_text(
            draw, 
            safe_text, 
            title_font, 
            template['text_color'],
            template['shadow_color'],
            offset_y=-200
        )
        
        # Do the same for subtitle
        temp_subtitle_file = f"temp/thumbnails/subtitle_{timestamp}.txt"
        with open(temp_subtitle_file, "w", encoding="utf-8") as f:
            f.write(template['subtitle'])
        
        # Read it back to ensure proper encoding
        with open(temp_subtitle_file, "r", encoding="utf-8") as f:
            safe_subtitle = f.read().strip()
        
        generator.draw_enhanced_text(
            draw, 
            safe_subtitle, 
            subtitle_font,
            template['text_color'],
            template['shadow_color'],
            offset_y=-50
        )
        
        # Add icon if available
        if 'icon' in template and template['icon']:
            emoji_img = get_emoji_image(template['icon'], size=300)
            if emoji_img:
                img.paste(emoji_img, (540, 1200), emoji_img)
        
        # Save with high quality
        img.save(output_path, 'JPEG', quality=95)
        log_success(f"Generated thumbnail: {output_path}")
        
        # Clean up temporary files
        try:
            os.remove(temp_text_file)
            os.remove(temp_subtitle_file)
        except:
            pass
            
        return output_path
        
    except Exception as e:
        log_error(f"Error generating thumbnail: {str(e)}")
        traceback.print_exc()
        
        # Try a simpler fallback approach
        try:
            log_warning("Attempting fallback thumbnail generation")
            
            # Create a simple solid color background
            img = Image.new('RGB', (1080, 1920), "#1E1E1E")
            draw = ImageDraw.Draw(img)
            
            # Use a simple default font
            font = ImageFont.load_default()
            
            # Draw a simple text
            channel_text = content_type.replace('_', ' ').title()
            draw.text((540, 960), channel_text, fill="white", anchor="mm", font=font)
            
            # Save the fallback thumbnail
            fallback_path = f"temp/thumbnails/{content_type}_fallback_{timestamp}.jpg"
            img.save(fallback_path, 'JPEG', quality=95)
            log_warning(f"Generated fallback thumbnail: {fallback_path}")
            
            return fallback_path
        except Exception as fallback_error:
            log_error(f"Fallback thumbnail generation also failed: {str(fallback_error)}")
            return None

if __name__ == "__main__":
    # Display a colorful banner when running directly
    print("\n")
    cprint("╔═════════════════════════════════════════════╗", "cyan", attrs=["bold"])
    cprint("║                                             ║", "cyan", attrs=["bold"])
    cprint("║  🎬  YOUTUBE SHORTS VIDEO GENERATOR  🎬  ║", "yellow", attrs=["bold"])
    cprint("║                                             ║", "cyan", attrs=["bold"])
    cprint("╚═════════════════════════════════════════════╝", "cyan", attrs=["bold"])
    print("\n")
    
    # Run the test function
    test_single_subtitle()