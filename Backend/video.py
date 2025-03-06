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

from tiktokvoice import tts
from music_provider import MusicProvider
import concurrent.futures
from pydub import AudioSegment

# First activate your virtual environment and install pysrt:
# python -m pip install pysrt --no-cache-dir

# ===== LOGGING FUNCTIONS =====
def log_section(title, emoji="✨"):
    """Print a section header with emoji"""
    print("\n")
    cprint(f"=== {emoji} {title} {emoji} ===", "blue", attrs=["bold"])

def log_error(message, emoji="❌"):
    """Log an error message in red with emoji"""
    cprint(f"{emoji} Error: {message}", "red")

def log_warning(message, emoji="⚠️"):
    """Log a warning message in yellow with emoji"""
    cprint(f"{emoji} Warning: {message}", "yellow")

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

# ===== END LOGGING FUNCTIONS =====

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
        # First, check if we have a cleaned script in the JSON file
        json_path = f"cache/scripts/{content_type}_latest.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    script_data = json.load(f)
                    if "cleaned_script" in script_data and script_data["cleaned_script"].strip():
                        script = script_data["cleaned_script"]
                        print(colored(f"Using pre-cleaned script from JSON file for subtitles", "green"))
            except Exception as e:
                print(colored(f"Error reading JSON file: {str(e)}", "yellow"))
                # Continue with the provided script
        
        # Get audio duration
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        audio.close()
        
        # Create temp directory
        os.makedirs("temp/subtitles", exist_ok=True)
        subtitles_path = "temp/subtitles/generated_subtitles.srt"
        
        # Print the original script
        print(colored(f"Original script for subtitles:\n{script}", "blue"))
        
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
        
        # Define common emojis to add for different content types if not already present
        emoji_sets = {
            'tech_humor': ['💻', '🤖', '😂', '🚀', '🔥', '⚡', '🧠', '🤯', '👨‍💻', '👩‍💻', '🛠️', '🔧'],
            'ai_money': ['💰', '💸', '🤖', '📈', '💼', '🚀', '💡', '🔑', '💵', '🏆', '📊', '💹'],
            'baby_tips': ['👶', '🍼', '🧸', '👨‍👩‍👧‍👦', '💤', '🌙', '❤️', '🤱', '👨‍👩‍👦', '🧠', '🍎', '🏠'],
            'quick_meals': ['🍳', '🍲', '⏱️', '🥗', '🍽️', '🔪', '🥘', '🍴', '🥑', '🍅', '🧄', '🧅'],
            'fitness_motivation': ['💪', '🏋️', '🏃', '🧘', '🥗', '❤️', '🔥', '⚡', '🏆', '🌟', '⏱️', '🧠']
        }
        
        # Check if script already has emojis
        has_emojis = any(emoji.is_emoji(c) for c in script)
        
        # Get emoji set for this content type or use tech_humor as default
        emojis = emoji_sets.get(content_type, emoji_sets['tech_humor'])
        
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
                # Check if line already has emojis
                line_has_emoji = any(emoji.is_emoji(c) for c in line)
                
                # Only add emojis if the script doesn't already have them
                if not has_emojis and not line_has_emoji:
                    # Add emoji at the end of the line (30% chance)
                    if random.random() < 0.3:
                        line = f"{line} {random.choice(emojis)}"
                
                raw_sentences.append(line)
        
        # Check if we have any sentences after processing
        if not raw_sentences:
            print(colored("Warning: No sentences found after processing. Using original script with headers removed.", "yellow"))
            # Use the original script but remove the headers
            for line in lines:
                if line.strip():
                    # Skip lines that only contain special characters
                    if line.strip() in special_patterns or line.strip("-*#") == "":
                        continue
                        
                    # Remove numeric prefixes
                    clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
                    # Remove all section headers
                    for header in section_headers:
                        if header in clean_line:
                            clean_line = clean_line.replace(header, '')
                    if clean_line.strip():
                        raw_sentences.append(clean_line.strip())
        
        # Split long sentences into smaller chunks for better readability
        sentences = []
        max_chars_per_subtitle = 80  # Maximum characters per subtitle line
        
        for sentence in raw_sentences:
            # If sentence is short enough, add it as is
            if len(sentence) <= max_chars_per_subtitle:
                sentences.append(sentence)
            else:
                # Split by natural break points (., !, ?, :, ;)
                parts = re.split(r'([.!?:;])', sentence)
                
                # Recombine the parts with their punctuation
                chunks = []
                current_chunk = ""
                
                for i in range(0, len(parts), 2):
                    part = parts[i]
                    punctuation = parts[i+1] if i+1 < len(parts) else ""
                    
                    if len(current_chunk) + len(part) + len(punctuation) <= max_chars_per_subtitle:
                        current_chunk += part + punctuation
                    else:
                        # If the current chunk is not empty, add it to chunks
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # Start a new chunk
                        current_chunk = part + punctuation
                
                # Add the last chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If we couldn't split by punctuation (or the parts are still too long),
                # split by phrases (commas) or just by length
                if not chunks or any(len(chunk) > max_chars_per_subtitle for chunk in chunks):
                    chunks = []
                    parts = re.split(r'([,])', sentence)
                    
                    current_chunk = ""
                    for i in range(0, len(parts), 2):
                        part = parts[i]
                        comma = parts[i+1] if i+1 < len(parts) else ""
                        
                        if len(current_chunk) + len(part) + len(comma) <= max_chars_per_subtitle:
                            current_chunk += part + comma
                        else:
                            # If the current chunk is not empty, add it to chunks
                            if current_chunk:
                                chunks.append(current_chunk)
                            
                            # Start a new chunk
                            current_chunk = part + comma
                    
                    # Add the last chunk if it's not empty
                    if current_chunk:
                        chunks.append(current_chunk)
                
                # If we still have chunks that are too long, split them by length
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) <= max_chars_per_subtitle:
                        final_chunks.append(chunk)
                    else:
                        # Split by words, trying to keep chunks under max_chars_per_subtitle
                        words = chunk.split()
                        current_chunk = ""
                        
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= max_chars_per_subtitle:
                                current_chunk += (" " if current_chunk else "") + word
                            else:
                                final_chunks.append(current_chunk)
                                current_chunk = word
                        
                        if current_chunk:
                            final_chunks.append(current_chunk)
                
                # Add all chunks to sentences
                sentences.extend(final_chunks)
        
        # Print the processed sentences
        print(colored(f"Processed {len(sentences)} sentences for subtitles", "blue"))
        for i, sentence in enumerate(sentences):
            print(colored(f"  {i+1}: {sentence}", "blue"))
        
        # Calculate total word count for better timing distribution
        total_words = sum(len([w for w in sentence.split() if not all(emoji.is_emoji(c) for c in w)]) 
                          for sentence in sentences)
        
        # Analyze audio duration to better sync subtitles
        # Reserve 95% of the audio duration for subtitles, leaving 5% for natural pauses
        usable_duration = total_duration * 0.95
        
        # Calculate average speaking rate (words per second)
        if total_words > 0:
            words_per_second = total_words / usable_duration
            print(colored(f"Speaking rate: {words_per_second:.2f} words per second", "blue"))
        else:
            words_per_second = 2.5  # Default speaking rate if no words
        
        # Write SRT file with improved timing
        with open(subtitles_path, "w", encoding="utf-8-sig") as f:
            current_time = 0.2  # Start with a small initial delay (0.2s)
            
            for i, sentence in enumerate(sentences):
                # Calculate word count for timing (excluding emojis)
                word_count = len([w for w in sentence.split() if not all(emoji.is_emoji(c) for c in w)])
                
                # Calculate duration based on word count and speaking rate
                # Add a small buffer for natural pauses between sentences
                if word_count > 0:
                    est_duration = (word_count / words_per_second) + 0.3
                    # Ensure minimum and maximum durations
                    est_duration = max(1.0, min(5.0, est_duration))
                else:
                    # If the sentence is just emojis, use a short duration
                    est_duration = 1.0
                
                # Ensure we don't exceed audio duration
                if current_time + est_duration > total_duration:
                    est_duration = total_duration - current_time
                
                if est_duration <= 0:
                    break
                
                # Write SRT entry
                start_time = format_time(current_time)
                end_time = format_time(current_time + est_duration)
                
                f.write(f"{i+1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{sentence}\n\n")
                
                # Update current time for next subtitle
                # Overlap slightly for smoother transitions
                current_time += est_duration - 0.1
                if current_time < 0:
                    current_time = 0
            
            # Ensure the last subtitle extends to the end of the audio
            if sentences and current_time < total_duration:
                # We need to reopen the file in read mode first, then write mode
                f.close()
                
                # Read the file content
                with open(subtitles_path, "r", encoding="utf-8-sig") as read_f:
                    subs = read_f.read()
                
                lines = subs.split('\n')
                
                # Find the last timestamp line
                for i in range(len(lines) - 1, -1, -1):
                    if ' --> ' in lines[i]:
                        # Update the end time
                        parts = lines[i].split(' --> ')
                        lines[i] = f"{parts[0]} --> {format_time(total_duration)}"
                        break
                
                # Write the updated content back to the file
                with open(subtitles_path, "w", encoding="utf-8-sig") as write_f:
                    write_f.write('\n'.join(lines))
                
                print(colored("Extended final subtitle until end of video", "blue"))
        
        print(colored(f"✅ Generated {len(sentences)} subtitles with improved timing", "green"))
        return subtitles_path
        
    except Exception as e:
        print(colored(f"Error generating subtitles: {str(e)}", "red"))
        traceback.print_exc()
        return None

def combine_videos(video_paths, audio_duration, target_duration, n_threads=4):
    """Combine multiple videos with smooth transitions for YouTube Shorts"""
    try:
        log_section("Video Combination", "🔄")
        
        # Calculate total target duration
        total_duration = audio_duration if audio_duration else target_duration
        
        # IMPORTANT: Slightly reduce the target duration to prevent black fade at end
        adjusted_duration = total_duration - 0.1  # Reduce by 0.1 seconds
        
        log_info(f"Audio duration: {total_duration:.2f}s")
        log_info(f"Target duration: {target_duration:.2f}s")
        log_info(f"Adjusted duration: {adjusted_duration:.2f}s")
        log_info(f"Number of videos: {len(video_paths)}")
        
        # If only one video, just use it directly
        if len(video_paths) == 1:
            log_info("Only one video provided, using it directly")
            video_path = video_paths[0]
            if isinstance(video_path, dict):
                video_path = video_path.get('path')
            elif isinstance(video_path, tuple) and len(video_path) == 2:
                # Handle tuple format (path, "short") for short videos
                video_path = video_path[0]
                
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
        log_info(f"Segment duration per video: {segment_duration:.2f}s")
        
        # Create a temporary directory for segment files
        temp_dir = "temp/segments"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process each video into a separate file
        segment_files = []
        
        for i, video_data in enumerate(video_paths):
            log_processing(f"Processing video {i+1}/{num_videos}")
            
            # Get path (handle both dict and string formats)
            if isinstance(video_data, dict):
                video_path = video_data.get('path')
            elif isinstance(video_data, tuple) and len(video_data) == 2:
                # Handle tuple format (path, "short") for short videos
                video_path = video_data[0]
                is_short = video_data[1] == "short"
            else:
                video_path = video_data
                is_short = False
            
            log_info(f"Video path: {video_path}")
            
            # Verify file exists
            if not os.path.exists(video_path):
                log_warning(f"Video file does not exist: {video_path}")
                continue
            
            try:
                # Load video
                video = VideoFileClip(video_path, fps_source="fps")
                
                # Handle short videos by looping them
                if video.duration < segment_duration:
                    if is_short or isinstance(video_data, tuple):
                        log_info(f"Short video ({video.duration:.2f}s), looping to reach segment duration")
                        video = vfx.loop(video, duration=segment_duration)
                    else:
                        # Even if not explicitly marked as short, loop it if needed
                        log_info(f"Video shorter than segment duration ({video.duration:.2f}s < {segment_duration:.2f}s), looping")
                        video = vfx.loop(video, duration=segment_duration)
                
                # Choose a random starting point if video is longer than needed
                if video.duration > segment_duration:
                    max_start = max(0, video.duration - segment_duration)
                    start_time = random.uniform(0, max_start)
                    # Create a subclip
                    video = video.subclip(start_time, start_time + segment_duration)
                
                # Resize to vertical format
                video = resize_to_vertical(video)
                
                # Save segment to a file
                segment_path = os.path.join(temp_dir, f"segment_{i}.mp4")
                video.write_videofile(
                    segment_path,
                    codec="libx264",
                    audio=False,
                    fps=30,
                    preset="medium",
                    threads=n_threads
                )
                
                segment_files.append(segment_path)
                log_success(f"Created segment {i+1}: {segment_path}")
                
                # Close clips
                video.close()
                
            except Exception as e:
                log_error(f"Error processing video {i+1}: {str(e)}")
                continue
        
        # Check if we have any segments
        if not segment_files:
            log_error("No valid segments created")
            return None
        
        log_info(f"Created {len(segment_files)} video segments")
        
        # If we only have one segment, just use it
        if len(segment_files) == 1:
            log_info("Only one segment created, using it directly")
            output_path = "temp_combined.mp4"
            shutil.copy(segment_files[0], output_path)
            return output_path
        
        # Create a file list for ffmpeg
        list_file = os.path.join(temp_dir, "segments.txt")
        with open(list_file, "w") as f:
            for segment in segment_files:
                f.write(f"file '{os.path.abspath(segment)}'\n")
        
        # Use ffmpeg to concatenate the segments
        output_path = "temp_combined.mp4"
        try:
            log_info("Concatenating segments with ffmpeg")
            
            # Build the ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                output_path
            ]
            
            # Run the command
            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode != 0:
                log_error(f"FFmpeg error: {process.stderr}")
                
                # Try an alternative approach using moviepy
                log_warning("FFmpeg concatenation failed, trying MoviePy concatenation")
                try:
                    # Load all segments
                    clips = [VideoFileClip(segment) for segment in segment_files]
                    
                    # Concatenate clips
                    final_clip = concatenate_videoclips(clips)
                    
                    # Write to file
                    final_clip.write_videofile(
                        output_path,
                        codec="libx264",
                        audio=False,
                        fps=30,
                        preset="medium",
                        threads=n_threads
                    )
                    
                    # Close clips
                    for clip in clips:
                        clip.close()
                    final_clip.close()
                    
                    log_success("Successfully concatenated segments using MoviePy")
                    return output_path
                except Exception as e:
                    log_error(f"MoviePy concatenation error: {str(e)}")
                    
                    # Fallback to using the first segment
                    log_warning("All concatenation methods failed, using first segment as fallback")
                    shutil.copy(segment_files[0], output_path)
            else:
                log_success("Successfully concatenated segments")
            
            return output_path
            
        except Exception as e:
            log_error(f"Error concatenating segments: {str(e)}")
            
            # Fallback to using the first segment
            try:
                log_warning("Using first segment as fallback")
                shutil.copy(segment_files[0], output_path)
                return output_path
            except Exception as e2:
                log_error(f"Error using fallback segment: {str(e2)}")
                return None
                
    except Exception as e:
        log_error(f"Error in combine_videos: {str(e)}")
        return None

def get_background_music(content_type: str, duration: float = None) -> str:
    """Get background music based on content type"""
    try:
        # Use the new music provider to get music
        music_path = asyncio.run(MusicProvider.get_music_for_channel(content_type, duration))
        
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
        
        # Load voice audio
        voice = AudioFileClip(voice_path)
        voice_duration = voice.duration
        log_info(f"Voice duration: {voice_duration:.2f}s")
        
        # Check if music path exists and is valid
        if not music_path or not os.path.exists(music_path):
            log_warning(f"No valid music path provided: {music_path}")
            log_info("Using voice audio only")
            # Normalize voice audio for consistent levels
            voice = voice.fx(afx.audio_normalize)
            # Ensure the voice starts after exactly 1 second of silence
            voice = voice.set_start(1.0)
            voice.write_audiofile(output_path, fps=44100, bitrate="192k")
            return output_path
        
        # Load and prepare music
        try:
            music = AudioFileClip(music_path)
            
            # Get file size in MB and duration for logging
            music_size_mb = os.path.getsize(music_path) / (1024 * 1024)
            music_duration = music.duration
            log_info(f"Music file: {os.path.basename(music_path)}")
            log_info(f"Music duration: {music_duration:.2f}s, size: {music_size_mb:.2f} MB")
            
            # Apply dynamic volume adjustment to music
            try:
                # Create a function that adjusts volume dynamically
                def adjust_volume(t):
                    # Reduce volume during speech segments
                    # This is a simple ducking effect - can be enhanced with more sophisticated analysis
                    return music_volume
                
                # Apply volume adjustment with error handling
                music = music.fl(lambda gf, t: gf(t) * adjust_volume(t), keep_duration=True)
            except ValueError as e:
                log_warning(f"Error applying dynamic volume: {str(e)}")
                # Fallback to simple volume adjustment
                music = music.volumex(music_volume)
            
            # Apply fades to music
            if fade_in > 0:
                music = music.audio_fadein(fade_in)
            
            if fade_out > 0:
                music = music.audio_fadeout(fade_out)
            
            # Handle music duration relative to voice
            if music_duration < voice_duration:
                log_info(f"Music shorter than voice ({music_duration:.2f}s < {voice_duration:.2f}s), looping")
                # Loop music to match voice duration
                music = afx.audio_loop(music, duration=voice_duration)
            elif music_duration > voice_duration:
                log_info(f"Music longer than voice ({music_duration:.2f}s > {voice_duration:.2f}s), trimming")
                # Trim music to match voice duration
                music = music.subclip(0, voice_duration)
            
            # Normalize voice audio for consistent levels
            voice = voice.fx(afx.audio_normalize)
            
            # Boost voice slightly to ensure clarity over music
            voice = voice.volumex(1.2)
            
            # Ensure the voice starts after exactly 1 second of silence
            voice = voice.set_start(1.0)
            
            # Composite audio - put music first in the list so voice is layered on top
            # This ensures the voice is more prominent in the mix
            final_audio = CompositeAudioClip([music, voice])
            
            # Write the mixed audio to the output path with high quality settings
            final_audio.write_audiofile(output_path, fps=44100, bitrate="192k")
            
            log_success(f"Successfully mixed voice and music: {os.path.basename(music_path)}")
            return output_path
            
        except Exception as music_error:
            log_warning(f"Error processing music: {str(music_error)}")
            log_info("Falling back to voice audio only")
            # Normalize voice audio for consistent levels
            voice = voice.fx(afx.audio_normalize)
            # Ensure the voice starts after exactly 1 second of silence
            voice = voice.set_start(1.0)
            voice.write_audiofile(output_path, fps=44100, bitrate="192k")
            return output_path
    
    except Exception as e:
        log_error(f"Error mixing audio: {str(e)}")
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
        
        # Detect and extract emojis
        emoji_pattern = emoji.get_emoji_regexp()
        
        # IMPROVED TEXT WRAPPING FOR SHORTS
        # Calculate max width for text (narrower than the full width)
        max_width = int(size[0] * 0.85)  # Use only 85% of width
        
        # Split text into words for wrapping
        words = txt.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            # Check if word contains emoji
            has_emoji = emoji_pattern.search(word)
            
            # Calculate word width (approximate for emojis)
            if has_emoji:
                # Count emojis in word
                emoji_count = len([c for c in word if emoji.is_emoji(c)])
                non_emoji_text = ''.join([c for c in word if not emoji.is_emoji(c)])
                
                try:
                    text_width = draw.textlength(non_emoji_text, font=font)
                except AttributeError:
                    text_width, _ = draw.textsize(non_emoji_text, font=font)
                
                word_width = text_width + (emoji_count * 80)  # Emoji width approximation
            else:
                try:
                    word_width = draw.textlength(word, font=font)
                except AttributeError:
                    word_width, _ = draw.textsize(word, font=font)
            
            # Check if adding this word exceeds max width
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + 10  # Add space width
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # If we have no lines (rare case), just use the original text
        if not lines:
            lines = [txt]
        
        # Calculate total height needed for all lines
        line_height = int(font.size * 1.2)  # 20% spacing between lines
        total_text_height = len(lines) * line_height
        
        # Position text vertically centered
        y_position = (size[1] - total_text_height) // 2
        
        # Draw each line
        for line in lines:
            # Detect and extract emojis for this line
            parts = []
            last_end = 0
            
            for match in emoji_pattern.finditer(line):
                start, end = match.span()
                if start > last_end:
                    parts.append((line[last_end:start], False))
                parts.append((line[start:end], True))
                last_end = end
            
            if last_end < len(line):
                parts.append((line[last_end:], False))
            
            # Calculate line width for centering
            text_without_emoji = "".join(part for part, is_emoji in parts if not is_emoji)
            try:
                text_width = draw.textlength(text_without_emoji, font=font)
            except AttributeError:
                text_width, _ = draw.textsize(text_without_emoji, font=font)
            
            # Calculate emoji count and approximate width
            emoji_count = sum(1 for _, is_emoji in parts if is_emoji)
            emoji_width = emoji_count * 80  # Approximate width of each emoji
            
            total_width = text_width + emoji_width
            x_position = (size[0] - total_width) / 2
            
            # Draw text with emojis
            current_x = x_position
            
            for part, is_emoji in parts:
                if is_emoji:
                    # Enhanced emoji handling - try multiple approaches
                    try:
                        # First try with our improved emoji handling
                        emoji_img = get_emoji_image(part, size=80)  # Increased from 70 to 80
                        if emoji_img:
                            img.paste(emoji_img, (int(current_x), int(y_position)), emoji_img)
                            current_x += 80  # Increased from 70 to 80
                        else:
                            # If that fails, just draw the emoji as text
                            # Draw text outline/stroke for better visibility
                            stroke_width = 3
                            for offset_x in range(-stroke_width, stroke_width + 1):
                                for offset_y in range(-stroke_width, stroke_width + 1):
                                    if offset_x == 0 and offset_y == 0:
                                        continue
                                    draw.text((current_x + offset_x, y_position + offset_y), part, fill=(0, 0, 0, 255), font=font)
                            
                            # Draw the main text
                            draw.text((current_x, y_position), part, fill=(255, 255, 255, 255), font=font)
                            try:
                                current_x += draw.textlength(part, font=font)
                            except AttributeError:
                                width, _ = draw.textsize(part, font=font)
                                current_x += width
                    except Exception as e:
                        log_warning(f"Error processing emoji: {str(e)}")
                        # Just draw the emoji as text with stroke
                        stroke_width = 3
                        for offset_x in range(-stroke_width, stroke_width + 1):
                            for offset_y in range(-stroke_width, stroke_width + 1):
                                if offset_x == 0 and offset_y == 0:
                                    continue
                                draw.text((current_x + offset_x, y_position + offset_y), part, fill=(0, 0, 0, 255), font=font)
                        
                        # Draw the main text
                        draw.text((current_x, y_position), part, fill=(255, 255, 255, 255), font=font)
                        try:
                            current_x += draw.textlength(part, font=font)
                        except AttributeError:
                            width, _ = draw.textsize(part, font=font)
                            current_x += width
                else:
                    # Draw regular text with stroke/outline for better visibility
                    stroke_width = 3
                    for offset_x in range(-stroke_width, stroke_width + 1):
                        for offset_y in range(-stroke_width, stroke_width + 1):
                            if offset_x == 0 and offset_y == 0:
                                continue
                            draw.text((current_x + offset_x, y_position + offset_y), part, fill=(0, 0, 0, 255), font=font)
                    
                    # Draw the main text
                    draw.text((current_x, y_position), part, fill=(255, 255, 255, 255), font=font)
                    try:
                        current_x += draw.textlength(part, font=font)
                    except AttributeError:
                        width, _ = draw.textsize(part, font=font)
                        current_x += width
            
            # Move to next line
            y_position += line_height
        
        return img
        
    except Exception as e:
        log_error(f"Error creating text with emoji: {str(e)}")
        return None

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
    """Trim silence from the beginning and end of an audio file with improved handling"""
    try:
        # If no output path is provided, use the input path
        if not output_path:
            output_path = audio_path
        
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        
        # Get the original duration
        original_duration = len(audio) / 1000.0  # in seconds
        
        # Only trim if the audio is longer than 3 seconds
        if original_duration <= 3:
            print(colored(f"Audio too short ({original_duration:.2f}s), skipping trimming", "yellow"))
            return audio_path
        
        # Detect silence at the beginning
        silence_threshold = -50  # dB
        min_silence_len = 100    # ms
        
        # Find silence at the beginning
        def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
            """Detect silence at the beginning of a sound file"""
            trim_ms = 0
            while trim_ms < len(sound) and sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
                trim_ms += chunk_size
            return trim_ms
        
        # Find silence at the end
        def detect_trailing_silence(sound, silence_threshold=-50.0, chunk_size=10):
            """Detect silence at the end of a sound file"""
            trim_ms = len(sound)
            while trim_ms > 0 and sound[trim_ms-chunk_size:trim_ms].dBFS < silence_threshold:
                trim_ms -= chunk_size
            return trim_ms
        
        # Detect silence
        start_trim = detect_leading_silence(audio, silence_threshold)
        end_trim = detect_trailing_silence(audio, silence_threshold)
        
        # Ensure we don't trim too much
        # Never trim more than 0.5s from the beginning
        start_trim = min(start_trim, 500)  # 500ms = 0.5s
        
        # For the end, calculate a safe trim amount based on audio duration
        # For longer audio, we can trim a bit more
        if original_duration > 10:
            # For longer audio, trim at most 0.3s from the end
            max_end_trim = 300  # 300ms = 0.3s
        else:
            # For shorter audio, trim at most 0.1s from the end
            max_end_trim = 100  # 100ms = 0.1s
        
        # Calculate how much we would trim from the end
        end_trim_amount = len(audio) - end_trim
        
        # Limit the end trim
        end_trim_amount = min(end_trim_amount, max_end_trim)
        end_trim = len(audio) - end_trim_amount
        
        # Trim the audio
        trimmed_audio = audio[start_trim:end_trim]
        
        # Calculate how much was trimmed
        trimmed_duration = len(trimmed_audio) / 1000.0  # in seconds
        trimmed_amount = original_duration - trimmed_duration
        
        # Only save if we actually trimmed something
        if trimmed_amount > 0.05:  # Only save if we trimmed more than 50ms
            # Export the trimmed audio
            trimmed_audio.export(output_path, format="mp3")
            print(colored(f"Trimmed {trimmed_amount:.2f}s from audio (start: {start_trim/1000:.2f}s, end: {end_trim_amount/1000:.2f}s)", "green"))
        else:
            print(colored("No significant silence detected, using original audio", "blue"))
            # If we didn't trim anything significant, just copy the original file
            if output_path != audio_path:
                shutil.copy(audio_path, output_path)
        
        return output_path
        
    except Exception as e:
        print(colored(f"Error trimming audio: {str(e)}", "red"))
        traceback.print_exc()
        # Return the original file if trimming fails
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
    """Process subtitles and add them to the video with improved styling"""
    try:
        # Read subtitles
        subtitle_data = read_subtitles_file(subs_path)
        
        if not subtitle_data:
            log_error("No valid subtitles found")
            return base_video
            
        log_success(f"Found {len(subtitle_data)} subtitles")
        
        # Convert tuple format to dictionary format for easier processing
        processed_subtitles = []
        for times, text in subtitle_data:
            processed_subtitles.append({
                'start': times[0],
                'end': times[1],
                'text': text
            })
        
        # Prepare subtitle images
        log_info("Preparing subtitle images...")
        
        # Extend the last subtitle to the end of the video to prevent it from disappearing too early
        if processed_subtitles:
            last_sub = processed_subtitles[-1]
            video_end = base_video.duration
            
            # Always extend the last subtitle to the full end of the video
            # This ensures it stays on screen until the very end
            log_success("Extended final subtitle until end of video")
            last_sub['end'] = video_end  # No buffer at the end, stay until the very last frame
        
        # Apply start padding offset to all subtitles
        if start_padding > 0:
            for sub in processed_subtitles:
                sub['start'] += start_padding
                sub['end'] += start_padding
            log_info(f"Applied {start_padding}s offset to all subtitles for start padding")
        
        # Show progress for larger subtitle sets
        total_subs = len(processed_subtitles)
        subtitle_clips = []
        
        for i, data in enumerate(processed_subtitles):
            # Create subtitle background with text
            style = {
                'font_size': 40,
                'bg_color': (0, 0, 0, 180),  # Semi-transparent black
                'text_color': (255, 255, 255),
                'padding': 15,
                'border_radius': 10
            }
            
            # Check if this is the last subtitle
            is_last = (i == total_subs - 1)
            
            try:
                # Create text image with emoji support
                text_image = create_text_with_emoji(data['text'], size=(1080, 800))
                
                if text_image is None:
                    continue
                    
                # Convert PIL image to MoviePy clip
                img_array = np.array(text_image)
                duration = data['end'] - data['start']
                
                # Create clip with NO fade effects - completely eliminate transitions
                # Position in the center of the screen (both horizontally and vertically)
                sub_clip = (ImageClip(img_array)
                    .set_duration(duration)
                    .set_position(('center', 'center'))  # Center in the middle of the screen
                    .set_start(data['start']))
                
                subtitle_clips.append(sub_clip)
            except Exception as clip_error:
                log_error(f"Error creating subtitle clip {i+1}: {str(clip_error)}")
            
            # Show progress for larger subtitle sets
            if total_subs > 5 and (i % 2 == 0 or i == total_subs - 1):
                log_progress(i + 1, total_subs, prefix="Subtitle Preparation:", suffix="Complete")
        
        # If no subtitle clips were created, return the base video
        if not subtitle_clips:
            log_warning("No valid subtitle clips were created, returning base video")
            return base_video
            
        # Create final video with subtitles
        log_processing("Compositing final video with centered subtitles")
        final_video = CompositeVideoClip([base_video] + subtitle_clips)
        
        log_success("Added centered subtitles with clean transitions", "🎬")
        return final_video
    
    except Exception as e:
        log_error(f"Error processing subtitles: {str(e)}")
        log_error(traceback.format_exc())
        return base_video

def generate_video(background_path, audio_path, subtitles_path=None, content_type=None, target_duration=None, 
                  use_background_music=True, music_volume=0.3, music_fade_in=2.0, music_fade_out=3.0, script_path=None):
    """Generate a video with audio and subtitles"""
    try:
        log_section("Video Generation", "🎬")
        
        # Load audio to get duration
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        log_info(f"Audio duration: {audio_duration:.2f}s")
        
        # Calculate video duration with padding
        # Note: If using delayed audio (with 1s silence at start), we don't need additional start padding
        # Check if the audio filename contains 'delayed' which indicates it already has the start padding
        if '_delayed.' in audio_path:
            start_padding = 0.0  # No additional start padding needed
            log_info("Using audio with built-in delay, no additional start padding needed")
        else:
            start_padding = 1.0  # 1 second at start
            
        end_padding = 3.0    # 3 seconds at end (increased from 2 to 3 seconds)
        video_duration = audio_duration + start_padding + end_padding
        log_info(f"Video duration: {video_duration:.2f}s (with {start_padding}s start and {end_padding}s end padding)")
        
        # Use target_duration if provided, otherwise use calculated duration
        if target_duration:
            log_info(f"Using provided target duration: {target_duration:.2f}s")
            # IMPORTANT: Slightly reduce the target duration to prevent black fade at end
            # Using a smaller adjustment to minimize the gap at the end
            video_duration = target_duration - 0.05
            log_info(f"Adjusted to {video_duration:.2f}s to prevent black end frame")
        
        # Handle different background path formats
        if isinstance(background_path, list) and len(background_path) > 0:
            # Check if we have multiple videos to combine
            if len(background_path) > 1:
                log_info(f"Combining {len(background_path)} videos for a dynamic background")
                try:
                    # Use the combine_videos function to create a dynamic background
                    combined_video_path = combine_videos(background_path, audio_duration, video_duration)
                    if combined_video_path and os.path.exists(combined_video_path):
                        background_video = VideoFileClip(combined_video_path, fps_source="fps")
                        log_success("Successfully combined multiple videos for a dynamic background")
                    else:
                        # Fallback to using the first video if combination fails
                        log_warning("Video combination failed, using first video as fallback")
                        background_video_path = background_path[0]
                        if isinstance(background_video_path, dict):
                            background_video_path = background_video_path.get('path')
                        background_video = VideoFileClip(background_video_path, fps_source="fps")
                except Exception as e:
                    log_error(f"Error in video combination: {str(e)}")
                    # Fallback to using the first video
                    log_warning("Using first video as fallback due to combination error")
                    background_video_path = background_path[0]
                    if isinstance(background_video_path, dict):
                        background_video_path = background_video_path.get('path')
                    background_video = VideoFileClip(background_video_path, fps_source="fps")
            else:
                # Just one video in the list
                log_info(f"Using single video from list")
                background_video_path = background_path[0]
                if isinstance(background_video_path, dict):
                    background_video_path = background_video_path.get('path')
                
                # Verify the file exists and is readable
                if not os.path.exists(background_video_path) or os.path.getsize(background_video_path) == 0:
                    log_error(f"Background video file does not exist or is empty: {background_video_path}")
                    # Create a default background
                    log_warning("Creating default background")
                    background_video = ColorClip(size=(1080, 1920), color=(25, 45, 65), duration=video_duration)
                else:
                    try:
                        background_video = VideoFileClip(background_video_path, fps_source="fps")
                    except Exception as e:
                        log_error(f"Error loading background video: {str(e)}")
                        # Create a default background
                        log_warning("Creating default background")
                        background_video = ColorClip(size=(1080, 1920), color=(25, 45, 65), duration=video_duration)
        else:
            # Single background video or no background
            if background_path and os.path.exists(background_path):
                log_info(f"Using single background video: {background_path}")
                try:
                    background_video = VideoFileClip(background_path, fps_source="fps")
                except Exception as e:
                    log_error(f"Error loading background video: {str(e)}")
                    # Create a default background
                    log_warning("Creating default background")
                    background_video = ColorClip(size=(1080, 1920), color=(25, 45, 65), duration=video_duration)
            else:
                # Create a default background
                log_warning("No valid background video provided, creating default")
                background_video = ColorClip(size=(1080, 1920), color=(25, 45, 65), duration=video_duration)
        
        # Process the background video
        try:
            # Resize and crop to vertical format
            log_processing("Resizing video to vertical format")
            background_video = resize_to_vertical(background_video)
            
            # Loop if needed to match audio duration
            if background_video.duration < video_duration:
                log_processing(f"Looping background video ({background_video.duration:.2f}s) to match audio ({video_duration:.2f}s)")
                background_video = vfx.loop(background_video, duration=video_duration)
            
            # Trim if longer than needed
            if background_video.duration > video_duration:
                log_processing(f"Trimming background video from {background_video.duration:.2f}s to {video_duration:.2f}s")
                background_video = background_video.subclip(0, video_duration)
            
            # Add audio to video with delay
            log_processing("Adding audio to video with delay")
            
            # Add background music if enabled
            if use_background_music:
                try:
                    log_processing("Adding audio to video with delay")
                    
                    # Initialize music provider
                    from music_provider import MusicProvider
                    music_provider = MusicProvider()
                    if music_provider.initialize_freesound_client():
                        log_success("Freesound API client initialized")
                    
                    # Load used music to avoid repetition
                    music_provider.load_used_music()
                    
                    # Get background music for the content type
                    log_info(f"Getting background music for {content_type}...")
                    
                    # Get music path - first try to extract keywords from script if available
                    music_path = None
                    video_keywords = []
                    
                    if script_path and os.path.exists(script_path):
                        try:
                            # Read the script file
                            with open(script_path, 'r', encoding='utf-8') as f:
                                script_text = f.read()
                            
                            # Extract keywords from script
                            import re
                            from collections import Counter
                            import nltk
                            from nltk.corpus import stopwords
                            
                            try:
                                nltk.data.find('corpora/stopwords')
                            except LookupError:
                                nltk.download('stopwords')
                            
                            # Clean the script text
                            script_text = re.sub(r'[^\w\s]', '', script_text.lower())
                            
                            # Tokenize and count words
                            words = script_text.split()
                            stop_words = set(stopwords.words('english'))
                            filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
                            
                            # Count word frequency
                            word_count = Counter(filtered_words)
                            
                            # Get the most frequent words as keywords
                            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
                            video_keywords = [word for word, count in sorted_words[:5]]
                            log_info(f"Extracted keywords from script: {', '.join(video_keywords)}")
                        except Exception as e:
                            log_warning(f"Error extracting keywords from script: {str(e)}")
                    
                    # Handle async call properly based on context
                    try:
                        # Try to get the current event loop
                        loop = asyncio.get_event_loop()
                        
                        # Create a synchronous version of the async function
                        def get_music_sync():
                            # Use a new event loop in a separate thread to avoid conflicts
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                # Use the new download_music_for_video function if keywords are available
                                if video_keywords:
                                    return new_loop.run_until_complete(
                                        music_provider.download_music_for_video(content_type, video_keywords, video_duration)
                                    )
                                else:
                                    return new_loop.run_until_complete(
                                        music_provider.get_music_for_channel(content_type, video_duration)
                                    )
                            finally:
                                new_loop.close()
                        
                        # Run the synchronous function in a thread executor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            music_path = executor.submit(get_music_sync).result(timeout=30)
                            
                    except Exception as e:
                        log_warning(f"Error in async music retrieval: {str(e)}")
                        # Fallback to a default music file
                        music_path = music_provider.get_default_music(content_type)
                    
                    if music_path and os.path.exists(music_path):
                        # Get file size and name for logging
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
                            # Load the mixed audio
                            final_audio = AudioFileClip(mixed_audio_path)
                            
                            # Set audio to video
                            video = background_video.set_audio(final_audio)
                            log_success("Added background music to video using enhanced mixing")
                            log_info(f"Music: {os.path.basename(music_path)} | Volume: {music_volume}")
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
        
        # Extract main text from script (first line usually)
        lines = script.strip().split('\n')
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
        emojis = [c for c in script if emoji.is_emoji(c)]
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
            gradient = generator.create_gradient(template['gradient'])
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
            print(colored(f"Error loading fonts: {str(e)}", "red"))
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            
        draw = ImageDraw.Draw(img)
        
        # Add text with enhanced shadow
        generator.draw_enhanced_text(
            draw, 
            template['text'], 
            title_font, 
            template['text_color'],
            template['shadow_color'],
            offset_y=-200
        )
        
        generator.draw_enhanced_text(
            draw, 
            template['subtitle'], 
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
        print(colored(f"✓ Generated thumbnail: {output_path}", "green"))
        
        return output_path
    
    except Exception as e:
        print(colored(f"Error generating thumbnail: {str(e)}", "red"))
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