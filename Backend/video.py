import os
import uuid
import requests
from typing import List
from moviepy.editor import *
from termcolor import colored
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

# First activate your virtual environment and install pysrt:
# python -m pip install pysrt --no-cache-dir

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
    """Generate SRT subtitles from script with fixed emoji handling"""
    try:
        # Get audio duration
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        # Create temp directory
        os.makedirs("temp/subtitles", exist_ok=True)
        subtitles_path = "temp/subtitles/generated_subtitles.srt"
        
        # Process script line by line to preserve original emoji positioning
        lines = script.replace('"', '').split('\n')
        sentences = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove any numeric prefixes (like "1.")
            line = re.sub(r'^\d+\.\s*', '', line)
            sentences.append(line)
        
        # Write SRT file directly from the original lines
        with open(subtitles_path, "w", encoding="utf-8-sig") as f:
            current_time = 0
            
            for i, sentence in enumerate(sentences):
                # Calculate word count for timing
                word_count = len([w for w in sentence.split() if not all(emoji.is_emoji(c) for c in w)])
                
                # Calculate duration - shorter for better pacing
                est_duration = max(1.3, min(2.8, word_count * 0.3))
                
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
                
                # Gap between subtitles
                current_time += est_duration + 0.15
            
        print(colored(f"Generated {len(sentences)} subtitles", "green"))
        return subtitles_path
    
    except Exception as e:
        print(colored(f"Error generating subtitles: {str(e)}", "red"))
        return None

def combine_videos(video_paths, audio_duration, target_duration, n_threads=2):
    """Combine multiple videos with smooth transitions for YouTube Shorts"""
    try:
        # Calculate total target duration
        total_duration = audio_duration if audio_duration else target_duration
        
        # IMPORTANT: Slightly reduce the target duration to prevent black fade at end
        # This ensures the video ends on the last frame without fading to black
        adjusted_duration = total_duration - 0.1  # Reduce by 0.1 seconds
        
        print(colored("\n=== Video Combination Debug Info ===", "blue"))
        print(colored(f"Audio duration: {total_duration:.2f}s", "cyan"))
        print(colored(f"Target duration: {target_duration:.2f}s", "cyan"))
        print(colored(f"Adjusted duration: {adjusted_duration:.2f}s", "cyan"))
        print(colored(f"Number of videos: {len(video_paths)}", "cyan"))
        
        # Calculate how long each video should be
        num_videos = len(video_paths)
        segment_duration = adjusted_duration / num_videos
        print(colored(f"Segment duration per video: {segment_duration:.2f}s", "cyan"))
        
        # Keep track of current position in the timeline
        current_time = 0
        
        # Create clips list with positions for each clip
        positioned_clips = []
        
        # Transition duration (shorter for smoother transitions)
        transition_duration = 0.5
        
        # Store the last processed clip for potential end-frame freeze
        last_processed_clip = None
        last_video_path = None
        
        for i, video_data in enumerate(video_paths):
            try:
                print(colored(f"\n=== Processing Video {i+1}/{num_videos} ===", "blue"))
                
                # Get path (handle both dict and string formats)
                if isinstance(video_data, dict):
                    video_path = video_data.get('path')
                else:
                    video_path = video_data
                
                print(colored(f"Video path: {video_path}", "cyan"))
                
                # Store the last video path for potential reuse
                last_video_path = video_path
                
                # Load video
                video = VideoFileClip(video_path)
                print(colored(f"Video duration: {video.duration:.2f}s", "cyan"))
                
                # Calculate how much of this video to use
                this_segment_duration = min(segment_duration, adjusted_duration - current_time)
                if this_segment_duration <= 0:
                    print(colored("Skipping video (no time left)", "yellow"))
                    continue
                
                # Determine which portion to use
                video_duration = video.duration
                
                # Avoid the first 10% and last 10% of the video for better segments
                usable_start = video_duration * 0.1
                usable_end = video_duration * 0.9
                usable_duration = usable_end - usable_start
                
                # If the usable portion is too short, use the whole video
                if usable_duration < this_segment_duration:
                    usable_start = 0
                    usable_end = video_duration
                    usable_duration = video_duration
                
                # Choose a starting point
                latest_possible_start = usable_end - this_segment_duration
                if latest_possible_start < usable_start:
                    video_start = usable_start
                else:
                    # Use 25% through the usable portion for more interesting content
                    video_start = usable_start + (usable_duration * 0.25)
                    if video_start > latest_possible_start:
                        video_start = usable_start
                
                print(colored(f"Using segment: {video_start:.2f}s to {video_start + this_segment_duration:.2f}s", "cyan"))
                
                # Create clip - Use explicit values instead of lambda functions
                clip = video.subclip(video_start, video_start + this_segment_duration + (transition_duration if i < num_videos - 1 else 0))
                
                # Store the last clip for potential freeze frame
                last_processed_clip = clip
                
                # Resize to maintain height
                clip = clip.resize(height=1920)
                
                # Fix the crop to use explicit values instead of lambda functions
                if clip.w > 1080:
                    x_center = clip.w / 2
                    x1 = int(x_center - 540)  # 540 is half of 1080
                    x2 = int(x_center + 540)
                    clip = clip.crop(x1=x1, y1=0, x2=x2, y2=1920)
                
                # If video is too narrow, zoom to fill width
                if clip.w < 1080:
                    scale_factor = 1080 / clip.w
                    clip = clip.resize(width=1080)
                    # If this made it too tall, crop height
                    if clip.h > 1920:
                        y_center = clip.h / 2
                        y1 = int(y_center - 960)  # 960 is half of 1920
                        y2 = int(y_center + 960)
                        clip = clip.crop(x1=0, y1=y1, x2=1080, y2=y2)
                
                # Apply subtle zoom effect (simple version without lambda)
                if i % 2 == 0:  # Alternate between zoom in and out
                    # For zoom effects, we'll use a simpler approach
                    clip = clip.resize(1.02)  # Just a static 2% zoom
                else:
                    clip = clip.resize(1.03)  # 3% zoom
                
                # Apply very subtle brightness adjustment
                clip = clip.fx(vfx.colorx, 1.05)  # Simpler than using enhancer
                
                # Calculate clip position with overlapping for transitions
                # Each clip starts a bit before the previous one ends
                clip_start = current_time
                if i > 0:
                    clip_start -= transition_duration
                
                # Add clip with position
                positioned_clips.append((clip, clip_start))
                
                # Update current time
                current_time += this_segment_duration
                print(colored(f"Current total time: {current_time:.2f}s / {adjusted_duration:.2f}s", "green"))
                
            except Exception as e:
                print(colored(f"Error processing video {i+1}:", "red"))
                print(colored(f"Error details: {str(e)}", "red"))
                print(colored("Creating fallback styled clip", "yellow"))
                
                # Create a styled fallback clip instead of plain black
                try:
                    # Create styled background using thumbnail generator principles
                    from thumbnail_generator import ThumbnailGenerator
                    thumbnail_gen = ThumbnailGenerator()
                    
                    # Generate appropriate background based on segment index
                    colors = [
                        ['#3a1c71', '#d76d77'],  # Purple gradient
                        ['#4e54c8', '#8f94fb'],  # Blue gradient
                        ['#11998e', '#38ef7d'],  # Green gradient
                        ['#e1eec3', '#f05053'],  # Yellow-red gradient
                        ['#355c7d', '#6c5b7b']   # Cool blue-purple gradient
                    ]
                    
                    color_choice = i % len(colors)
                    gradient = thumbnail_gen.create_gradient(colors[color_choice])
                    
                    # Convert to numpy array for MoviePy
                    gradient_array = np.array(gradient)
                    
                    # Create a nicer fallback clip with gradient
                    fallback_clip = ImageClip(gradient_array, duration=this_segment_duration)
                    fallback_clip = fallback_clip.resize((1080, 1920))
                    
                    # Add a subtle text label
                    txt_clip = TextClip("Next tip...", fontsize=70, color='white', 
                                       font='Arial-Bold', stroke_color='black', stroke_width=2)
                    txt_clip = txt_clip.set_position(('center', 'center')).set_duration(this_segment_duration)
                    
                    fallback_clip = CompositeVideoClip([fallback_clip, txt_clip])
                    
                    # Store this as the last processed clip
                    last_processed_clip = fallback_clip
                    
                    # Add clip with position
                    positioned_clips.append((fallback_clip, current_time))
                    current_time += this_segment_duration
                    
                except Exception:
                    # If styled clip fails, use simple color clip
                    fallback_clip = ColorClip(size=(1080, 1920), color=(25, 45, 65), duration=this_segment_duration)
                    last_processed_clip = fallback_clip
                    positioned_clips.append((fallback_clip, current_time))
                    current_time += this_segment_duration
        
        print(colored("\n=== Creating Final Composite Video ===", "blue"))
        print(colored(f"Number of clips to combine: {len(positioned_clips)}", "cyan"))
        
        # SPECIAL HANDLING FOR LAST FRAME: 
        # Instead of adding a freeze frame, extend the last clip to cover the full duration
        # This ensures the video ends on the actual last frame of the last clip
        if positioned_clips:
            last_clip, last_start = positioned_clips[-1]
            
            # Calculate how much we need to extend the last clip
            current_end = last_start + last_clip.duration
            extension_needed = adjusted_duration - current_end + 0.5  # Add a small buffer
            
            if extension_needed > 0:
                print(colored(f"Extending last clip by {extension_needed:.2f}s to prevent black end frame", "yellow"))
                
                try:
                    # Get the last frame of the last clip
                    last_frame_time = max(0, min(last_clip.duration - 0.05, last_clip.duration * 0.9))
                    last_frame = last_clip.get_frame(last_frame_time)
                    
                    # Create a freeze frame of the exact last frame
                    freeze_clip = ImageClip(last_frame, duration=extension_needed)
                    
                    # Position it to start exactly when the last clip ends (minus a tiny overlap)
                    freeze_start = current_end - 0.05  # Small overlap for smooth transition
                    freeze_clip = freeze_clip.set_start(freeze_start)
                    
                    # Add to positioned clips
                    positioned_clips.append((freeze_clip, freeze_start))
                    print(colored("✓ Extended last clip with its final frame", "green"))
                except Exception as e:
                    print(colored(f"Error extending last clip: {str(e)}", "yellow"))
        
        # Create a list of clips with set_start times to handle proper overlapping transitions
        composite_clips = []
        for clip, start_time in positioned_clips:
            # Set the start time for each clip for proper overlapping
            composite_clips.append(clip.set_start(start_time))
        
        # Create composite with all clips
        final_clip = CompositeVideoClip(composite_clips, size=(1080, 1920))
        print(colored(f"Final clip duration: {final_clip.duration:.2f}s", "cyan"))
        
        # IMPORTANT: Set exact duration to match the adjusted duration
        # This prevents MoviePy from adding any automatic fade at the end
        if abs(final_clip.duration - adjusted_duration) > 0.1:
            print(colored(f"Setting exact duration to {adjusted_duration:.2f}s", "yellow"))
            final_clip = final_clip.set_duration(adjusted_duration)
        
        # Save combined video with specific end behavior
        output_path = "temp_combined.mp4"
        print(colored(f"\nSaving to: {output_path}", "blue"))
        
        # IMPORTANT: Disable any automatic fades by setting specific parameters
        final_clip.write_videofile(
            output_path,
            threads=n_threads,
            codec='libx264',
            audio=False,
            fps=30,
            ffmpeg_params=["-shortest"]  # This helps prevent extra frames
        )
        
        # Clean up
        print(colored("\nCleaning up resources...", "blue"))
        for clip, _ in positioned_clips:
            clip.close()
        final_clip.close()
        
        return output_path
        
    except Exception as e:
        print(colored("\n=== Error in combine_videos ===", "red"))
        print(colored(f"Error type: {type(e).__name__}", "red"))
        print(colored(f"Error details: {str(e)}", "red"))
        return None

def get_background_music(content_type: str, duration: float = None) -> str:
    """Get background music from Pixabay based on content type"""
    try:
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

def mix_audio(voice_path: str, music_path: str, output_path: str, music_volume: float = 0.1) -> str:
    """Mix voice audio with background music"""
    try:
        # Load audio clips
        voice = AudioFileClip(voice_path)
        music = AudioFileClip(music_path)

        # Loop music if needed
        if music.duration < voice.duration:
            loops = int(np.ceil(voice.duration / music.duration))
            music = concatenate_audioclips([music] * loops)
        
        # Trim music to match voice duration
        music = music.subclip(0, voice.duration)
        
        # Adjust music volume
        music = music.volumex(music_volume)
        
        # Composite audio
        final_audio = CompositeAudioClip([voice, music])
        
        # Write output
        final_audio.write_audiofile(output_path, fps=44100)
        
        # Clean up
        voice.close()
        music.close()
        final_audio.close()
        
        return output_path

    except Exception as e:
        print(colored(f"Error mixing audio: {str(e)}", "red"))
        return None

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
        
        # Pre-process the emoji string to remove all variation selectors
        # This includes FE0F (variation selector-16) and other invisible formatting characters
        cleaned_emoji = ''.join(c for c in emoji_char if not (0xFE00 <= ord(c) <= 0xFE0F))
        
        # Convert emoji to unicode code points
        emoji_code = "-".join(
            format(ord(c), 'x').lower()
            for c in cleaned_emoji
            if not (ord(c) >= 0xFE00 and ord(c) <= 0xFE0F)  # More thoroughly skip all variation selectors
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
                print(colored(f"Failed to download emoji: {cleaned_emoji} (HTTP {response.status_code})", "yellow"))
                return create_fallback_emoji(size, cleaned_emoji)
        except Exception as e:
            print(colored(f"Error downloading emoji: {str(e)}", "red"))
            return create_fallback_emoji(size, cleaned_emoji)
            
    except Exception as e:
        print(colored(f"Error getting emoji image for {emoji_char}: {str(e)}", "red"))
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
    """Create text with colored emojis using Pillow"""
    try:
        # Create a transparent background for Shorts
        image = Image.new('RGBA', (1080, 800), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # First split into sentences to preserve order
        sentences = txt.split('. ')
        formatted_lines = []
        
        for sentence in sentences:
            # Process each sentence
            words = []
            fragments = sentence.split()
            
            # Process each word, keeping emojis separate
            for fragment in fragments:
                emoji_parts = []
                text_parts = []
                current_text = ""
                
                # Find emojis in this fragment
                for char in fragment:
                    if emoji.is_emoji(char):
                        if current_text:
                            text_parts.append(current_text)
                            current_text = ""
                        emoji_parts.append(char)
                    else:
                        current_text += char
                
                if current_text:
                    text_parts.append(current_text)
                
                # Add text first, then emojis (to maintain order)
                if text_parts:
                    words.append(("text", " ".join(text_parts)))
                for emoji_char in emoji_parts:
                    words.append(("emoji", emoji_char))
            
            # Add this processed sentence
            if words:
                formatted_lines.append(words)
        
        # Load font - slightly smaller for better fitting
        try:
            text_font = ImageFont.truetype('arial.ttf', 95)
        except:
            text_font = ImageFont.load_default()
        
        # Compute wrapped lines with proper word breaks
        final_lines = []
        max_width = 850  # Narrow width to ensure wrapping
        
        for sentence_parts in formatted_lines:
            current_line = []
            current_width = 0
            
            for part_type, content in sentence_parts:
                width_to_add = 0
                if part_type == "text":
                    # Split long text if needed
                    words = content.split()
                    if len(words) > 1:
                        for word in words:
                            word_bbox = draw.textbbox((0, 0), word + " ", font=text_font)
                            word_width = word_bbox[2] - word_bbox[0]
                            
                            if current_width + word_width > max_width and current_line:
                                final_lines.append(current_line)
                                current_line = [("text", word)]
                                current_width = word_width
                            else:
                                current_line.append(("text", word))
                                current_width += word_width + 10
                    else:
                        word_bbox = draw.textbbox((0, 0), content + " ", font=text_font)
                        word_width = word_bbox[2] - word_bbox[0]
                        
                        if current_width + word_width > max_width and current_line:
                            final_lines.append(current_line)
                            current_line = [("text", content)]
                            current_width = word_width
                        else:
                            current_line.append(("text", content))
                            current_width += word_width + 10
                else:  # Emoji
                    emoji_width = 130
                    if current_width + emoji_width > max_width and current_line:
                        final_lines.append(current_line)
                        current_line = [("emoji", content)]
                        current_width = emoji_width
                    else:
                        current_line.append(("emoji", content))
                        current_width += emoji_width + 10
            
            if current_line:
                final_lines.append(current_line)
        
        # Calculate total height for all wrapped lines
        line_height = max(text_font.size, 130) + 30
        total_height = len(final_lines) * line_height
        y = (image.height - total_height) // 2
        
        # Draw each line
        for line in final_lines:
            # Calculate line width for centering
            line_width = 0
            for part_type, content in line:
                if part_type == "text":
                    bbox = draw.textbbox((0, 0), content + " ", font=text_font)
                    line_width += bbox[2] - bbox[0]
                else:  # emoji
                    line_width += 130 + 10
            
            # Center this line
            x = (image.width - line_width) // 2
            
            # Draw each part
            for part_type, content in line:
                if part_type == "text":
                    # Draw text with stroke
                    stroke_width = 6
                    for adj in range(-stroke_width, stroke_width+1):
                        for adj2 in range(-stroke_width, stroke_width+1):
                            draw.text((x+adj, y+adj2), content, font=text_font, fill=(0, 0, 0, 255))
                    
                    draw.text((x, y), content, font=text_font, fill=(255, 255, 255, 255))
                    bbox = draw.textbbox((0, 0), content + " ", font=text_font)
                    x += bbox[2] - bbox[0] + 5
                else:
                    # Draw emoji
                    emoji_img = get_emoji_image(content, size=130)
                    if emoji_img:
                        emoji_y = y + (text_font.size - 130) // 2
                        image.paste(emoji_img, (x, emoji_y), emoji_img)
                    x += 130 + 10
            
            y += line_height
        
        return image
    
    except Exception as e:
        print(colored(f"Error creating text with emoji: {str(e)}", "red"))
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
        
        # Create image with text and emojis
        text_image = create_text_with_emoji(clean_txt)
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
            .set_position(('center', 'center'))
            .crossfadein(0.2))  # Slightly longer fade in
        
        # Add fade out based on position
        if is_last:
            final_clip = final_clip.crossfadeout(0.5)
        else:
            final_clip = final_clip.crossfadeout(0.3)  # Longer fade out for better transitions
        
        return final_clip
        
    except Exception as e:
        print(colored(f"Subtitle error: {str(e)}", "red"))
        print(colored(f"Text content: {txt}", "yellow"))
        return None

def trim_audio_file(audio_path, trim_end=0.15):
    """Trim and clean audio file for professional sound"""
    try:
        # Load audio
        audio = AudioFileClip(audio_path)
        
        # Apply trimming and audio processing
        if audio.duration > trim_end + 0.5:
            # Trim end and slightly fade out
            trimmed_audio = audio.subclip(0, audio.duration - trim_end)
            # Add subtle fadeout to prevent abrupt ending
            trimmed_audio = trimmed_audio.audio_fadeout(0.15)
            
            # Save to a new output path
            output_dir = os.path.dirname(audio_path)
            base_name = os.path.basename(audio_path)
            name_parts = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name_parts[0]}_clean{name_parts[1]}")
            
            # Write to new file with high quality
            trimmed_audio.write_audiofile(
                output_path, 
                codec='mp3', 
                fps=44100, 
                bitrate="192k"  # Higher bitrate for better quality
            )
            print(colored(f"✓ Enhanced audio quality and trimmed {trim_end}s", "green"))
            
            # Clean up
            audio.close()
            trimmed_audio.close()
            
            return output_path
        
        return audio_path
        
    except Exception as e:
        print(colored(f"Error processing audio: {str(e)}", "yellow"))
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

def process_subtitles(subs_path, base_video):
    """Process subtitles with clean, instant transitions"""
    try:
        print(colored("\n=== Processing Subtitles ===", "blue"))
        
        # Load subtitles - validate they're proper SRT format
        try:
            subs = pysrt.open(subs_path)
            # Filter out any overly long subtitles that might be errors
            subs = [sub for sub in subs if len(sub.text) < 200]  # Skip any suspiciously long subtitle
            print(colored(f"Found {len(subs)} subtitles", "green"))
        except Exception as e:
            print(colored(f"Error loading subtitles: {str(e)}", "red"))
            return base_video
        
        # Pre-render all subtitle images with consistent styling
        subtitle_data = []
        
        for i, sub in enumerate(subs):
            # Clean subtitle text
            text = sub.text.replace('\n', ' ').strip().strip('"')
            if not text:
                continue
                
            # Skip duplicate/combined subtitles (longer than 100 chars)
            if len(text) > 100:
                print(colored(f"Skipping long subtitle: {text[:30]}...", "yellow"))
                continue
            
            # Calculate timing
            start_time = sub.start.ordinal / 1000.0
            end_time = sub.end.ordinal / 1000.0
            
            # Handle last subtitle - make it stay until end of video
            is_last = (i == len(subs) - 1)
            if is_last:
                # ALWAYS extend the last subtitle to the end to prevent black screens
                end_time = base_video.duration
                print(colored(f"✓ Extended final subtitle until end of video", "green"))
            
            # Create text image
            text_image = create_text_with_emoji(text)
            if text_image is None:
                continue
                
            # Store the pre-rendered image
            subtitle_data.append({
                'text': text,
                'image': text_image,
                'start': start_time,
                'end': end_time,
                'is_last': is_last,
                'is_first': (i == 0)
            })
            
            print(colored(f"Prepared subtitle: {text}", "green"))
        
        # Fast track: If no subtitles, return base video
        if not subtitle_data:
            return base_video
        
        # Create clips for each subtitle with NO transitions
        subtitle_clips = []
        
        for data in subtitle_data:
            # Convert PIL image to MoviePy clip
            img_array = np.array(data['image'])
            duration = data['end'] - data['start']
            
            # Create clip with NO fade effects - completely eliminate transitions
            sub_clip = (ImageClip(img_array)
                .set_duration(duration)
                .set_position(('center', 'center'))
                .set_start(data['start']))
            
            # NO FADES AT ALL - this prevents any black text issues
            # Just use the clip as is with no transitions
            
            subtitle_clips.append(sub_clip)
        
        # Create a semi-transparent overlay for the entire video
        overlay = ColorClip(size=base_video.size, color=(0, 0, 0))
        overlay = overlay.set_opacity(0.15)  # Very subtle darkening
        overlay = overlay.set_duration(base_video.duration)
        
        # Create final video with subtitles
        final_video = CompositeVideoClip([base_video, overlay] + subtitle_clips)
        
        print(colored("✓ Added subtitles with clean, instant transitions (no fades)", "green"))
        return final_video
    
    except Exception as e:
        print(colored(f"Error processing subtitles: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))
        return base_video

def generate_video(background_path, audio_path, subtitles_path=None, content_type=None, target_duration=None):
    """Generate a video with audio and subtitles"""
    try:
        print(colored("\n=== Video Generation ===", "blue"))
        
        # Load audio to get duration
        audio = AudioFileClip(audio_path)
        audio_duration = audio.duration
        print(colored(f"Audio duration: {audio_duration:.2f}s", "cyan"))
        
        # Use target_duration if provided, otherwise use audio duration plus buffer
        if target_duration:
            print(colored(f"Using provided target duration: {target_duration:.2f}s", "cyan"))
            # IMPORTANT: Slightly reduce the target duration to prevent black fade at end
            video_duration = target_duration - 0.1
            print(colored(f"Adjusted to {video_duration:.2f}s to prevent black end frame", "cyan"))
        else:
            # Add a small buffer to prevent cutting off
            video_duration = audio_duration + 1.9  # 2.0 - 0.1 for end frame fix
            print(colored(f"Calculated video duration: {video_duration:.2f}s (audio + buffer)", "cyan"))
        
        # Handle different background path formats
        if isinstance(background_path, list):
            # Multiple background videos
            print(colored(f"Combining {len(background_path)} background videos", "cyan"))
            background_video_path = combine_videos(background_path, audio_duration, video_duration)
            if not background_video_path:
                raise ValueError("Failed to combine background videos")
            background_video = VideoFileClip(background_video_path)
        else:
            # Single background video
            print(colored(f"Using single background video: {background_path}", "cyan"))
            background_video = VideoFileClip(background_path)
            
            # Resize and crop to vertical format
            background_video = resize_to_vertical(background_video)
            
            # Loop if needed to match audio duration
            if background_video.duration < video_duration:
                print(colored(f"Looping background video ({background_video.duration:.2f}s) to match audio ({video_duration:.2f}s)", "yellow"))
                background_video = vfx.loop(background_video, duration=video_duration)
            
            # Trim if longer than needed
            if background_video.duration > video_duration:
                print(colored(f"Trimming background video from {background_video.duration:.2f}s to {video_duration:.2f}s", "yellow"))
                background_video = background_video.subclip(0, video_duration)
        
        # Set audio
        video_with_audio = background_video.set_audio(audio)
        
        # Add subtitles if provided
        if subtitles_path:
            print(colored("Adding subtitles", "cyan"))
            video_with_subtitles = process_subtitles(subtitles_path, video_with_audio)
        else:
            video_with_subtitles = video_with_audio
        
        # IMPORTANT: Set exact duration to prevent black frames
        # This ensures the video ends on the last frame without fading to black
        if abs(video_with_subtitles.duration - video_duration) > 0.05:
            print(colored(f"Setting exact duration to {video_duration:.2f}s", "yellow"))
            video_with_subtitles = video_with_subtitles.set_duration(video_duration)
        
        # Write final video with specific parameters to prevent black frames
        output_path = "temp_output.mp4"
        print(colored(f"Writing final video to {output_path}", "cyan"))
        
        video_with_subtitles.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp_audio.m4a',
            remove_temp=True,
            fps=30,
            threads=4,
            ffmpeg_params=["-shortest"]  # This helps prevent extra frames
        )
        
        # Clean up
        background_video.close()
        video_with_audio.close()
        video_with_subtitles.close()
        audio.close()
        
        print(colored("✓ Video generation complete", "green"))
        return output_path
        
    except Exception as e:
        print(colored(f"Error generating video: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))
        return None

def generate_tts_audio(sentences: List[str], voice: str = "nova", style: str = "humorous") -> AudioFileClip:
    """Generate text-to-speech audio for the entire script at once"""
    try:
        # Join all sentences into one script, preserving natural pauses
        full_script = ". ".join(sentence.strip().strip('"') for sentence in sentences)
        
        print(colored("\n=== Generating TTS Audio ===", "blue"))
        print(colored(f"Using voice: {voice}", "cyan"))
        print(colored(f"Full script:", "cyan"))
        print(colored(f"{full_script}", "yellow"))
        
        # Single API call for the entire script
        client = OpenAI()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=full_script,
            speed=0.95,  # Slightly slower for better clarity
            response_format="mp3"
        )
        
        print(colored("✓ Generated voice audio", "green"))
        
        # Save the audio file
        audio_path = "temp/tts/tech_humor_latest.mp3"
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        with open(audio_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=4096):
                f.write(chunk)
        
        # Load and process audio
        audio_clip = AudioFileClip(audio_path)
        
        # Add a small fade out at the end
        audio_clip = audio_clip.audio_fadeout(0.5)
        
        # Trim any silence at the end
        duration = audio_clip.duration - 0.1
        audio_clip = audio_clip.subclip(0, duration)
        
        print(colored(f"✓ Generated TTS audio: {audio_clip.duration:.2f}s", "green"))
        
        return audio_clip
        
    except Exception as e:
        print(colored(f"Error generating TTS: {str(e)}", "red"))
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
        # Create a test video clip (dark gray background)
        test_video = ColorClip(size=(1080, 1920), color=(40, 40, 40)).set_duration(3)
        
        # Create test subtitles file with just one subtitle
        test_srt = """1
00:00:00,000 --> 00:00:03,000
Because it turns their 'do not disturb' mode on. 🚫👩‍💻"""

        # Save test subtitles
        test_srt_path = "../temp/test_single_sub.srt"
        with open(test_srt_path, "w", encoding="utf-8-sig") as f:
            f.write(test_srt)

        # Process subtitles
        print(colored("\n=== Testing Single Subtitle ===", "blue"))
        final_video = process_subtitles(test_srt_path, test_video)
        
        # Save test video
        output_path = "../temp/single_subtitle_test.mp4"
        final_video.write_videofile(output_path, fps=30)
        
        print(colored(f"\n✓ Test video saved to: {output_path}", "green"))
        
    except Exception as e:
        print(colored(f"Error in subtitle test: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))

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
    test_single_subtitle()