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
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime
import html
import traceback
import pysrt
import emoji
from moviepy.video.VideoClip import ImageClip
from emoji_data_python import emoji_data
from io import BytesIO
import re

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
    """Combine multiple videos together to match the audio duration.
    Each video will contribute a portion to the final video.
    """
    try:
        clips = []
        
        # Get audio clip to analyze timing
        audio_clip = AudioFileClip("temp/tts/tech_humor_latest.mp3")
        total_duration = audio_clip.duration
        
        print(colored("\n=== Video Combination Debug Info ===", "blue"))
        print(colored(f"Audio duration: {total_duration:.2f}s", "cyan"))
        print(colored(f"Target duration: {target_duration:.2f}s", "cyan"))
        print(colored(f"Number of videos: {len(video_paths)}", "cyan"))
        
        # Calculate how long each video should be
        num_videos = len(video_paths)
        segment_duration = total_duration / num_videos
        print(colored(f"Segment duration per video: {segment_duration:.2f}s", "cyan"))
        
        # Keep track of current position in the timeline
        current_time = 0
        
        for i, video_data in enumerate(video_paths):
            try:
                print(colored(f"\n=== Processing Video {i+1}/{num_videos} ===", "blue"))
                print(colored(f"Video data: {video_data}", "cyan"))
                
                # Get path
                video_path = video_data.get('path', video_data)
                
                print(colored(f"Video path: {video_path}", "cyan"))
                
                # Load video
                video = VideoFileClip(video_path)
                print(colored(f"Video duration: {video.duration:.2f}s", "cyan"))
                
                # Calculate how much of this video to use
                this_segment_duration = min(segment_duration, total_duration - current_time)
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
                
                # Choose a random starting point within the usable portion
                # But ensure we have enough video left from that point
                latest_possible_start = usable_end - this_segment_duration
                if latest_possible_start < usable_start:
                    video_start = usable_start
                else:
                    video_start = random.uniform(usable_start, latest_possible_start)
                
                print(colored(f"Using segment: {video_start:.2f}s to {video_start + this_segment_duration:.2f}s", "cyan"))
                
                # Create clip
                print(colored("\nCreating clip...", "blue"))
                clip = (video
                       .subclip(video_start, video_start + this_segment_duration)
                       .resize(width=1080)
                       .set_position(("center", "center")))
                
                # Handle transition effects
                if i > 0:
                    print(colored("Adding crossfade...", "cyan"))
                    clip = clip.crossfadein(0.5)
                
                if i < num_videos - 1:
                    print(colored("Adding crossfade out...", "cyan"))
                    clip = clip.crossfadeout(0.5)
                
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
        
        # Convert emoji to unicode code points
        emoji_code = "-".join(
            format(ord(c), 'x').lower()
            for c in emoji_char
            if c != '\ufe0f'  # Skip variation selectors
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
                print(colored(f"Failed to download emoji: {emoji_char} (HTTP {response.status_code})", "yellow"))
                return create_fallback_emoji(size, emoji_char)
        except Exception as e:
            print(colored(f"Error downloading emoji: {str(e)}", "red"))
            return create_fallback_emoji(size, emoji_char)
            
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

def process_subtitles(subs_path, base_video):
    """Process subtitles and add them to video with efficient rendering"""
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
        
        # Pre-render all subtitle images first (for speed)
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
            
            # Create text image
            text_image = create_text_with_emoji(text)
            if text_image is None:
                continue
                
            # Store the pre-rendered image
            subtitle_data.append({
                'text': text,
                'image': text_image,
                'start': start_time,
                'end': end_time
            })
            
            print(colored(f"Prepared subtitle: {text}", "green"))
        
        # Fast track: If no subtitles, return base video
        if not subtitle_data:
            return base_video
        
        # Optimize for speed: Generate one composite clip per subtitle
        clips = [base_video]
        
        for data in subtitle_data:
            # Convert PIL image to MoviePy clip
            img_array = np.array(data['image'])
            duration = data['end'] - data['start']
            
            # Create clip with minimal processing
            sub_clip = (ImageClip(img_array)
                .set_duration(duration)
                .set_position(('center', 'center'))
                .set_start(data['start'])
            )
            
            # Simple fade effect
            if duration > 0.5:
                sub_clip = sub_clip.fadein(0.15).fadeout(0.15)
            
            clips.append(sub_clip)
            
        # Use efficient compositing
        print(colored("✓ Rendering video with subtitles", "green"))
        final_video = CompositeVideoClip(clips)
        print(colored("✓ Subtitles added successfully", "green"))
        
        return final_video
    
    except Exception as e:
        print(colored(f"Error processing subtitles: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))
        return base_video

def generate_video(background_path, audio_path, subtitles_path=None, content_type=None):
    """Generate final video with audio and optional subtitles"""
    try:
        # Load background clips
        background_clips = []
        if isinstance(background_path, (str, Path)):
            paths = [background_path]
        elif isinstance(background_path, (list, tuple)):
            paths = background_path
        else:
            raise ValueError("Invalid background_path type")

        # Load audio and get base duration
        audio = AudioFileClip(audio_path)
        base_duration = audio.duration - 0.1  # Trim slight silence
        
        # Create extended audio with silence for the extra duration
        if subtitles_path and os.path.exists(subtitles_path):
            # Create silence clip for extension
            silence = AudioClip(lambda t: 0, duration=2)
            extended_audio = concatenate_audioclips([audio, silence])
            total_duration = base_duration + 2
        else:
            extended_audio = audio
            total_duration = base_duration

        # Process videos with the total duration
        num_videos = len(paths)
        segment_duration = base_duration / num_videos

        print(colored(f"\n=== Video Generation Info ===", "blue"))
        print(colored(f"Total duration: {total_duration:.2f}s", "cyan"))
        print(colored(f"Using {num_videos} videos", "cyan"))
        print(colored(f"Segment duration: {segment_duration:.2f}s", "cyan"))

        # Process each video into a segment
        current_time = 0
        for i, path in enumerate(paths):
            try:
                print(colored(f"\nProcessing video {i+1}/{num_videos}: {path}", "blue"))
                clip = VideoFileClip(str(path))
                
                # Calculate this segment's duration
                if i == num_videos - 1:
                    this_segment_duration = total_duration - current_time
                else:
                    this_segment_duration = segment_duration + 0.5  # Add overlap for transition
                
                print(colored(f"Video duration: {clip.duration:.2f}s", "cyan"))
                print(colored(f"Segment duration: {this_segment_duration:.2f}s", "cyan"))
                
                # Select portion of video
                if clip.duration > this_segment_duration:
                    max_start = clip.duration - this_segment_duration
                    start_time = random.uniform(0, max_start)
                    clip = clip.subclip(start_time, start_time + this_segment_duration)
                    print(colored(f"Using section {start_time:.2f}s to {start_time + this_segment_duration:.2f}s", "cyan"))
                else:
                    clip = clip.loop(duration=this_segment_duration)
                    print(colored("Looping video to match duration", "yellow"))
                
                # Resize to vertical format
                clip = resize_to_vertical(clip)
                
                # Add smooth transitions
                if i > 0:
                    # Fade in while previous clip is still playing
                    clip = clip.set_start(current_time - 0.5)  # Start 0.5s before current time
                    clip = clip.crossfadein(0.5)  # Longer, smoother crossfade
                
                if i < num_videos - 1:
                    clip = clip.crossfadeout(0.5)  # Longer fadeout to match fadein
                
                background_clips.append(clip)
                current_time += segment_duration  # Use original segment duration for timing
                
            except Exception as e:
                print(colored(f"Error processing video {i+1}: {str(e)}", "red"))
                # Instead of black clip, duplicate previous clip or use next clip
                if background_clips:
                    fallback_clip = background_clips[-1].copy()
                elif i < len(paths) - 1:
                    # Try to use next clip if this is not the last one
                    try:
                        fallback_clip = VideoFileClip(str(paths[i+1]))
                    except:
                        fallback_clip = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=this_segment_duration)
                else:
                    fallback_clip = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=this_segment_duration)
                
                fallback_clip = fallback_clip.set_duration(this_segment_duration)
                background_clips.append(fallback_clip)
                current_time += segment_duration

        # Combine all clips
        print(colored("\nCombining video segments...", "blue"))
        final_background = CompositeVideoClip(background_clips)
        final_background = final_background.set_duration(total_duration)
        
        # Create final video with extended audio
        video = final_background.set_audio(extended_audio)

        # Add subtitles if provided
        if subtitles_path and os.path.exists(subtitles_path):
            print(colored("\n=== Processing Subtitles ===", "blue"))
            video = process_subtitles(subtitles_path, video)

        # Write final video
        output_path = "temp/final_video.mp4"
        video.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            preset='medium'
        )
        
        # Clean up
        video.close()
        final_background.close()
        audio.close()
        extended_audio.close()
        for clip in background_clips:
            clip.close()
        
        return output_path

    except Exception as e:
        print(colored(f"Error generating video: {str(e)}", "red"))
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
                current_times = [float(sum(float(x) * 60 ** i for i, x in enumerate(reversed(t.split(':')))))\
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

if __name__ == "__main__":
    test_single_subtitle()