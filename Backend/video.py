import os
import uuid
import requests
import srt_equalizer
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

def generate_subtitles(sentences: List[str], audio_clips: List[AudioFileClip]) -> str:
    """Generates subtitles locally without external API"""
    def convert_to_srt_time(seconds):
        return str(timedelta(seconds=seconds)).rstrip('0').replace('.', ',')

    subtitles = []
    start_time = 0
    
    for i, (sentence, clip) in enumerate(zip(sentences, audio_clips), start=1):
        end_time = start_time + clip.duration
        subtitle = f"{i}\n{convert_to_srt_time(start_time)} --> {convert_to_srt_time(end_time)}\n{sentence}\n"
        subtitles.append(subtitle)
        start_time = end_time

    subtitles_text = "\n".join(subtitles)
    subtitles_path = f"temp/subtitles/{uuid.uuid4()}.srt"
    os.makedirs("temp/subtitles", exist_ok=True)
    
    with open(subtitles_path, "w", encoding='utf-8') as f:
        f.write(subtitles_text)
    
    # Equalize subtitles for better readability
    srt_equalizer.equalize_srt_file(subtitles_path, subtitles_path, max_chars=10)
    return subtitles_path

def combine_videos(video_paths: List[str], total_duration: float, max_clip_duration: int = 5, threads: int = 2) -> str:
    """Combines multiple videos into one vertical video"""
    clips = []
    total_time = 0
    output_path = f"temp/videos/combined_{uuid.uuid4()}.mp4"
    
    for video_path in video_paths:
        if total_time >= total_duration:
            break
            
        clip = VideoFileClip(video_path)
        needed_duration = min(max_clip_duration, total_duration - total_time)
        
        if clip.duration > needed_duration:
            clip = clip.subclip(0, needed_duration)
            
        # Standardize to vertical format (9:16)
        if (clip.w/clip.h) < 0.5625:  # 9:16 ratio
            clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), 
                       x_center=clip.w/2, y_center=clip.h/2)
        else:
            clip = crop(clip, width=round(0.5625*clip.h), height=clip.h,
                       x_center=clip.w/2, y_center=clip.h/2)
            
        clip = clip.resize((1080, 1920))  # Standard shorts/reels size
        clips.append(clip)
        total_time += clip.duration

    final_clip = concatenate_videoclips(clips).set_fps(30)
    final_clip.write_videofile(output_path, threads=threads)
    return output_path

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

        # Load and process clips
        for path in paths:
            try:
                clip = VideoFileClip(str(path))
                print(colored(f"Processing video: {path}", "blue"))
                
                # Ensure clip is RGB
                if hasattr(clip, 'to_RGB'):
                    clip = clip.to_RGB()
                
                # Get dimensions as integers
                target_width = 1080
                target_height = 1920
                clip_w = int(clip.w)
                clip_h = int(clip.h)
                
                # Calculate aspect ratios
                target_ar = target_width / target_height
                clip_ar = clip_w / clip_h
                
                # Crop and resize
                if clip_ar > target_ar:
                    # Too wide, crop width
                    new_w = int(clip_h * target_ar)
                    x1 = int((clip_w - new_w) / 2)
                    clip = clip.crop(x1=x1, x2=x1+new_w, y1=0, y2=clip_h)
                else:
                    # Too tall, crop height
                    new_h = int(clip_w / target_ar)
                    y1 = int((clip_h - new_h) / 2)
                    clip = clip.crop(x1=0, x2=clip_w, y1=y1, y2=y1+new_h)
                
                # Final resize to target dimensions
                clip = clip.resize((target_width, target_height))
                background_clips.append(clip)
                
            except Exception as e:
                print(colored(f"Warning: Failed to process clip {path}: {str(e)}", "yellow"))
                continue

        if not background_clips:
            raise ValueError("No valid clips after processing")

        # Load audio
        audio = AudioFileClip(audio_path)
        
        # Concatenate and loop background if needed
        final_background = concatenate_videoclips(background_clips)
        if audio.duration > final_background.duration:
            n_loops = int(np.ceil(audio.duration / final_background.duration))
            final_background = concatenate_videoclips([final_background] * n_loops)
        final_background = final_background.subclip(0, audio.duration)
        
        # Create final video with audio
        video = final_background.set_audio(audio)

        # Add subtitles if provided
        if subtitles_path and os.path.exists(subtitles_path):
            # Define styles with RGB tuples for colors
            styles = {
                'tech_humor': {
                    'font': 'Arial-Bold',  # Use more common font
                    'fontsize': 85,
                    'color': (255, 255, 255),  # white as RGB
                    'stroke_color': (46, 204, 113),  # green as RGB
                    'stroke_width': 3,
                    'position': ('center', 960),
                    'method': 'caption',
                    'size': (800, None),
                    'bg_color': (0, 0, 0),  # black as RGB
                    'bg_opacity': 0.5,
                    'line_spacing': 5
                },
                'ai_money': {
                    'font': 'Arial-Bold',
                    'fontsize': 80,
                    'color': (255, 215, 0),  # gold as RGB
                    'stroke_color': (0, 0, 0),  # black as RGB
                    'stroke_width': 3,
                    'position': ('center', 960),
                    'method': 'caption',
                    'size': (850, None),
                    'bg_color': (0, 0, 0),
                    'bg_opacity': 0.6,
                    'line_spacing': 5
                },
                'default': {
                    'font': 'Arial-Bold',
                    'fontsize': 80,
                    'color': (255, 255, 255),  # white as RGB
                    'stroke_color': (0, 0, 0),  # black as RGB
                    'stroke_width': 3,
                    'position': ('center', 960),
                    'method': 'caption',
                    'size': (800, None),
                    'bg_color': (0, 0, 0),  # black as RGB
                    'bg_opacity': 0.4,
                    'line_spacing': 5
                }
            }
            
            # Get style based on content type
            style = styles.get(content_type, styles['default'])
            
            def wrap_text(text, max_chars=25):
                """Wrap text to prevent overflow"""
                words = text.split()
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    word_length = len(word)
                    if current_length + word_length + 1 <= max_chars:
                        current_line.append(word)
                        current_length += word_length + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = word_length
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                return '\n'.join(lines)
            
            # Create background for subtitles
            def create_subtitle_bg(txt):
                """Create clean, simple subtitles with good readability"""
                try:
                    # Clean and wrap text
                    txt = txt.strip()
                    
                    # Create text clip with clean styling
                    txt_clip = TextClip(
                        txt=txt,
                        font='Arial-Bold',
                        fontsize=70,
                        color='white',
                        stroke_color='black',
                        stroke_width=2,
                        method='label',
                        size=(900, None),  # Limit width, auto-height
                        align='center'
                    )
                    
                    # Simple semi-transparent background
                    bg_width = txt_clip.w + 40   # 20px padding each side
                    bg_height = txt_clip.h + 30  # 15px padding top and bottom
                    
                    bg_clip = ColorClip(
                        size=(bg_width, bg_height),
                        color=(0, 0, 0)
                    ).set_opacity(0.6)
                    
                    # Center text on background
                    txt_clip = txt_clip.set_position(('center', 'center'))
                    
                    # Combine text and background
                    composite = CompositeVideoClip(
                        [bg_clip, txt_clip],
                        size=(bg_width, bg_height)
                    ).set_duration(2)
                    
                    return composite
                    
                except Exception as e:
                    print(colored(f"Subtitle creation error: {str(e)}", "red"))
                    # Ultra simple fallback
                    return TextClip(
                        txt=txt,
                        font='Arial-Bold',
                        fontsize=70,
                        color='white',
                        stroke_color='black',
                        stroke_width=2,
                        method='label'
                    ).set_duration(2)

            # Create subtitles with background
            try:
                subtitles = SubtitlesClip(subtitles_path, create_subtitle_bg)
                subtitles = subtitles.set_position(style['position'])
                
                # Add fade effects
                subtitles = subtitles.crossfadein(0.3).crossfadeout(0.3)
                
                # Composite with video using explicit size
                video = CompositeVideoClip(
                    [video, subtitles],
                    size=(1080, 1920)  # Ensure final size is explicit
                )
            except Exception as e:
                print(colored(f"Warning: Failed to add subtitles: {str(e)}", "yellow"))
                # Continue without subtitles if they fail
                pass

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
        for clip in background_clips:
            clip.close()
        
        return output_path

    except Exception as e:
        print(colored(f"Error generating video: {str(e)}", "red"))
        return None
