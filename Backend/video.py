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
from moviepy.config import change_settings

# Configure MoviePy to use ImageMagick
if os.getenv('IMAGEMAGICK_BINARY'):
    change_settings({"IMAGEMAGICK_BINARY": os.getenv('IMAGEMAGICK_BINARY')})

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
        print(colored(f"Video paths type: {type(video_paths)}", "cyan"))
        print(colored(f"Video paths content: {video_paths}", "cyan"))
        
        # Calculate segment duration for each video
        num_videos = len(video_paths)
        segment_duration = total_duration / num_videos
        
        print(colored("\n=== Segment Calculations ===", "blue"))
        print(colored(f"Segment duration per video: {segment_duration:.2f}s", "cyan"))
        
        # Track our position in the timeline
        current_time = 0
        
        for i, video_data in enumerate(video_paths):
            try:
                print(colored(f"\n=== Processing Video {i+1}/{num_videos} ===", "blue"))
                print(colored(f"Video data: {video_data}", "cyan"))
                
                # Extract path from video_data dictionary
                video_path = video_data['path'] if isinstance(video_data, dict) else video_data
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
                else:
                    video_start = 0
                    print(colored(f"Video shorter than needed - will loop", "yellow"))
                    print(colored(f"- Original duration: {video.duration:.2f}s", "yellow"))
                    print(colored(f"- Need duration: {this_segment_duration:.2f}s", "yellow"))
                    video = vfx.loop(video, duration=this_segment_duration)
                
                # Create clip
                print(colored("\nCreating clip...", "blue"))
                clip = (video
                       .subclip(video_start, video_start + this_segment_duration)
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
        final_clip = concatenate_videoclips(clips, method="compose")
        
        print(colored(f"Final clip duration: {final_clip.duration:.2f}s", "cyan"))
        print(colored(f"Target duration: {total_duration:.2f}s", "cyan"))
        
        if abs(final_clip.duration - total_duration) > 0.1:
            print(colored(f"Duration mismatch of {abs(final_clip.duration - total_duration):.2f}s - adjusting...", "yellow"))
            final_clip = final_clip.set_duration(total_duration)
        
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

def create_subtitle_bg(txt):
    """Create clean, reliable subtitles with ImageMagick"""
    try:
        # Clean and wrap text
        txt = txt.strip()
        
        # Create text clip with ImageMagick
        txt_clip = TextClip(
            txt=txt,
            font='Arial-Bold',
            fontsize=70,
            color='white',
            stroke_color='black',
            stroke_width=2,
            method='caption',  # Use caption method for better rendering
            size=(900, None),
            align='center',
            kerning=2
        )
        
        # Create background with proper padding
        pad_x = 40
        pad_y = 30
        bg_width = txt_clip.w + pad_x
        bg_height = txt_clip.h + pad_y
        
        # Create solid background
        bg_clip = ColorClip(
            size=(bg_width, bg_height),
            color=(0, 0, 0)
        ).set_opacity(0.7)
        
        # Center text on background
        txt_clip = txt_clip.set_position(('center', 'center'))
        
        # Combine with crossfade
        composite = CompositeVideoClip(
            [bg_clip, txt_clip],
            size=(bg_width, bg_height)
        )
        
        # Add fade effects
        final_clip = (composite
            .set_duration(2)
            .fadein(0.25)
            .fadeout(0.25))
        
        return final_clip
        
    except Exception as e:
        print(colored(f"Subtitle creation error: {str(e)}", "red"))
        # Simple fallback
        return TextClip(
            txt=txt,
            font='Arial-Bold',
            fontsize=70,
            color='white',
            method='caption',
            size=(900, None)
        ).set_duration(2)

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

        # Load audio to get total duration
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        # Calculate segment duration
        num_videos = len(paths)
        segment_duration = total_duration / num_videos
        
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

        # Combine all clips with composite method
        print(colored("\nCombining video segments...", "blue"))
        final_background = CompositeVideoClip(background_clips)
        final_background = final_background.set_duration(total_duration)
        print(colored(f"Final duration: {final_background.duration:.2f}s", "cyan"))
        
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
            subtitles = SubtitlesClip(subtitles_path, create_subtitle_bg)
            subtitles = subtitles.set_position(style['position'])
            
            # Composite with video using explicit size
            video = CompositeVideoClip(
                [video, subtitles],
                size=(1080, 1920)  # Ensure final size is explicit
            )

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
