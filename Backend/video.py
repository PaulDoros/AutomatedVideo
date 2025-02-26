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
from PIL import Image, ImageFont
from datetime import datetime
import html
import traceback
import pysrt

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

def generate_subtitles(script: str, audio_path: str, content_type: str = None) -> str:
    """Generate SRT subtitles from script"""
    try:
        # Get audio duration and trim end silence
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration - 0.15  # Trim more silence
        audio.close()
        
        # Calculate timing with better spacing
        lines = [line.strip() for line in script.split('\n') 
                if line.strip() and not line[0].isdigit()]
        
        avg_duration = total_duration / len(lines)
        current_time = 0
        srt_blocks = []
        
        for i, line in enumerate(lines, 1):
            # Calculate duration based on content
            word_count = len(line.split())
            is_last = (i == len(lines))
            
            if is_last:
                duration = avg_duration + 2.5
            else:
                # Start subtitles slightly earlier
                duration = min(max(word_count * 0.4, 2.5), avg_duration * 1.2)
            
            # Start each subtitle 0.2s earlier for better sync
            start_time = max(0, current_time - 0.2)
            end_time = start_time + duration
            
            # Format for SRT
            start_str = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}".replace('.', ',')
            end_str = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}".replace('.', ',')
            
            srt_block = f"{i}\n{start_str} --> {end_str}\n{line}\n\n"
            srt_blocks.append(srt_block)
            
            current_time = end_time + 0.1
        
        # Write SRT file
        subtitles_path = "temp/subtitles/latest.srt"
        os.makedirs(os.path.dirname(subtitles_path), exist_ok=True)
        
        # Write with UTF-8 encoding and BOM
        with open(subtitles_path, 'w', encoding='utf-8-sig') as f:
            f.write(''.join(srt_blocks))
        
        return subtitles_path
        
    except Exception as e:
        print(colored(f"Error generating subtitles: {str(e)}", "red"))
        return None

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

def create_subtitle_bg(txt, style=None, is_last=False, total_duration=None):
    """Create dynamic centered subtitles with emoji support"""
    try:
        if not txt:
            return None
            
        # Clean text while preserving emojis
        clean_txt = txt if isinstance(txt, str) else str(txt)
        clean_txt = clean_txt.strip().strip('"')
        if not clean_txt:
            return None
        
        # Set duration
        if is_last:
            duration = total_duration + 1.8
            fade_out = 0.6
        else:
            duration = 3.2
            fade_out = 0.35
        
        # Create text clip with emoji support
        txt_clip = TextClip(
            txt=clean_txt,
            font='Segoe-UI-Emoji',  # Font with emoji support
            fontsize=110,
            color='white',
            stroke_color='black',
            stroke_width=6,
            method='caption',  # Better for emoji support
            size=(900, None),
            align='center',
            bg_color='transparent'
        ).set_position('center')
        
        # Create background
        w, h = txt_clip.size
        bg_w, bg_h = w + 80, h + 50
        
        # Create semi-transparent background
        bg_clip = ColorClip(
            size=(bg_w, bg_h),
            color=(0, 0, 0)
        ).set_opacity(0.6)
        
        # Combine text and background
        txt_with_bg = CompositeVideoClip(
            [bg_clip, txt_clip],
            size=(bg_w, bg_h)
        )
        
        # Add effects
        final_clip = (txt_with_bg
            .set_duration(duration)
            .set_position(('center', 'center'))
            .crossfadein(0.2))
        
        if not is_last:
            final_clip = final_clip.crossfadeout(fade_out)
        else:
            final_clip = final_clip.crossfadeout(0.6)
        
        return final_clip
        
    except Exception as e:
        print(colored(f"Subtitle error: {str(e)}", "red"))
        print(colored(f"Text content: {txt}", "yellow"))
        return None

def process_subtitles(subtitles_path, video):
    """Process subtitles using pysrt"""
    try:
        print(colored("\n=== Processing Subtitles ===", "blue"))
        
        # Read subtitles with pysrt
        subs = pysrt.open(subtitles_path, encoding='utf-8-sig')
        subtitle_clips = []
        
        print(colored(f"Found {len(subs)} subtitles", "cyan"))
        
        for i, sub in enumerate(subs):
            try:
                # Convert timecode to seconds
                start_time = sub.start.seconds + sub.start.milliseconds/1000.0
                end_time = sub.end.seconds + sub.end.milliseconds/1000.0
                duration = end_time - start_time
                
                # Get text content directly
                text = sub.text.strip()
                
                clip = create_subtitle_bg(
                    txt=text,  # Pass text directly
                    is_last=(i == len(subs) - 1),
                    total_duration=duration
                )
                
                if clip:
                    clip = clip.set_start(start_time)
                    subtitle_clips.append(clip)
                    print(colored(f"Created subtitle: {text}", "cyan"))
                    
            except Exception as e:
                print(colored(f"Error processing subtitle {i+1}: {str(e)}", "red"))
                print(colored(traceback.format_exc(), "red"))
        
        if subtitle_clips:
            # Combine all subtitle clips
            subtitles = CompositeVideoClip(
                subtitle_clips,
                size=video.size
            ).set_duration(video.duration)
            
            # Add subtitles to video
            final_video = CompositeVideoClip([
                video,
                subtitles
            ])
            
            print(colored("✓ Subtitles added successfully", "green"))
            return final_video
        
        return video
        
    except Exception as e:
        print(colored(f"Error processing subtitles: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))
        return video

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
                current_text += line + ' '
                
        # Add the last subtitle if exists
        if current_times and current_text:
            times_texts.append((current_times, current_text.strip()))
            
        return times_texts
        
    except Exception as e:
        print(colored(f"Error reading subtitles: {str(e)}", "red"))
        return []

def test_subtitles():
    """Test function for subtitle styling"""
    try:
        # Create a test video clip (black background)
        test_video = ColorClip(size=(1080, 1920), color=(0, 0, 0)).set_duration(15)
        
        # Create test subtitles file
        test_srt = """1
00:00:00,000 --> 00:00:03,000
Why do programmers love coffee so much? ☕️

2
00:00:03,000 --> 00:00:06,000
Because it turns their 'do not disturb' mode on. 🚫👩‍💻

3
00:00:06,000 --> 00:00:09,000
But without it, they hit a real bug...

4
00:00:09,000 --> 00:00:12,000
Sleep deprivation exception caught! 😴

5
00:00:12,000 --> 00:00:15,000
Smash like for caffeine boosts! ⚡"""

        # Save test subtitles
        test_srt_path = "../temp/test_subs.srt"
        with open(test_srt_path, "w", encoding="utf-8") as f:
            f.write(test_srt)

        # Process subtitles
        print(colored("\n=== Testing Subtitle Styling ===", "blue"))
        final_video = process_subtitles(test_srt_path, test_video)
        
        # Save test video
        output_path = "../temp/subtitle_test.mp4"
        final_video.write_videofile(output_path, fps=30)
        
        print(colored(f"\n✓ Test video saved to: {output_path}", "green"))
        
    except Exception as e:
        print(colored(f"Error in subtitle test: {str(e)}", "red"))
        print(colored(traceback.format_exc(), "red"))

# Run the test if this file is run directly
if __name__ == "__main__":
    test_subtitles()
