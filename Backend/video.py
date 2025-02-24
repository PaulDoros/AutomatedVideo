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

def generate_video(background_path, audio_path, subtitles_path=None):
    """Generate final video with audio and optional subtitles"""
    try:
        # Handle both single path and list of paths
        background_paths = background_path if isinstance(background_path, list) else [background_path]
        
        # Validate and check video files
        valid_paths = []
        for path in background_paths:
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                print(colored(f"Warning: Video not found: {path}", "yellow"))
                continue
            if not path.endswith(('.mp4', '.mov')):
                print(colored(f"Warning: Invalid video format: {path}", "yellow"))
                continue
            valid_paths.append(abs_path)
        
        if not valid_paths:
            raise ValueError("No valid background videos found")
            
        print(colored(f"Found {len(valid_paths)} valid background videos", "green"))
        
        # Load the audio first to get target duration
        audio = AudioFileClip(audio_path)
        target_duration = audio.duration
        
        # Load and prepare background clips
        background_clips = []
        current_duration = 0
        
        while current_duration < target_duration:
            for path in valid_paths:
                if current_duration >= target_duration:
                    break
                    
                try:
                    clip = VideoFileClip(path)
                    print(colored(f"Loaded video: {path}", "green"))
                    
                    remaining_duration = target_duration - current_duration
                    if clip.duration > remaining_duration:
                        clip = clip.subclip(0, remaining_duration)
                    
                    background_clips.append(clip)
                    current_duration += clip.duration
                    
                    if current_duration >= target_duration:
                        break
                except Exception as e:
                    print(colored(f"Error loading video {path}: {str(e)}", "red"))
                    continue
        
        if not background_clips:
            raise ValueError("Could not load any background clips")
        
        # Concatenate all background clips
        final_background = concatenate_videoclips(background_clips)
        
        # Set the audio
        video = final_background.set_audio(audio)
        
        # Add subtitles if provided
        if subtitles_path and os.path.exists(subtitles_path):
            generator = lambda txt: TextClip(
                txt, 
                font='Arial', 
                fontsize=70, 
                color='white',
                stroke_color='black',
                stroke_width=2
            )
            
            subtitles = SubtitlesClip(subtitles_path, generator)
            video = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])
        
        # Write the final video
        output_path = "temp/final_video.mp4"
        video.write_videofile(
            output_path,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            threads=4
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
