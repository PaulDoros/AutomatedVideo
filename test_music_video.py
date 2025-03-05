import os
import sys
import uuid
from termcolor import colored
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip, ColorClip
import random
import numpy as np

# Add the Backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Backend'))

# Import functions from video.py
from video import mix_audio, log_info, log_success, log_error

def create_test_video_with_music():
    """
    Create a test video with clearly audible background music.
    This function will:
    1. Create a simple color background
    2. Find a voice audio file or create one
    3. Find a music file
    4. Mix them together with HIGH music volume
    5. Generate a video
    """
    try:
        print(colored("=== Creating Test Video With Music ===", "blue", attrs=["bold"]))
        
        # Create output directory
        os.makedirs("output/test", exist_ok=True)
        
        # Step 1: Create a simple color background
        log_info("Creating a simple color background")
        # Create a vertical video (9:16 aspect ratio)
        width, height = 1080, 1920
        # Create a gradient background (blue to purple)
        color_clip = ColorClip(size=(width, height), color=[0, 0, 255])
        
        # Step 2: Find a voice audio file from temp directory or create one
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        voice_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                      if f.endswith('.mp3') and os.path.isfile(os.path.join(temp_dir, f))]
        
        voice_path = None
        if voice_files:
            voice_path = random.choice(voice_files)
            log_info(f"Using existing voice file: {os.path.basename(voice_path)}")
        else:
            # If no voice files found, create a simple one with text-to-speech
            try:
                from gtts import gTTS
                
                # Create a simple TTS file
                tts = gTTS("This is a test video with background music. The music should be clearly audible. Let's check if our audio mixing is working properly.", lang='en')
                voice_path = os.path.join(temp_dir, f"test_voice_{uuid.uuid4()}.mp3")
                tts.save(voice_path)
                log_info(f"Created new voice file: {os.path.basename(voice_path)}")
            except Exception as e:
                log_error(f"Error creating TTS: {str(e)}")
                return None
        
        # Step 3: Find a music file
        music_dir = "assets/music"
        music_files = []
        
        if os.path.exists(music_dir):
            # Look in all subdirectories for music files
            for root, dirs, files in os.walk(music_dir):
                for file in files:
                    if file.endswith('.mp3') and os.path.isfile(os.path.join(root, file)):
                        music_files.append(os.path.join(root, file))
        
        # If no music files found in assets/music, check temp directory
        if not music_files:
            log_warning("No music files found in assets/music, checking temp directory")
            for file in os.listdir(temp_dir):
                if file.endswith('.mp3') and "music" in file.lower() and os.path.isfile(os.path.join(temp_dir, file)):
                    music_files.append(os.path.join(temp_dir, file))
        
        # If still no music files, download a sample
        if not music_files:
            log_warning("No music files found, downloading a sample")
            try:
                import requests
                # URL to a royalty-free music sample
                music_url = "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0c6ff1bbd.mp3"
                music_path = os.path.join(temp_dir, "sample_music.mp3")
                
                response = requests.get(music_url)
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                
                music_files.append(music_path)
                log_info(f"Downloaded sample music: {os.path.basename(music_path)}")
            except Exception as e:
                log_error(f"Error downloading sample music: {str(e)}")
                return None
        
        music_path = random.choice(music_files)
        log_info(f"Using music file: {os.path.basename(music_path)}")
        
        # Step 4: Mix audio with HIGH music volume
        music_volume = 0.5  # 50% volume - much higher than default
        mixed_audio_path = os.path.join(temp_dir, f"mixed_audio_{uuid.uuid4()}.mp3")
        
        log_info(f"Mixing audio with music volume: {music_volume}")
        mixed_audio_path = mix_audio(
            voice_path=voice_path,
            music_path=music_path,
            output_path=mixed_audio_path,
            music_volume=music_volume,
            fade_in=2.0,
            fade_out=3.0
        )
        
        if not mixed_audio_path or not os.path.exists(mixed_audio_path):
            log_error("Failed to mix audio")
            return None
            
        log_success(f"Successfully mixed audio: {os.path.basename(mixed_audio_path)}")
        
        # Step 5: Generate video
        output_path = os.path.join("output/test", f"test_music_video_{uuid.uuid4()}.mp4")
        
        # Load the mixed audio to get its duration
        mixed_audio = AudioFileClip(mixed_audio_path)
        audio_duration = mixed_audio.duration
        
        # Set the duration of the color clip
        color_clip = color_clip.set_duration(audio_duration)
        
        # Add the mixed audio to the color clip
        final_video = color_clip.set_audio(mixed_audio)
        
        # Write the video file
        log_info(f"Writing video to: {output_path}")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"temp/temp_audio_{uuid.uuid4()}.m4a",
            remove_temp=True,
            fps=30
        )
        
        # Clean up
        mixed_audio.close()
        final_video.close()
        
        log_success(f"Successfully created test video: {output_path}")
        return output_path
        
    except Exception as e:
        log_error(f"Error creating test video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def log_warning(message, emoji="⚠️"):
    print(f"{emoji} {message}")

if __name__ == "__main__":
    video_path = create_test_video_with_music()
    if video_path:
        print(colored(f"\nTest video created successfully: {video_path}", "green", attrs=["bold"]))
        print(colored("Please check this video to verify the background music is audible.", "green"))
    else:
        print(colored("\nFailed to create test video", "red", attrs=["bold"])) 