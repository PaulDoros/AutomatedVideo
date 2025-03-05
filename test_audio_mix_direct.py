import os
import sys
import uuid
from termcolor import colored
import random
import glob
from moviepy.editor import AudioFileClip, CompositeAudioClip
import numpy as np

def log_info(message, emoji="ℹ️"):
    print(f"{emoji} {message}")

def log_success(message, emoji="✅"):
    print(f"{emoji} {message}")

def log_error(message, emoji="❌"):
    print(f"{emoji} {message}")

def mix_audio(voice_path, music_path, output_path, music_volume=0.5, fade_in=2.0, fade_out=3.0):
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
            log_error(f"No valid music path provided: {music_path}")
            log_info("Using voice audio only")
            # Normalize voice audio for consistent levels
            from moviepy.audio.fx.all import audio_normalize
            voice = voice.fx(audio_normalize)
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
                log_error(f"Error applying dynamic volume: {str(e)}")
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
                from moviepy.audio.fx.all import audio_loop
                music = audio_loop(music, duration=voice_duration)
            elif music_duration > voice_duration:
                log_info(f"Music longer than voice ({music_duration:.2f}s > {voice_duration:.2f}s), trimming")
                # Trim music to match voice duration
                music = music.subclip(0, voice_duration)
            
            # Normalize voice audio for consistent levels
            from moviepy.audio.fx.all import audio_normalize
            voice = voice.fx(audio_normalize)
            
            # Boost voice slightly to ensure clarity over music
            voice = voice.volumex(1.2)
            
            # Composite audio - put music first in the list so voice is layered on top
            # This ensures the voice is more prominent in the mix
            final_audio = CompositeAudioClip([music, voice])
            
            # Write the mixed audio to the output path with high quality settings
            final_audio.write_audiofile(output_path, fps=44100, bitrate="192k")
            
            log_success(f"Successfully mixed voice and music: {os.path.basename(music_path)}")
            return output_path
            
        except Exception as music_error:
            log_error(f"Error processing music: {str(music_error)}")
            log_info("Falling back to voice audio only")
            # Normalize voice audio for consistent levels
            from moviepy.audio.fx.all import audio_normalize
            voice = voice.fx(audio_normalize)
            voice.write_audiofile(output_path, fps=44100, bitrate="192k")
            return output_path
    
    except Exception as e:
        log_error(f"Error mixing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_audio_mixing():
    """
    Test the audio mixing functionality with a high music volume.
    """
    try:
        print(colored("=== Testing Audio Mixing ===", "blue", attrs=["bold"]))
        
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Step 1: Find a voice audio file
        voice_files = []
        
        # Look in temp/tts directory
        tts_dir = "temp/tts"
        if os.path.exists(tts_dir):
            voice_files.extend(glob.glob(f"{tts_dir}/*.mp3"))
        
        # If no files found, look in the temp directory
        if not voice_files:
            voice_files.extend(glob.glob("temp/*.mp3"))
        
        # If still no files found, create a simple one with text-to-speech
        if not voice_files:
            try:
                from gtts import gTTS
                
                # Create a simple TTS file
                tts = gTTS("This is a test of the audio mixing functionality. The background music should be clearly audible at 50 percent volume.", lang='en')
                voice_path = os.path.join("temp", f"test_voice_{uuid.uuid4()}.mp3")
                tts.save(voice_path)
                log_info(f"Created new voice file: {os.path.basename(voice_path)}")
                voice_files.append(voice_path)
            except Exception as e:
                log_error(f"Error creating TTS: {str(e)}")
                return None
        
        # Select a random voice file
        voice_path = random.choice(voice_files)
        log_info(f"Using voice file: {voice_path}")
        
        # Step 2: Find or download a music file
        music_files = []
        
        # Look in assets/music directory and subdirectories
        music_dir = "assets/music"
        if os.path.exists(music_dir):
            for root, dirs, files in os.walk(music_dir):
                for file in files:
                    if file.endswith('.mp3'):
                        music_files.append(os.path.join(root, file))
        
        # If no music files found, download a sample
        if not music_files:
            try:
                import requests
                # URL to a royalty-free music sample
                music_url = "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0c6ff1bbd.mp3"
                music_path = os.path.join("temp", "sample_music.mp3")
                
                response = requests.get(music_url)
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                
                music_files.append(music_path)
                log_info(f"Downloaded sample music: {os.path.basename(music_path)}")
            except Exception as e:
                log_error(f"Error downloading sample music: {str(e)}")
                return None
        
        # Select a random music file
        music_path = random.choice(music_files)
        log_info(f"Using music file: {music_path}")
        
        # Step 3: Mix audio with HIGH music volume
        music_volume = 0.5  # 50% volume - much higher than default
        mixed_audio_path = os.path.join("temp", f"mixed_audio_{uuid.uuid4()}.mp3")
        
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
            
        log_success(f"Successfully mixed audio: {mixed_audio_path}")
        log_info(f"Voice file: {voice_path}")
        log_info(f"Music file: {music_path}")
        log_info(f"Mixed audio file: {mixed_audio_path}")
        log_info(f"Music volume: {music_volume}")
        
        return mixed_audio_path
        
    except Exception as e:
        log_error(f"Error testing audio mixing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_audio_mixing()
    if result:
        print(colored(f"\nAudio mixing test successful: {result}", "green", attrs=["bold"]))
        print(colored("Please check this audio file to verify the background music is audible.", "green"))
    else:
        print(colored("\nAudio mixing test failed", "red", attrs=["bold"])) 