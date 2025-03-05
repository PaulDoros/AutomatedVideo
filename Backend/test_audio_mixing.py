import os
import sys
import uuid
from termcolor import colored
from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_audioclips
import numpy as np
import traceback
import shutil

def mix_audio(voice_path, music_path, output_path, music_volume=0.3, fade_in=2.0, fade_out=3.0):
    """Mix voice audio with background music with enhanced quality and balance"""
    try:
        print(colored(f"Mixing audio with music volume: {music_volume} (fade in: {fade_in}s, fade out: {fade_out}s)", "blue"))
        
        # Load voice audio
        voice = AudioFileClip(voice_path)
        voice_duration = voice.duration
        print(colored(f"Voice duration: {voice_duration:.2f}s", "blue"))
        
        # Check if music path exists and is valid
        if not music_path or not os.path.exists(music_path):
            print(colored(f"No valid music path provided: {music_path}", "yellow"))
            print(colored("Using voice audio only", "blue"))
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
            print(colored(f"Music file: {os.path.basename(music_path)}", "blue"))
            print(colored(f"Music duration: {music_duration:.2f}s, size: {music_size_mb:.2f} MB", "blue"))
            
            # Apply dynamic volume adjustment to music
            try:
                # Create a function that adjusts volume dynamically
                def adjust_volume(t):
                    # Base volume is the music_volume parameter
                    base_vol = float(music_volume)  # Ensure it's a float
                    
                    # Fade in at the beginning
                    if t < fade_in and fade_in > 0:
                        return base_vol * (t / fade_in)
                    
                    # Fade out at the end
                    elif fade_out > 0 and t > voice_duration - fade_out:
                        fade_position = (t - (voice_duration - fade_out)) / fade_out
                        return base_vol * (1 - fade_position)
                    
                    # Normal volume during the rest
                    return base_vol
                
                # Apply volume adjustment with error handling
                print(colored(f"Applying dynamic volume adjustment", "blue"))
                music = music.fl(lambda gf, t: gf(t) * adjust_volume(t), keep_duration=True)
            except ValueError as e:
                print(colored(f"Error applying dynamic volume: {str(e)}", "yellow"))
                # Fallback to simple volume adjustment
                print(colored(f"Falling back to simple volume adjustment", "blue"))
                music = music.volumex(music_volume)
            
            # Apply fades to music
            if fade_in > 0:
                print(colored(f"Applying fade in: {fade_in}s", "blue"))
                music = music.audio_fadein(fade_in)
            
            if fade_out > 0:
                print(colored(f"Applying fade out: {fade_out}s", "blue"))
                music = music.audio_fadeout(fade_out)
            
            # Handle music duration relative to voice
            if music_duration < voice_duration:
                print(colored(f"Music shorter than voice ({music_duration:.2f}s < {voice_duration:.2f}s), looping", "blue"))
                # Loop music to match voice duration
                from moviepy.audio.fx.all import audio_loop
                music = audio_loop(music, duration=voice_duration)
            elif music_duration > voice_duration:
                print(colored(f"Music longer than voice ({music_duration:.2f}s > {voice_duration:.2f}s), trimming", "blue"))
                # Trim music to match voice duration
                music = music.subclip(0, voice_duration)
            
            # Normalize voice audio for consistent levels
            from moviepy.audio.fx.all import audio_normalize
            print(colored("Normalizing voice audio", "blue"))
            voice = voice.fx(audio_normalize)
            
            # Boost voice slightly to ensure clarity over music
            print(colored("Boosting voice volume for clarity", "blue"))
            voice = voice.volumex(1.2)
            
            # Composite audio - put music first in the list so voice is layered on top
            # This ensures the voice is more prominent in the mix
            print(colored("Creating composite audio", "blue"))
            final_audio = CompositeAudioClip([music, voice])
            
            # Write the mixed audio to the output path with high quality settings
            print(colored(f"Writing output to: {output_path}", "blue"))
            final_audio.write_audiofile(output_path, fps=44100, bitrate="192k")
            
            print(colored(f"✓ Successfully mixed voice and music: {os.path.basename(music_path)}", "green"))
            return output_path
            
        except Exception as music_error:
            print(colored(f"Error processing music: {str(music_error)}", "yellow"))
            print(colored("Falling back to voice audio only", "blue"))
            # Normalize voice audio for consistent levels
            from moviepy.audio.fx.all import audio_normalize
            voice = voice.fx(audio_normalize)
            voice.write_audiofile(output_path, fps=44100, bitrate="192k")
            return output_path
    
    except Exception as e:
        print(colored(f"Error mixing audio: {str(e)}", "red"))
        traceback.print_exc()  # Print the full traceback for debugging
        print(colored("Falling back to voice audio only", "yellow"))
        
        # Make sure we actually copy the voice audio to the output path
        try:
            # Copy the voice audio to the output path
            shutil.copy(voice_path, output_path)
            print(colored(f"Copied voice audio to {output_path}", "blue"))
            return output_path
        except Exception as copy_error:
            print(colored(f"Error copying voice audio: {str(copy_error)}", "red"))
            # If we can't even copy the file, return the original voice path
            return voice_path

def main():
    """Test audio mixing function"""
    print(colored("=== Testing Audio Mixing ===", "blue"))
    
    # Check if we have the required arguments
    if len(sys.argv) < 3:
        print(colored("Usage: python test_audio_mixing.py <voice_file> <music_file> [music_volume]", "yellow"))
        return
    
    voice_path = sys.argv[1]
    music_path = sys.argv[2]
    
    # Optional music volume parameter
    music_volume = 0.3
    if len(sys.argv) > 3:
        try:
            music_volume = float(sys.argv[3])
            print(colored(f"Using provided music volume: {music_volume}", "blue"))
        except ValueError:
            print(colored(f"Invalid music volume: {sys.argv[3]}, using default: {music_volume}", "yellow"))
    
    # Check if files exist
    if not os.path.exists(voice_path):
        print(colored(f"Voice file not found: {voice_path}", "red"))
        return
    
    if not os.path.exists(music_path):
        print(colored(f"Music file not found: {music_path}", "red"))
        return
    
    # Create output directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Mix audio
    output_path = f"temp/mixed_audio_{uuid.uuid4()}.mp3"
    result = mix_audio(voice_path, music_path, output_path, music_volume=music_volume)
    
    if result:
        print(colored(f"✓ Audio mixing successful: {output_path}", "green"))
    else:
        print(colored("✗ Audio mixing failed", "red"))

if __name__ == "__main__":
    main() 