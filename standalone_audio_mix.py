import os
import sys
import numpy as np
from moviepy.editor import AudioFileClip, CompositeAudioClip
import traceback

def log_info(message, emoji="ℹ️"):
    print(f"{emoji} {message}")

def log_success(message, emoji="✅"):
    print(f"{emoji} {message}")

def log_error(message, emoji="❌"):
    print(f"{emoji} {message}")

def mix_audio_standalone(voice_path, music_path, output_path, music_volume=0.1, fade_in=2.0, fade_out=3.0):
    """Mix voice audio with background music."""
    try:
        log_info(f"Loading voice audio: {voice_path}")
        voice = AudioFileClip(voice_path)
        
        log_info(f"Loading music audio: {music_path}")
        music = AudioFileClip(music_path)
        
        # Get the duration of the voice clip
        voice_duration = voice.duration
        log_info(f"Voice duration: {voice_duration}s")
        
        # If music is shorter than voice, loop it
        if music.duration < voice_duration:
            log_info(f"Music duration ({music.duration}s) is shorter than voice, looping it")
            n_loops = int(np.ceil(voice_duration / music.duration))
            music = music.loop(n=n_loops)
        
        # Trim music to match voice duration
        music = music.subclip(0, voice_duration)
        
        # Apply a constant volume adjustment to music
        log_info(f"Adjusting music volume (base: {music_volume})")
        music = music.volumex(music_volume)
        
        # Apply fade in and fade out
        if fade_in > 0:
            log_info(f"Applying fade in: {fade_in}s")
            music = music.audio_fadein(fade_in)
        
        if fade_out > 0:
            log_info(f"Applying fade out: {fade_out}s")
            music = music.audio_fadeout(fade_out)
        
        # Combine voice and music
        log_info("Combining voice and music")
        final_audio = CompositeAudioClip([voice, music])
        
        # Write to file
        log_info(f"Writing mixed audio to {output_path}")
        final_audio.write_audiofile(output_path, fps=44100)
        
        log_success(f"Successfully mixed audio: {output_path}")
        return output_path
    
    except Exception as e:
        log_error(f"Error mixing audio: {str(e)}")
        traceback.print_exc()
        return None

def find_audio_file(directory, extension=".mp3"):
    """Find an audio file in the specified directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                return os.path.join(root, file)
    return None

def test_audio_mixing():
    """Test the audio mixing functionality."""
    # Create output directory if it doesn't exist
    os.makedirs("output/test", exist_ok=True)
    
    # Define paths for testing
    voice_path = find_audio_file("assets/music")
    music_path = find_audio_file("assets/music")
    
    # If we found the same file for both, look in different directories
    if voice_path == music_path and voice_path is not None:
        # Try to find a different music file
        for root, dirs, files in os.walk("assets/music"):
            for file in files:
                if file.endswith(".mp3") and os.path.join(root, file) != voice_path:
                    music_path = os.path.join(root, file)
                    break
            if voice_path != music_path:
                break
    
    # If we still don't have two different files, create a test file
    if voice_path is None or music_path is None or voice_path == music_path:
        log_info("Creating test audio files...")
        from moviepy.editor import AudioClip
        
        # Create a simple sine wave audio for voice
        def make_voice_frame(t):
            return np.sin(2 * np.pi * 440 * t)
        
        voice_audio = AudioClip(make_voice_frame, duration=5)
        voice_path = "output/test/test_voice.mp3"
        voice_audio.write_audiofile(voice_path, fps=44100)
        log_success(f"Created test voice audio: {voice_path}")
        
        # Create a simple sine wave audio for music
        def make_music_frame(t):
            return np.sin(2 * np.pi * 220 * t)
        
        music_audio = AudioClip(make_music_frame, duration=10)
        music_path = "output/test/test_music.mp3"
        music_audio.write_audiofile(music_path, fps=44100)
        log_success(f"Created test music file: {music_path}")
    
    output_path = "output/test/mixed_audio_test.mp3"
    
    # Now mix the audio
    log_info(f"Mixing audio: voice={voice_path}, music={music_path}")
    mixed_path = mix_audio_standalone(
        voice_path=voice_path,
        music_path=music_path,
        output_path=output_path,
        music_volume=0.15,
        fade_in=2.0,
        fade_out=3.0
    )
    
    if mixed_path:
        log_success(f"Successfully mixed audio: {mixed_path}")
        log_info(f"You can listen to the mixed audio at: {os.path.abspath(mixed_path)}")
        return True
    else:
        log_error("Failed to mix audio")
        return False

if __name__ == "__main__":
    test_audio_mixing() 