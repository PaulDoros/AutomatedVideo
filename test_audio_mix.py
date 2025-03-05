import os
import sys
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions
from Backend.video import mix_audio, log_success, log_error, log_info

def test_audio_mixing():
    """Test the audio mixing functionality."""
    # Create output directory if it doesn't exist
    os.makedirs("output/test", exist_ok=True)
    
    # Define paths for testing
    voice_path = None
    music_path = None
    output_path = "output/test/mixed_audio_test.mp3"
    
    # Find a voice audio file
    log_info("Looking for a voice audio file...")
    for root, dirs, files in os.walk("temp/tts"):
        for file in files:
            if file.endswith(".mp3"):
                voice_path = os.path.join(root, file)
                log_success(f"Found voice audio: {voice_path}")
                break
        if voice_path:
            break
    
    # If no voice file found, look in assets
    if not voice_path:
        for root, dirs, files in os.walk("assets"):
            for file in files:
                if file.endswith(".mp3"):
                    voice_path = os.path.join(root, file)
                    log_success(f"Found voice audio in assets: {voice_path}")
                    break
            if voice_path:
                break
    
    # If still no voice file found, create a simple one
    if not voice_path:
        log_info("No voice audio found. Creating a test file...")
        from moviepy.editor import AudioClip
        import numpy as np
        
        # Create a simple sine wave audio
        def make_frame(t):
            return np.sin(2 * np.pi * 440 * t)
        
        audio = AudioClip(make_frame, duration=5)
        voice_path = "output/test/test_voice.mp3"
        audio.write_audiofile(voice_path, fps=44100)
        log_success(f"Created test voice audio: {voice_path}")
    
    # Find a music file
    log_info("Looking for a music file...")
    for root, dirs, files in os.walk("assets/music"):
        for file in files:
            if file.endswith(".mp3"):
                music_path = os.path.join(root, file)
                log_success(f"Found music file: {music_path}")
                break
        if music_path:
            break
    
    # If no music file found, create a simple one
    if not music_path:
        log_info("No music file found. Creating a test file...")
        from moviepy.editor import AudioClip
        import numpy as np
        
        # Create a simple sine wave audio with different frequency
        def make_frame(t):
            return np.sin(2 * np.pi * 220 * t)
        
        audio = AudioClip(make_frame, duration=10)
        music_path = "output/test/test_music.mp3"
        audio.write_audiofile(music_path, fps=44100)
        log_success(f"Created test music file: {music_path}")
    
    # Now mix the audio
    log_info(f"Mixing audio: voice={voice_path}, music={music_path}")
    try:
        # Fix for the adjust_volume function
        from Backend.video import adjust_volume
        
        # Print the original adjust_volume function
        import inspect
        log_info(f"Original adjust_volume function: {inspect.getsource(adjust_volume)}")
        
        # Try to mix the audio
        mixed_path = mix_audio(
            voice_path=voice_path,
            music_path=music_path,
            output_path=output_path,
            music_volume=0.15,
            fade_in=2.0,
            fade_out=3.0
        )
        
        log_success(f"Successfully mixed audio: {mixed_path}")
        log_info(f"You can listen to the mixed audio at: {os.path.abspath(mixed_path)}")
        
        return True
    except Exception as e:
        log_error(f"Error mixing audio: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_audio_mixing() 