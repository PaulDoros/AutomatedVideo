import os
import sys
import uuid
from termcolor import colored
import random
import glob

# Add the Backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Backend'))

# Import the mix_audio function from video.py
from video import mix_audio, log_info, log_success, log_error

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