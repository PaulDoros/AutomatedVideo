import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Backend.video import generate_video
from Backend.video import log_success, log_error

def test_background_music():
    # Define paths for testing
    background_path = "assets/videos/categorized/tech/tech_video_1.mp4"
    audio_path = "temp/tts/tech_latest.mp3"
    subtitles_path = "temp/subtitles/tech_latest.srt"
    script_path = "temp/scripts/tech_latest.txt"
    output_path = "output/videos/tech_test_music.mp4"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    os.makedirs(os.path.dirname(subtitles_path), exist_ok=True)
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a simple test script
    with open(script_path, "w") as f:
        f.write("This is a test script for background music functionality.")
    
    # Create a simple test subtitle file
    with open(subtitles_path, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:05,000\nThis is a test script for background music functionality.")
    
    # Check if the background video exists
    if not os.path.exists(background_path):
        log_error(f"Background video not found: {background_path}")
        # Try to find any video in the assets directory
        for root, dirs, files in os.walk("assets/videos"):
            for file in files:
                if file.endswith(".mp4"):
                    background_path = os.path.join(root, file)
                    log_success(f"Using alternative background video: {background_path}")
                    break
            if os.path.exists(background_path):
                break
    
    # Check if we have a background video
    if not os.path.exists(background_path):
        log_error("No background video found. Exiting.")
        return
    
    # Create a simple test audio file if it doesn't exist
    if not os.path.exists(audio_path):
        # Try to find any mp3 file to use as a test
        for root, dirs, files in os.walk("assets"):
            for file in files:
                if file.endswith(".mp3"):
                    import shutil
                    shutil.copy(os.path.join(root, file), audio_path)
                    log_success(f"Using {os.path.join(root, file)} as test audio")
                    break
            if os.path.exists(audio_path):
                break
    
    # Check if we have an audio file
    if not os.path.exists(audio_path):
        log_error("No audio file found. Exiting.")
        return
    
    try:
        # Generate the video with background music
        final_video_path = generate_video(
            background_path=background_path,
            audio_path=audio_path,
            subtitles_path=subtitles_path,
            content_type="tech",
            script_path=script_path,
            use_background_music=True,
            music_volume=0.15,
            music_fade_in=2.0,
            music_fade_out=3.0
        )
        
        log_success(f"Video generated successfully: {final_video_path}")
    except Exception as e:
        log_error(f"Error generating video: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_background_music() 