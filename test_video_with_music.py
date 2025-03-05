import os
import sys
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary functions
from Backend.video import generate_video, log_success, log_error, log_info

def test_video_generation():
    """Test generating a video with background music."""
    # Create output directory if it doesn't exist
    os.makedirs("output/test", exist_ok=True)
    
    # Define paths for testing
    background_path = None
    audio_path = None
    subtitles_path = "output/test/test_subtitles.srt"
    script_path = "output/test/test_script.txt"
    output_path = "output/test/test_video_with_music.mp4"
    
    # Create a simple test script
    script_text = "This is a test script for video generation with background music. We want to make sure that the audio mixing functionality works correctly."
    with open(script_path, "w") as f:
        f.write(script_text)
    
    # Create a simple test subtitle file
    with open(subtitles_path, "w") as f:
        f.write("1\n00:00:00,000 --> 00:00:05,000\nThis is a test script for video generation\n\n")
        f.write("2\n00:00:05,000 --> 00:00:10,000\nwith background music.\n\n")
        f.write("3\n00:00:10,000 --> 00:00:15,000\nWe want to make sure that the audio\n\n")
        f.write("4\n00:00:15,000 --> 00:00:20,000\nmixing functionality works correctly.")
    
    # Find a background video
    log_info("Looking for a background video...")
    for root, dirs, files in os.walk("assets/videos"):
        for file in files:
            if file.endswith(".mp4"):
                background_path = os.path.join(root, file)
                log_success(f"Found background video: {background_path}")
                break
        if background_path:
            break
    
    # If no background video found, exit
    if not background_path:
        log_error("No background video found. Exiting.")
        return False
    
    # Find an audio file
    log_info("Looking for an audio file...")
    for root, dirs, files in os.walk("assets/music"):
        for file in files:
            if file.endswith(".mp3"):
                audio_path = os.path.join(root, file)
                log_success(f"Found audio file: {audio_path}")
                break
        if audio_path:
            break
    
    # If no audio file found, exit
    if not audio_path:
        log_error("No audio file found. Exiting.")
        return False
    
    # Generate the video
    try:
        log_info("Generating video with background music...")
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
        log_info(f"You can watch the video at: {os.path.abspath(final_video_path)}")
        return True
    except Exception as e:
        log_error(f"Error generating video: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_video_generation() 