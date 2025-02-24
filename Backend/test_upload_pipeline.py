import asyncio
from content_validator import ScriptGenerator
from video_generator import VideoGenerator
from thumbnail_generator import ThumbnailGenerator
from youtube_uploader import YouTubeUploader
from termcolor import colored
import os
import json

async def test_upload_pipeline():
    """Test the full content generation and upload pipeline"""
    
    # Create necessary directories
    os.makedirs("output/videos", exist_ok=True)
    os.makedirs("test_thumbnails", exist_ok=True)
    os.makedirs("cache/scripts", exist_ok=True)
    
    # Initialize components
    uploader = YouTubeUploader()
    
    channels = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    for channel in channels:
        print(colored(f"\n=== Processing {channel} ===", "blue"))
        
        # Check for video file
        video_path = f"output/videos/{channel}_latest.mp4"
        if not os.path.exists(video_path):
            print(colored(f"✗ No video found for {channel}", "red"))
            continue
            
        # Check for script
        script_file = f"cache/scripts/{channel}_latest.json"
        if not os.path.exists(script_file):
            print(colored(f"✗ No script found for {channel}", "red"))
            continue
            
        # Load script for title
        with open(script_file, 'r') as f:
            script_data = json.load(f)
            script = script_data.get('script', '')
            # Extract title from first line of script
            title = script.split('\n')[0].strip('*# ')
        
        # Check for thumbnail
        thumb_path = f"test_thumbnails/{channel}.jpg"
        if not os.path.exists(thumb_path):
            print(colored(f"✗ No thumbnail found for {channel}", "red"))
            continue
            
        print(colored("Found all required files:", "green"))
        print(colored(f"- Video: {video_path}", "cyan"))
        print(colored(f"- Script: {script_file}", "cyan"))
        print(colored(f"- Thumbnail: {thumb_path}", "cyan"))
        
        # Upload to YouTube
        print(colored("\nUploading to YouTube...", "blue"))
        success, result = await uploader.upload_video(
            channel_type=channel,
            video_path=video_path,
            title=title,
            description=script,
            tags=[channel, 'shorts', 'content']
        )
        
        if success:
            print(colored(f"✓ Upload complete for {channel}", "green"))
            print(colored(f"Video ID: {result}", "cyan"))
        else:
            print(colored(f"✗ Upload failed for {channel}: {result}", "red"))

if __name__ == "__main__":
    asyncio.run(test_upload_pipeline()) 