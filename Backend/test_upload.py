from youtube_uploader import YouTubeUploader
import asyncio
from termcolor import colored
import os
import json

async def test_uploads():
    uploader = YouTubeUploader()
    channels = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    print(colored("\n=== Testing YouTube Uploads ===\n", "blue"))
    
    for channel in channels:
        print(colored(f"\nUploading {channel} content:", "blue"))
        
        # Check if we have the video file
        video_path = f"output/videos/{channel}_latest.mp4"
        if not os.path.exists(video_path):
            print(colored(f"✗ Video file not found for {channel}", "red"))
            continue

        # Get title from script
        cache_file = f"cache/scripts/{channel}_latest.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                script = cached.get('script', '')
                # Extract title from script
                title = script.split('\n')[0].strip('*# ')
        else:
            title = f"Test video for {channel}"

        # Upload
        success, result = await uploader.upload_video(
            channel_type=channel,
            video_path=video_path,
            title=title,
            description=None,  # Will use script as description
            tags=[channel, 'test', 'content']
        )
        
        if success:
            print(colored(f"✓ Upload complete for {channel}", "green"))
        else:
            print(colored(f"✗ Upload failed for {channel}: {result}", "red"))

if __name__ == "__main__":
    asyncio.run(test_uploads()) 