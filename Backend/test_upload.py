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

        # Get title and thumbnail title from script
        cache_file = f"cache/scripts/{channel}_latest.json"
        title = f"Test video for {channel}"
        thumbnail_title = ""
        script = ""
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    script = cached.get('script', '')
                    thumbnail_title = cached.get('thumbnail_title', '')
                    
                    # Use thumbnail title if available, otherwise extract from script
                    if thumbnail_title:
                        title = thumbnail_title
                        print(colored(f"Using thumbnail title: {title}", "cyan"))
                    else:
                        # Extract title from script (first line)
                        title = script.split('\n')[0].strip('*# ')
                        print(colored(f"Using first line of script as title: {title}", "cyan"))
            except Exception as e:
                print(colored(f"Warning: Could not load script from cache: {str(e)}", "yellow"))
                title = f"Test video for {channel}"

        # Upload
        success, result = await uploader.upload_video(
            channel_type=channel,
            video_path=video_path,
            title=title,
            description=None,  # Will use professional description in uploader
            tags=[channel, 'test', 'shorts', 'youtube', 'content']
        )
        
        if success:
            print(colored(f"✓ Upload complete for {channel}", "green"))
        else:
            print(colored(f"✗ Upload failed for {channel}: {result}", "red"))

if __name__ == "__main__":
    asyncio.run(test_uploads()) 