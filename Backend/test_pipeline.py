import asyncio
from content_validator import ScriptGenerator
from video_generator import VideoGenerator
from thumbnail_generator import ThumbnailGenerator
from youtube_uploader import YouTubeUploader
from termcolor import colored
import os

async def run_pipeline():
    """Run the full content generation and upload pipeline"""
    print(colored("\n=== Running Content Pipeline ===\n", "blue"))
    
    # Initialize components
    script_gen = ScriptGenerator()
    video_gen = VideoGenerator()
    thumb_gen = ThumbnailGenerator()
    uploader = YouTubeUploader()
    
    channels = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    for channel in channels:
        print(colored(f"\nProcessing {channel}:", "blue"))
        
        # Create output directories if they don't exist
        os.makedirs("output/videos", exist_ok=True)
        os.makedirs("test_thumbnails", exist_ok=True)
        
        # 1. Generate/Get Script
        script_file = f"cache/scripts/{channel}_latest.json"
        if not os.path.exists(script_file):
            print("Generating new script...")
            success, script = await script_gen.generate_script(
                topic=f"Test topic for {channel}",
                channel_type=channel
            )
            if not success:
                print(colored(f"✗ Script generation failed for {channel}", "red"))
                continue
        else:
            print(colored("✓ Using cached script", "green"))
        
        # 2. Generate Video
        video_path = f"output/videos/{channel}_latest.mp4"
        if not os.path.exists(video_path):
            print("Generating video...")
            success = await video_gen.generate_video(channel, script_file)
            if not success:
                print(colored(f"✗ Video generation failed for {channel}", "red"))
                continue
        else:
            print(colored("✓ Using existing video", "green"))
        
        # 3. Generate Thumbnail
        thumb_path = f"test_thumbnails/{channel}.jpg"
        if not os.path.exists(thumb_path):
            print("Generating thumbnail...")
            thumb_gen.generate_test_thumbnails()
        else:
            print(colored("✓ Using existing thumbnail", "green"))
        
        # 4. Upload to YouTube
        print("Uploading to YouTube...")
        success, video_id = await uploader.upload_video(
            channel_type=channel,
            video_path=video_path,
            title=None,  # Will be extracted from script
            description=None,  # Will use script as description
            tags=[channel, 'content']
        )
        
        if success:
            print(colored(f"✓ Pipeline complete for {channel}", "green"))
        else:
            print(colored(f"✗ Upload failed for {channel}", "red"))

if __name__ == "__main__":
    asyncio.run(run_pipeline()) 