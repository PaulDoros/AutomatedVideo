import asyncio
from content_validator import ScriptGenerator
from video_generator import VideoGenerator
from thumbnail_generator import ThumbnailGenerator
from youtube_uploader import YouTubeUploader
from termcolor import colored
import os
import json

async def generate_and_upload():
    """Generate and upload content for all channels"""
    
    # Initialize all components
    script_gen = ScriptGenerator()
    video_gen = VideoGenerator()
    thumb_gen = ThumbnailGenerator()
    uploader = YouTubeUploader()
    
    # Channel topics
    topics = {
        'tech_humor': 'When Your Code Works But You Don\'t Know Why',
        'ai_money': 'Make $100/Day with ChatGPT Automation',
        'baby_tips': 'Help Your Baby Sleep Through the Night',
        'quick_meals': '5-Minute Healthy Breakfast Ideas',
        'fitness_motivation': '10-Minute Morning Workout Routine'
    }
    
    # Create necessary directories
    os.makedirs("output/videos", exist_ok=True)
    os.makedirs("test_thumbnails", exist_ok=True)
    os.makedirs("cache/scripts", exist_ok=True)
    
    for channel, topic in topics.items():
        print(colored(f"\n=== Processing {channel}: {topic} ===", "blue"))
        
        try:
            # 1. Generate Script
            print(colored("\nGenerating script...", "blue"))
            success, script = await script_gen.generate_script(
                topic=topic,
                channel_type=channel
            )
            if not success:
                print(colored(f"✗ Script generation failed for {channel}", "red"))
                continue
            print(colored("✓ Script generated", "green"))
            
            # 2. Generate Video
            print(colored("\nGenerating video...", "blue"))
            script_file = f"cache/scripts/{channel}_latest.json"
            success = await video_gen.create_video(
                script_file=script_file,
                channel_type=channel
            )
            if not success:
                print(colored(f"✗ Video generation failed for {channel}", "red"))
                continue
            print(colored("✓ Video generated", "green"))
            
            # 3. Generate Thumbnail
            print(colored("\nGenerating thumbnail...", "blue"))
            thumb_gen.generate_test_thumbnails()
            print(colored("✓ Thumbnail generated", "green"))
            
            # 4. Upload to YouTube
            print(colored("\nUploading to YouTube...", "blue"))
            success, result = await uploader.upload_video(
                channel_type=channel,
                video_path=f"output/videos/{channel}_latest.mp4",
                title=topic,
                description=None,  # Will use script as description
                tags=[channel, 'shorts', 'content']
            )
            
            if success:
                print(colored(f"✓ Upload complete for {channel}", "green"))
                print(colored(f"Video ID: {result}", "cyan"))
            else:
                print(colored(f"✗ Upload failed for {channel}: {result}", "red"))
                
        except Exception as e:
            print(colored(f"✗ Error processing {channel}: {str(e)}", "red"))
            continue
        
        print(colored(f"\n=== Completed {channel} ===", "green"))

if __name__ == "__main__":
    asyncio.run(generate_and_upload()) 