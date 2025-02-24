import asyncio
from video_generator import VideoGenerator
from termcolor import colored
import os

async def test_single_video():
    """Test video generation for a single channel using existing script"""
    
    # Initialize video generator
    video_gen = VideoGenerator()
    
    # Choose one channel to test
    channel_type = 'tech_humor'  # or any other channel you prefer
    script_file = f"cache/scripts/{channel_type}_latest.json"
    
    print(colored(f"\n=== Testing Video Generation for {channel_type} ===", "blue"))
    
    # Check if script exists
    if not os.path.exists(script_file):
        print(colored(f"✗ No script found at: {script_file}", "red"))
        return
    
    print(colored("✓ Found existing script", "green"))
    print(colored("\nStarting video generation...", "blue"))
    
    # Generate video
    success = await video_gen.create_video(
        script_file=script_file,
        channel_type=channel_type
    )
    
    if success:
        print(colored("\nVideo files generated:", "green"))
        print(colored(f"- Latest: output/videos/{channel_type}_latest.mp4", "cyan"))
        print(colored(f"- Check output/videos/ for timestamped version", "cyan"))
    else:
        print(colored("\n✗ Video generation failed", "red"))

if __name__ == "__main__":
    asyncio.run(test_single_video()) 