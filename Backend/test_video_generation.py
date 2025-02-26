import asyncio
from video_generator import VideoGenerator
from content_validator import ScriptGenerator
from termcolor import colored
import json
import os

async def test_video_generation():
    """Test video generation pipeline"""
    
    # Create test directories
    os.makedirs("cache/scripts", exist_ok=True)
    os.makedirs("temp/videos", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)
    
    print(colored("\n=== Testing Video Generation for tech_humor ===", "blue"))
    
    # Generate script using OpenAI
    script_generator = ScriptGenerator()
    success, script_data = await script_generator.generate_script(
        topic="Why Programmers Need Coffee",
        channel_type="tech_humor"
    )
    
    if not success:
        print(colored("✗ Failed to generate script", "red"))
        return
    
    print(colored("✓ Script generated successfully", "green"))
    
    # Save script
    script_path = "cache/scripts/tech_humor_latest.json"
    with open(script_path, 'w') as f:
        json.dump({
            "script": script_data,
            "preview": "A humorous take on programmers' coffee addiction",
            "is_valid": True
        }, f, indent=2)
    
    # Initialize video generator
    generator = VideoGenerator()
    
    # Generate video
    success = await generator.create_video(
        script_file=script_path,
        channel_type="tech_humor"
    )
    
    if success:
        print(colored("\n✓ Video generation test passed", "green"))
        print(colored("\nVideo files generated:", "green"))
        print(colored("- Latest: output/videos/tech_humor_latest.mp4", "cyan"))
        print(colored("- Check output/videos/ for timestamped version", "cyan"))
    else:
        print(colored("\n✗ Video generation test failed", "red"))

if __name__ == "__main__":
    asyncio.run(test_video_generation()) 