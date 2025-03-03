import asyncio
   
from video_generator import VideoGenerator
from content_validator import ScriptGenerator
from termcolor import colored
import json
import os

async def test_video_generation():
    """Test video generation pipeline with voice diversification"""
    
    # Create test directories
    os.makedirs("cache/scripts", exist_ok=True)
    os.makedirs("temp/videos", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)
    
    print(colored("\n=== Testing Video Generation with Voice Diversification ===", "blue"))
    
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
    
    # Enable Coqui TTS with voice diversification
    generator.use_coqui_tts = True
    
    # Generate video
    success = await generator.create_video(
        script_file=script_path,
        channel_type="tech_humor"
    )
    
    if success:
        print(colored("\n✓ Video generation test passed with voice diversification", "green"))
        print(colored("\nVideo files generated:", "green"))
        print(colored("- Latest: output/videos/tech_humor_latest.mp4", "cyan"))
        print(colored("- Check output/videos/ for timestamped version", "cyan"))
    else:
        print(colored("\n✗ Video generation test failed", "red"))

async def test_multiple_voices():
    """Test video generation with multiple different voices"""
    
    # Create test directories
    os.makedirs("cache/scripts", exist_ok=True)
    os.makedirs("temp/videos", exist_ok=True)
    os.makedirs("output/videos", exist_ok=True)
    
    # Test different channel types with different voices
    channel_types = ["tech_humor", "ai_money"]
    topics = {
        "tech_humor": "10 Programming Hacks Every Developer Should Know",
        "ai_money": "How AI is Transforming the Financial Industry"
    }
    
    for channel_type in channel_types:
        print(colored(f"\n=== Testing Video Generation for {channel_type} ===", "blue"))
        
        # Generate script using OpenAI
        script_generator = ScriptGenerator()
        success, script_data = await script_generator.generate_script(
            topic=topics[channel_type],
            channel_type=channel_type
        )
        
        if not success:
            print(colored(f"✗ Failed to generate script for {channel_type}", "red"))
            continue
        
        print(colored(f"✓ Script generated successfully for {channel_type}", "green"))
        
        # Save script
        script_path = f"cache/scripts/{channel_type}_latest.json"
        with open(script_path, 'w') as f:
            json.dump({
                "script": script_data,
                "preview": f"Content for {channel_type}",
                "is_valid": True
            }, f, indent=2)
        
        # Initialize video generator
        generator = VideoGenerator()
        
        # Enable Coqui TTS with voice diversification
        generator.use_coqui_tts = True
        
        # Generate video
        success = await generator.create_video(
            script_file=script_path,
            channel_type=channel_type
        )
        
        if success:
            print(colored(f"\n✓ Video generation test passed for {channel_type}", "green"))
            print(colored(f"\nVideo files generated for {channel_type}:", "green"))
            print(colored(f"- Latest: output/videos/{channel_type}_latest.mp4", "cyan"))
        else:
            print(colored(f"\n✗ Video generation test failed for {channel_type}", "red"))

if __name__ == "__main__":
    # Choose which test to run
    print(colored("Select a test to run:", "cyan"))
    print(colored("1. Test single video generation with voice diversification", "cyan"))
    print(colored("2. Test multiple videos with different voices", "cyan"))
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_video_generation())
    elif choice == "2":
        asyncio.run(test_multiple_voices())
    else:
        print(colored("Invalid choice. Running default test.", "yellow"))
        asyncio.run(test_video_generation()) 