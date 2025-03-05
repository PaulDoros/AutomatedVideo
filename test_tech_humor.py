import os
import sys
import uuid
import asyncio
from termcolor import colored

# Add the Backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Backend'))

# Import functions from video_generator.py
from video_generator import VideoGenerator

async def test_tech_humor_video():
    """
    Generate a test video for the tech_humor channel with clearly audible background music.
    """
    try:
        print(colored("=== Creating Tech Humor Test Video ===", "blue", attrs=["bold"]))
        
        # Create output directory
        os.makedirs("output/test", exist_ok=True)
        
        # Initialize the video generator
        generator = VideoGenerator()
        
        # Set a higher music volume for testing
        generator.music_volume = 0.5  # 50% volume for clearer music
        print(colored(f"Set music volume to: {generator.music_volume}", "cyan"))
        
        # Generate a video for tech_humor channel
        channel_type = "tech_humor"
        
        # Create a simple script
        script = """
        Why did the programmer quit his job?
        Because he didn't get arrays.
        
        What's a programmer's favorite hangout spot?
        The Foo Bar.
        
        Why do programmers always mix up Christmas and Halloween?
        Because Oct 31 == Dec 25.
        
        How many programmers does it take to change a light bulb?
        None, that's a hardware problem.
        """
        
        # Save the script to a temporary file
        script_path = os.path.join("temp", f"tech_humor_test_{uuid.uuid4()}.txt")
        
        # Create script data in the format expected by the VideoGenerator
        script_data = {
            "script": script,
            "title": "Tech Humor Test",
            "channel_type": channel_type,
            "topic": "programming jokes",
            "timestamp": "2023-01-01T00:00:00"
        }
        
        # Save as JSON
        import json
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, indent=2)
        
        print(colored(f"Created test script: {script_path}", "cyan"))
        
        # Generate the video
        print(colored("Generating video...", "cyan"))
        
        # Generate timestamp for unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{channel_type}_test_{timestamp}.mp4"
        output_path = os.path.join("output/test", output_filename)
        
        # Call the async create_video method
        success = await generator.create_video(
            script_file=script_path,
            channel_type=channel_type,
            output_path=output_path
        )
        
        if success:
            print(colored(f"\nTest video created successfully: {output_path}", "green", attrs=["bold"]))
            print(colored("Please check this video to verify the background music is audible.", "green"))
            return output_path
        else:
            print(colored("\nFailed to create test video", "red", attrs=["bold"]))
            return None
        
    except Exception as e:
        print(colored(f"Error creating test video: {str(e)}", "red"))
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run the async test function"""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(test_tech_humor_video())
    return result

if __name__ == "__main__":
    main() 