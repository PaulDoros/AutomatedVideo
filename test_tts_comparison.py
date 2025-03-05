from video_generator import VideoGenerator
import asyncio
import os
from termcolor import colored

async def test_tts_comparison():
    """Test different TTS options and compare their output"""
    
    # Initialize video generator
    generator = VideoGenerator()
    
    # Test script with different styles
    test_scripts = {
        'tech_humor': """Hey everyone! Welcome to another exciting tech video! 
Today we're going to explore something mind-blowing about artificial intelligence. 
You won't believe what these neural networks can do! 😮""",
        
        'ai_money': """In this video, we'll analyze how AI is transforming the financial sector.
Let's look at three key ways machine learning is revolutionizing investment strategies.
The results might surprise you."""
    }
    
    # Test each channel type
    for channel_type, script in test_scripts.items():
        print(colored(f"\n=== Testing {channel_type} ===", "blue"))
        
        print("\n1. Testing Coqui TTS...")
        coqui_path = await generator._generate_tts_coqui(script, channel_type)
        
        print("\n2. Testing OpenAI TTS...")
        openai_path = await generator._generate_tts_openai(script, channel_type)
        
        if coqui_path and openai_path:
            print(colored("\n✓ Both TTS systems generated audio successfully!", "green"))
            print(f"Coqui output: {coqui_path}")
            print(f"OpenAI output: {openai_path}")
        else:
            print(colored("\n! Some TTS generations failed", "red"))
            if not coqui_path:
                print("- Coqui TTS failed")
            if not openai_path:
                print("- OpenAI TTS failed")

if __name__ == "__main__":
    print(colored("Starting TTS comparison test...", "blue"))
    asyncio.run(test_tts_comparison()) 