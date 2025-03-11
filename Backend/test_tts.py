#!/usr/bin/env python3
"""
Test script for TTS generation with improved English speaker selection
"""

import asyncio
from video_generator import VideoGenerator

async def test_tts():
    """Test the TTS generation with improved English speaker selection"""
    print("\n=== Testing TTS Generation with Improved English Speaker Selection ===\n")
    
    # Create a VideoGenerator instance
    vg = VideoGenerator()
    
    # Test text
    test_text = "This is a test of the text-to-speech system with better English speaker selection. The quick brown fox jumps over the lazy dog."
    
    # Test with different channel types
    channels = ["tech_humor", "ai_money", "baby_tips"]
    
    for channel in channels:
        print(f"\n--- Testing {channel} ---\n")
        tts_path = await vg._generate_tts(test_text, channel)
        
        if tts_path:
            print(f"\nSuccessfully generated TTS for {channel}: {tts_path}\n")
        else:
            print(f"\nFailed to generate TTS for {channel}\n")
    
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_tts()) 