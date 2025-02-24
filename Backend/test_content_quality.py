from content_validator import ContentValidator, ScriptGenerator
from thumbnail_validator import ThumbnailValidator
from thumbnail_generator import ThumbnailGenerator
from termcolor import colored
import os
import asyncio
import json

async def test_script_generation():
    """Test script generation and validation for each channel"""
    generator = ScriptGenerator()
    channels = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    print(colored("\n=== Testing Script Generation ===", "blue"))
    
    for channel in channels:
        print(colored(f"\nTesting {channel} channel:", "blue"))
        
        # Check cache first
        cache_file = f"cache/scripts/{channel}_latest.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                if cached.get('is_valid'):
                    print(colored("✓ Using cached valid script", "green"))
                    print(colored("Script preview:", "yellow"))
                    print(cached['script'][:200] + "...")
                    continue
        
        # Only generate new if no valid cache exists
        success, script = await generator.generate_script(
            topic=f"Test topic for {channel}",
            channel_type=channel
        )
        
        if success:
            print(colored("✓ Script generated successfully", "green"))
            print(colored("Script preview:", "yellow"))
            print(script[:200] + "...")
        else:
            print(colored(f"✗ Script generation failed: {script}", "red"))

def test_thumbnail_generation():
    """Test thumbnail generation and validation"""
    # First generate test thumbnails
    generator = ThumbnailGenerator()
    generator.generate_test_thumbnails()
    
    validator = ThumbnailValidator()
    test_thumbnails = {
        'tech_humor': 'test_thumbnails/tech_humor.jpg',
        'ai_money': 'test_thumbnails/ai_money.jpg',
        'baby_tips': 'test_thumbnails/baby_tips.jpg',
        'quick_meals': 'test_thumbnails/quick_meals.jpg',
        'fitness_motivation': 'test_thumbnails/fitness_motivation.jpg'
    }
    
    print(colored("\n=== Testing Thumbnail Validation ===", "blue"))
    
    for channel, thumb_path in test_thumbnails.items():
        print(colored(f"\nTesting {channel} thumbnail:", "blue"))
        if os.path.exists(thumb_path):
            success, message = validator.validate_thumbnail(thumb_path, channel)
            status = "✓" if success else "✗"
            color = "green" if success else "red"
            print(colored(f"{status} {message}", color))
        else:
            print(colored("✗ Test thumbnail not found", "red"))

async def main():
    print(colored("=== Content Quality Test Suite ===", "blue"))
    
    # Test script generation
    await test_script_generation()
    
    # Test thumbnail validation
    test_thumbnail_generation()

if __name__ == "__main__":
    asyncio.run(main()) 