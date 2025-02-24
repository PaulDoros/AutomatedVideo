from content_validator import ContentValidator, ScriptGenerator
from thumbnail_validator import ThumbnailValidator
from termcolor import colored
import os

def test_script_generation():
    """Test script generation and validation for each channel"""
    generator = ScriptGenerator()
    channels = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    print(colored("\n=== Testing Script Generation ===", "blue"))
    
    for channel in channels:
        print(colored(f"\nTesting {channel} channel:", "blue"))
        
        # Test script generation
        success, script = generator.generate_script(
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
    validator = ThumbnailValidator()
    test_thumbnails = {
        'tech_humor': 'test_thumbnails/tech.jpg',
        'ai_money': 'test_thumbnails/money.jpg',
        'baby_tips': 'test_thumbnails/baby.jpg',
        'quick_meals': 'test_thumbnails/food.jpg',
        'fitness_motivation': 'test_thumbnails/fitness.jpg'
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

def main():
    print(colored("=== Content Quality Test Suite ===", "blue"))
    
    # Test script generation
    test_script_generation()
    
    # Test thumbnail validation
    test_thumbnail_generation()

if __name__ == "__main__":
    main() 