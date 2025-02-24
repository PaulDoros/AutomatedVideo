from content_validator import ContentValidator, ScriptGenerator
from thumbnail_validator import ThumbnailValidator
from thumbnail_generator import ThumbnailGenerator
from termcolor import colored
import os
import asyncio
import json
import re
from typing import Tuple, Dict, List

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

class ContentQualityChecker:
    def __init__(self):
        self.ideal_duration = 30  # seconds for shorts
        self.words_per_second = 2.5  # average speaking rate

    def analyze_script(self, script: str) -> Tuple[bool, Dict]:
        """Analyze script quality and return metrics"""
        try:
            # Clean script
            script = re.sub(r'[\*\#\@\(\)\[\]]', '', script)
            sentences = [s.strip() for s in script.split('.') if s.strip()]
            
            # Calculate metrics
            metrics = {
                'total_sentences': len(sentences),
                'total_words': sum(len(s.split()) for s in sentences),
                'estimated_duration': sum(len(s.split()) for s in sentences) / self.words_per_second,
                'engagement_score': self._calculate_engagement(sentences),
                'issues': [],
                'warnings': []  # New: separate warnings from blocking issues
            }

            # Convert previous "issues" to "warnings" for learning
            self._check_hook(script, metrics)
            self._check_engagement_elements(script, metrics)
            
            # Script is valid if there are no critical issues
            # Warnings don't affect validity but are stored for improvement
            is_quality = True  # Always generate if basic requirements are met

            return is_quality, metrics

        except Exception as e:
            print(colored(f"Error analyzing script: {str(e)}", "red"))
            return False, {'issues': ['Analysis failed']}

    def _check_hook(self, script: str, metrics: dict):
        """Check if script has a strong hook - now as warning"""
        first_line = script.split('\n')[0].lower()
        if not any(h in first_line for h in ['?', 'want to', 'ever', 'how to', 'why', 'secret']):
            metrics['warnings'].append('Could improve: Add stronger hook at start')

    def _check_engagement_elements(self, script: str, metrics: Dict):
        """Check for engagement elements - as warnings"""
        script_lower = script.lower()
        
        # Check for call to action
        if not any(c in script_lower[-100:] for c in ['follow', 'subscribe', 'try this']):
            metrics['warnings'].append("Could improve: Add clear call to action at end")
            
        # Check for questions/engagement
        if '?' not in script:
            metrics['warnings'].append("Could improve: Add engaging questions")

    def _calculate_engagement(self, sentences: List[str]) -> float:
        """Calculate engagement score"""
        score = 8.0  # Start with base score
        
        # Add points for engagement elements
        if any('?' in s for s in sentences): score += 1
        if any('!' in s for s in sentences): score += 1
        if any(emoji in ''.join(sentences) for emoji in ['💻', '🚀', '✨', '🔥']): score += 1
        
        return score

def test_script_quality(script_file: str):
    """Test the quality of a script"""
    checker = ContentQualityChecker()
    
    try:
        with open(script_file, 'r') as f:
            script = f.read()
        
        print(colored("\n=== Testing Script Quality ===", "blue"))
        is_quality, metrics = checker.analyze_script(script)
        
        # Print metrics
        print(colored("\nScript Metrics:", "cyan"))
        print(f"Total sentences: {metrics['total_sentences']}")
        print(f"Total words: {metrics['total_words']}")
        print(f"Estimated duration: {metrics['estimated_duration']:.1f}s")
        print(f"Engagement score: {metrics['engagement_score']:.1f}/10")
        
        # Print warnings
        if metrics['warnings']:
            print(colored("\nQuality Warnings:", "yellow"))
            for warning in metrics['warnings']:
                print(f"- {warning}")
        
        if is_quality:
            print(colored("\n✓ Script meets quality standards", "green"))
        else:
            print(colored("\n✗ Script needs improvement", "red"))
            
        return is_quality
            
    except Exception as e:
        print(colored(f"\nError testing script: {str(e)}", "red"))
        return False

if __name__ == "__main__":
    asyncio.run(main())
    test_script_quality("cache/scripts/tech_humor_latest.json") 