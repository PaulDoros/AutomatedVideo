"""
Test script for improved tech humor generation.

This script tests the improved tech humor generator and creates a video with the generated script.
"""

import asyncio
import os
import sys
import re
from termcolor import colored
from improved_tech_humor import generate_improved_tech_humor
from video_generator import VideoGenerator

# Expanded list of tech topics for better variety
TECH_TOPICS = [
    # Programming Languages & Development
    "JavaScript", "Python", "Java", "C++", "TypeScript", "Rust", "Go", 
    "Ruby", "PHP", "Swift", "Kotlin", "debugging", "code reviews",
    "Stack Overflow", "GitHub", "Git merge conflicts", "pull requests",
    
    # Web Development
    "CSS", "HTML", "responsive design", "frontend development", 
    "backend development", "web frameworks", "React", "Angular", "Vue",
    
    # DevOps & Infrastructure
    "Docker", "Kubernetes", "cloud computing", "AWS", "Azure", 
    "DevOps", "CI/CD pipelines", "microservices", "serverless",
    
    # AI & Data
    "machine learning", "AI assistants", "ChatGPT", "data science",
    "neural networks", "big data", "algorithms", "natural language processing",
    
    # Tech Culture
    "startup culture", "tech interviews", "remote work", "Zoom meetings",
    "tech conferences", "hackathons", "agile development", "standup meetings",
    
    # Hardware & Systems
    "Windows updates", "Mac vs PC", "Linux", "mechanical keyboards",
    "multiple monitors", "gaming PCs", "smartphone addiction",
    
    # Miscellaneous Tech
    "dark mode", "password security", "tech support", "legacy code",
    "documentation", "coffee and coding", "blockchain", "NFTs",
    "virtual reality", "tech buzzwords", "Internet of Things"
]

def clean_script_for_tts(script):
    """Clean script to ensure it's in the right format for TTS"""
    # Remove any line numbers or prefixes like "1. Hook:" or "**1. Hook:**"
    script = re.sub(r'^\s*\**\d+\.\s*\w+:\s*\**', '', script, flags=re.MULTILINE)
    
    # Remove any markdown formatting
    script = re.sub(r'\*\*|\*|__|\^', '', script)
    
    # Remove any empty lines
    script = re.sub(r'\n\s*\n', '\n', script)
    
    # Ensure there's content
    if not script.strip():
        return "This is a placeholder text for TTS generation."
    
    return script.strip()

async def main():
    """Main function to test improved tech humor generation"""
    print(colored("\n=== Testing Improved Tech Humor Generation ===", "blue"))
    
    # Ask if user wants to use a custom topic or generate multiple jokes
    mode = input("Choose an option:\n1. Generate a single joke with custom topic\n2. Generate a single joke with random topic\n3. Generate multiple jokes and pick the best\nEnter choice (1-3): ").strip()
    
    if mode == "1":
        # Custom topic
        topic = input("Enter a tech topic for the joke: ").strip()
        if not topic:
            import random
            topic = random.choice(TECH_TOPICS)
            print(colored(f"No topic entered. Selected random topic: {topic}", "cyan"))
        
        # Generate improved tech humor script
        print(colored(f"\nGenerating improved tech humor script for topic: {topic}", "blue"))
        success, script = await generate_improved_tech_humor(topic)
        
        if not success or not script:
            print(colored("✗ Failed to generate script", "red"))
            return
        
        print(colored("\nGenerated Script:", "green"))
        print(colored(script, "cyan"))
        
    elif mode == "3":
        # Generate multiple jokes and pick the best
        num_jokes = int(input("How many jokes would you like to generate (2-5)? "))
        num_jokes = max(2, min(5, num_jokes))  # Ensure between 2-5
        
        # Generate jokes for different topics
        jokes = []
        import random
        selected_topics = random.sample(TECH_TOPICS, num_jokes)
        
        print(colored(f"\nGenerating {num_jokes} different tech humor scripts...", "blue"))
        
        for i, topic in enumerate(selected_topics):
            print(colored(f"\n[{i+1}/{num_jokes}] Generating script for topic: {topic}", "blue"))
            success, script = await generate_improved_tech_humor(topic)
            
            if success and script:
                jokes.append({"topic": topic, "script": script})
                print(colored(f"✓ Generated script for {topic}", "green"))
            else:
                print(colored(f"✗ Failed to generate script for {topic}", "red"))
        
        # Display all jokes
        print(colored("\n=== Generated Jokes ===", "blue"))
        for i, joke in enumerate(jokes):
            print(colored(f"\nJoke #{i+1} - Topic: {joke['topic']}", "yellow"))
            print(colored(joke["script"], "cyan"))
            print("-" * 50)
        
        # Let user pick the best joke
        selected = int(input(f"Select the best joke (1-{len(jokes)}): ")) - 1
        if 0 <= selected < len(jokes):
            # Clean the script for TTS
            clean_script = clean_script_for_tts(jokes[selected]["script"])
            
            # Save the selected joke to the standard location
            import json
            script_data = {
                "script": clean_script,
                "thumbnail_title": f"Hilarious {jokes[selected]['topic']} Jokes! 🤣💻"
            }
            
            os.makedirs("cache/scripts", exist_ok=True)
            with open(f"cache/scripts/tech_humor_latest.json", "w", encoding="utf-8") as f:
                json.dump(script_data, f, ensure_ascii=False, indent=2)
            
            print(colored(f"✓ Selected joke saved to cache/scripts/tech_humor_latest.json", "green"))
            print(colored(f"✓ Script cleaned for TTS generation", "green"))
            
            # Set the script for video generation
            script = clean_script
        else:
            print(colored("Invalid selection", "red"))
            return
    else:
        # Random topic (default)
        import random
        topic = random.choice(TECH_TOPICS)
        print(colored(f"Selected random topic: {topic}", "cyan"))
        
        # Generate improved tech humor script
        print(colored(f"\nGenerating improved tech humor script for topic: {topic}", "blue"))
        success, script = await generate_improved_tech_humor(topic)
        
        if not success or not script:
            print(colored("✗ Failed to generate script", "red"))
            return
        
        print(colored("\nGenerated Script:", "green"))
        print(colored(script, "cyan"))
    
    # Ask if user wants to generate a video
    generate_video = input("\nGenerate video with this script? (y/n): ").strip().lower()
    
    if generate_video == 'y':
        # Create video with the generated script
        print(colored("\nGenerating video with the improved script...", "blue"))
        
        # Initialize video generator
        video_gen = VideoGenerator()
        
        # Create video
        success = await video_gen.create_video(
            script_file="cache/scripts/tech_humor_latest.json",
            channel_type="tech_humor"
        )
        
        if success:
            print(colored("\n✓ Video generated successfully with improved humor!", "green"))
            print(colored("Check output/videos/tech_humor_latest.mp4", "cyan"))
        else:
            print(colored("\n✗ Failed to generate video", "red"))

if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(script_dir))  # Change to parent directory
    
    # Run the main function
    asyncio.run(main()) 