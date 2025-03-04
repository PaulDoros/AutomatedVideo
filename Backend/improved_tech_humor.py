"""
Improved Tech Humor Prompts

This module contains enhanced prompts for generating better tech humor content.
"""

import os
import json
import re
from typing import Tuple, Optional
import random
from termcolor import colored
from openai import OpenAI

# Enhanced system prompt for tech humor
TECH_HUMOR_SYSTEM_PROMPT = """You are a professional comedy writer who specializes in tech humor with experience writing for Silicon Valley, The IT Crowd, and tech conferences.

Key Capabilities:
- Craft clever, witty tech jokes that land perfectly with both technical and non-technical audiences
- Create unexpected punchlines that subvert expectations
- Use technical accuracy as the foundation for humor (jokes should make sense to real programmers)
- Master comedic timing and misdirection techniques
- Incorporate current tech trends, programming languages, and developer experiences

Style Guidelines:
- Use conversational, natural language that sounds like a real comedian
- Employ clever wordplay and puns that aren't obvious
- Reference programming concepts, tech culture, and industry inside jokes
- Create visual humor through descriptive scenarios
- Build tension and release through technical storytelling
- Use self-deprecating humor about the tech industry

Your jokes should feel like they could be delivered by a stand-up comedian at a tech conference - smart, insightful, and genuinely funny."""

# Enhanced channel prompt for tech humor
def get_improved_tech_humor_prompt(topic):
    """Get an improved prompt for tech humor content"""
    
    # List of example jokes to inspire better humor
    example_jokes = [
        """Why do programmers prefer dark mode? 🌙
        Because light attracts bugs! 🐛
        And they already have enough of those in their code...
        The only debugging tool they really need is a flyswatter! 😂
        Follow for more dev humor! 💻""",
        
        """Programmers don't die, they just... 💀
        Go out of scope! 🔍
        But their legacy code lives on forever,
        Haunting the next developer like a vengeful ghost! 👻
        Like for more coding nightmares! 🧟‍♂️""",
        
        """My code and my relationships have one thing in common... 💔
        They both have commitment issues! 📝
        I keep trying to push changes,
        But all I get are rejection messages! 😭
        Comment your worst commit message! 👇""",
        
        """AI told me it could replace programmers... 🤖
        So I asked it to center a div with CSS! 💻
        Three hours later it was still trying,
        Just like the rest of us humans! 😅
        Share if you've been there! ⚡""",
        
        """Database administrators never get invited to parties... 🎉
        Because they always drop tables! 💥
        One minute everyone's having fun,
        Next minute: "Error 404: Party not found!" 🚫
        Follow for more SQL jokes! 📊"""
    ]
    
    # Randomly select an example joke
    example = random.choice(example_jokes)
    
    base_length_guide = """
    CRITICAL LENGTH REQUIREMENTS:
    - Target length: 20-30 seconds
    - Maximum word count: 50-60 words
    - Aim for 4-5 lines total
    - Each line: 10-15 words maximum
    
    Your script MUST fit these requirements for short-form video.
    """
    
    return f"""
    {base_length_guide}
    
    Create a HILARIOUS and CLEVER tech joke about {topic} for short-form video.
    Focus on ONE strong punchline with perfect setup and delivery!
    
    Style Guide:
    - Start with a relatable tech situation or question
    - Build tension with a clever setup
    - Deliver an unexpected punchline that subverts expectations
    - Use wordplay, puns, or technical irony when possible
    - Keep energy high and delivery snappy
    
    Structure (4-5 lines ONLY):
    1. Hook: Relatable tech situation or question (10-12 words)
    2. Setup: Build the context with technical details (10-15 words)
    3. Tension: Add detail or twist that leads to the joke (10-15 words)
    4. Punchline: The unexpected payoff (8-12 words)
    5. Quick call to action (5-8 words)
    
    Example of GREAT Tech Humor:
    {example}
    
    IMPORTANT: Write the script as plain text without line numbers or labels.
    Just write the actual lines of the joke, one per line, with appropriate emojis.
    
    Focus on ONE strong joke with an unexpected twist!
    Make it clever, technically accurate, and genuinely funny!
    """

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

# Function to generate improved tech humor script
async def generate_improved_tech_humor(topic: str) -> Tuple[bool, Optional[str]]:
    """Generate an improved tech humor script"""
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Get improved prompts
        system_prompt = TECH_HUMOR_SYSTEM_PROMPT
        content_prompt = get_improved_tech_humor_prompt(topic)
        
        # Add request for thumbnail title
        content_prompt += """
        
        AFTER you've completed the script, please also include a separate catchy thumbnail title.
        Place it at the very end after a blank line and prefix it with "THUMBNAIL_TITLE:" 
        The thumbnail title should:
        - Be attention-grabbing (5-7 words max)
        - Include 1-2 relevant emoji
        - NOT be part of the script itself
        """
        
        # Calculate approximate token count for cost estimation
        approx_tokens = len(system_prompt.split()) + len(content_prompt.split())
        
        # Generate script using GPT-4o
        print(colored(f"Using GPT-4o for script generation (~{approx_tokens} tokens)", "yellow"))
        print(colored(f"Estimated cost: ${(approx_tokens/1000) * 0.0025:.6f} for input", "yellow"))
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for better quality
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_prompt}
            ],
            temperature=0.9  # Slightly higher temperature for more creative jokes
        )
        
        # Calculate and log cost
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        prompt_cost = (prompt_tokens / 1000000) * 2.50  # $2.50 per 1M tokens for input
        completion_cost = (completion_tokens / 1000000) * 10.00  # $10.00 per 1M tokens for output
        total_cost = prompt_cost + completion_cost
        
        print(colored(f"\nAPI Usage: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens", "yellow"))
        print(colored(f"API Cost: ${total_cost:.6f}", "yellow"))
        
        # Extract response
        full_response = response.choices[0].message.content.strip()
        
        # Extract the script and thumbnail title separately
        script = full_response
        thumbnail_title = ""
        
        # Check for thumbnail title marker
        if "THUMBNAIL_TITLE:" in full_response:
            parts = full_response.split("THUMBNAIL_TITLE:")
            script = parts[0].strip()
            thumbnail_title = parts[1].strip()
            print(colored(f"\nExtracted Thumbnail Title: \"{thumbnail_title}\"", "cyan"))
        
        # Clean the script for TTS
        clean_script = clean_script_for_tts(script)
        
        # Save script to file
        script_data = {
            "script": clean_script,
            "thumbnail_title": thumbnail_title
        }
        
        os.makedirs("cache/scripts", exist_ok=True)
        with open(f"cache/scripts/tech_humor_latest.json", "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
        
        print(colored(f"✓ Script saved to cache/scripts/tech_humor_latest.json", "green"))
        print(colored(f"✓ Script cleaned for TTS generation", "green"))
        
        return True, script
        
    except Exception as e:
        print(colored(f"✗ Error generating improved tech humor: {str(e)}", "red"))
        return False, None 