import os
from dotenv import load_dotenv
import json
from termcolor import colored

# Constants
MAX_DAILY_UPLOADS_PER_CHANNEL = 10  # Maximum number of videos to upload per channel per day

def setup_channels():
    """Setup all channel configurations"""
    
    # Template for .env additions
    env_template = """
# YouTube Channel Credentials
YOUTUBE_CHANNEL_AI_TECH="client_secret_ai_tech.json"
YOUTUBE_CHANNEL_AI_HUMOR="client_secret_ai_humor.json"
YOUTUBE_CHANNEL_FOOD="client_secret_food.json"
YOUTUBE_CHANNEL_PARENTING="client_secret_parenting.json"

# TikTok Session IDs
TIKTOK_SESSION_ID_AI_TECH="{tiktok_ai_tech}"
TIKTOK_SESSION_ID_AI_HUMOR="{tiktok_ai_humor}"
TIKTOK_SESSION_ID_FOOD="{tiktok_food}"
TIKTOK_SESSION_ID_PARENTING="{tiktok_parenting}"

# Channel Configuration
CHANNEL_AI_TECH_NAME="Future Tech AI"
CHANNEL_AI_HUMOR_NAME="Tech Laughs"
CHANNEL_FOOD_NAME="Quick & Healthy Meals"
CHANNEL_PARENTING_NAME="Modern Parenting Tips"

# Upload Settings
UPLOAD_RETRY_ATTEMPTS=3
CONTENT_LANGUAGE="en"
"""

    try:
        # Load existing .env
        load_dotenv()
        
        # Check for existing credentials
        missing_creds = []
        
        # YouTube credentials check
        youtube_channels = ['AI_TECH', 'AI_HUMOR', 'FOOD', 'PARENTING']
        for channel in youtube_channels:
            cred_file = f"client_secret_{channel.lower()}.json"
            if not os.path.exists(cred_file):
                missing_creds.append(f"YouTube credentials for {channel}")
        
        # TikTok session check
        tiktok_sessions = {
            'AI_TECH': os.getenv('TIKTOK_SESSION_ID_AI_TECH'),
            'AI_HUMOR': os.getenv('TIKTOK_SESSION_ID_AI_HUMOR'),
            'FOOD': os.getenv('TIKTOK_SESSION_ID_FOOD'),
            'PARENTING': os.getenv('TIKTOK_SESSION_ID_PARENTING')
        }
        
        for channel, session in tiktok_sessions.items():
            if not session:
                missing_creds.append(f"TikTok session for {channel}")
        
        if missing_creds:
            print(colored("\nMissing Credentials:", "yellow"))
            for cred in missing_creds:
                print(colored(f"- {cred}", "yellow"))
            
            print(colored("\nPlease add the following to your .env file:", "blue"))
            print(env_template)
        else:
            print(colored("✓ All channel credentials found!", "green"))
        
        # Create channel-specific directories
        for channel in youtube_channels:
            channel_dir = f"content/{channel.lower()}"
            os.makedirs(channel_dir, exist_ok=True)
            print(colored(f"✓ Created directory: {channel_dir}", "green"))
        
        return True
            
    except Exception as e:
        print(colored(f"Error setting up channels: {str(e)}", "red"))
        return False

if __name__ == "__main__":
    setup_channels() 