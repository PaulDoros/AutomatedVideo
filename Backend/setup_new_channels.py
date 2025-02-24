from youtube import test_channel_connection
from termcolor import colored
import os
from dotenv import load_dotenv

def setup_youtube_channels():
    """Setup new YouTube channels"""
    channels = [
        {
            'name': 'Smart Parenting Tips',
            'description': 'Daily tips and advice for new parents',
            'keywords': 'parenting,baby,newborn,tips,advice'
        },
        {
            'name': 'Quick & Healthy Meals',
            'description': '15-minute healthy meal recipes and ideas',
            'keywords': 'recipes,cooking,healthy,quick meals,food'
        },
        {
            'name': 'Daily Fitness Motivation',
            'description': 'Quick workouts and fitness motivation',
            'keywords': 'fitness,workout,exercise,motivation,health'
        }
    ]
    
    print(colored("\n=== Setting Up New YouTube Channels ===", "blue"))
    print(colored("Please follow these steps for each channel:", "yellow"))
    
    for channel in channels:
        print(colored(f"\nSetting up: {channel['name']}", "blue"))
        print(colored("1. Go to https://console.cloud.google.com/", "yellow"))
        print(colored("2. Create a new project for this channel", "yellow"))
        print(colored("3. Enable YouTube Data API v3", "yellow"))
        print(colored("4. Create OAuth credentials", "yellow"))
        print(colored("5. Download the client_secret.json file", "yellow"))
        print(colored("6. Rename it to match the channel", "yellow"))
        input(colored("Press Enter when ready to test the connection...", "green"))
        
        # Test connection
        test_channel_connection(channel['name'].lower().replace(' ', '_'))

if __name__ == "__main__":
    setup_youtube_channels() 