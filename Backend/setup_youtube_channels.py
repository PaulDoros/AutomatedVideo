from youtube import test_channel_connection
from termcolor import colored
import os
import json
from dotenv import load_dotenv

def setup_youtube_channels():
    """Setup new YouTube channels using existing credentials"""
    # Get the Backend directory path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    client_secret_path = os.path.join(backend_dir, 'client_secret.json')
    
    if not os.path.exists(client_secret_path):
        print(colored(f"[-] Error: client_secret.json not found in {backend_dir}", "red"))
        print(colored("Please make sure client_secret.json is in the Backend directory", "yellow"))
        return
    
    new_channels = [
        {
            'name': 'Smart Parenting Tips',
            'env_key': 'YOUTUBE_CHANNEL_PARENTING',
            'description': 'Daily tips and advice for new parents',
            'keywords': 'parenting,baby,newborn,tips,advice'
        },
        {
            'name': 'Quick & Healthy Meals',
            'env_key': 'YOUTUBE_CHANNEL_MEALS',
            'description': '15-minute healthy meal recipes and ideas',
            'keywords': 'recipes,cooking,healthy,quick meals,food'
        },
        {
            'name': 'Daily Fitness Motivation',
            'env_key': 'YOUTUBE_CHANNEL_FITNESS',
            'description': 'Quick workouts and fitness motivation',
            'keywords': 'fitness,workout,exercise,motivation,health'
        }
    ]
    
    print(colored("\n=== Setting Up Additional YouTube Channels ===", "blue"))
    print(colored(f"Using existing client_secret.json from: {client_secret_path}", "green"))
    
    for channel in new_channels:
        print(colored(f"\nSetting up: {channel['name']}", "blue"))
        print(colored("1. Make sure you've created this channel on YouTube", "yellow"))
        print(colored("2. You'll need to authorize this app for the new channel", "yellow"))
        input(colored("Press Enter when ready to connect...", "green"))
        
        # Update .env file to use the relative path to client_secret.json
        update_env_file(channel['env_key'], 'Backend/client_secret.json')
        update_env_file(f"{channel['env_key']}_NAME", channel['name'])
        
        # Test connection
        print(colored("\nTesting connection...", "blue"))
        test_channel_connection(channel['name'].lower().replace(' ', '_'))
        
        print(colored("\nChannel setup complete!", "green"))
        input(colored("Press Enter to continue to next channel...", "yellow"))
    
    print(colored("\nAll channels have been set up!", "green"))
    print(colored("\nSummary of all channels:", "blue"))
    print(colored("- LOLRoboJAJA (Tech Humor) - Existing", "green"))
    print(colored("- AI Money Hacks - Existing", "green"))
    for channel in new_channels:
        print(colored(f"- {channel['name']} - New", "green"))

def update_env_file(key, value):
    """Update .env file with new channel"""
    # Get the root directory (where .env is located)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, '.env')
    
    # Read existing content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Check if key exists
    key_exists = False
    for i, line in enumerate(lines):
        if line.startswith(key + '='):
            lines[i] = f'{key}="{value}"\n'
            key_exists = True
            break
    
    # Add new key if it doesn't exist
    if not key_exists:
        lines.append(f'\n{key}="{value}"\n')
    
    # Write back to file
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    print(colored(f"[+] Updated {env_path} with {key}", "green"))

if __name__ == "__main__":
    setup_youtube_channels() 