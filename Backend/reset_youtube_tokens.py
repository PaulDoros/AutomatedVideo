import os
import json
from termcolor import colored

def reset_youtube_tokens():
    """Reset YouTube tokens by removing the token files"""
    token_dir = 'Backend/tokens'
    
    if not os.path.exists(token_dir):
        print(colored(f"Token directory {token_dir} does not exist. Creating it...", "yellow"))
        os.makedirs(token_dir, exist_ok=True)
        print(colored("Token directory created.", "green"))
        return
    
    # List all token files
    token_files = [f for f in os.listdir(token_dir) if f.startswith('token_')]
    
    if not token_files:
        print(colored("No token files found to reset.", "yellow"))
        return
    
    print(colored(f"Found {len(token_files)} token files to reset:", "blue"))
    for file in token_files:
        file_path = os.path.join(token_dir, file)
        try:
            os.remove(file_path)
            print(colored(f"✓ Removed {file}", "green"))
        except Exception as e:
            print(colored(f"✗ Failed to remove {file}: {e}", "red"))
    
    print(colored("\nAll token files have been reset.", "green"))
    print(colored("Next time you run the tests, you will need to re-authenticate with YouTube.", "yellow"))
    print(colored("This will open browser windows for each channel to authorize the application.", "yellow"))

if __name__ == "__main__":
    reset_youtube_tokens() 