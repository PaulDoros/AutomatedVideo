from youtube import test_channel_connection
from tiktok_upload import TikTokUploader
from termcolor import colored
import os
from dotenv import load_dotenv

def test_setup():
    load_dotenv()
    
    print(colored("\n=== Testing YouTube Channels ===", "blue"))
    
    # Test LOLRoboJAJA channel
    print(colored("\nTesting LOLRoboJAJA channel:", "blue"))
    test_channel_connection('main')
    
    # Test AI Money Hacks channel
    print(colored("\nTesting AI Money Hacks channel:", "blue"))
    test_channel_connection('second')
    
    print(colored("\n=== Testing TikTok Account ===", "blue"))
    tiktok_session = os.getenv('TIKTOK_SESSION_ID')
    if tiktok_session:
        uploader = TikTokUploader(tiktok_session)
        # Test connection
        print(colored("✓ TikTok session ID found", "green"))
        print(colored(f"Account: {os.getenv('TIKTOK_ACCOUNT_NAME')}", "blue"))
    else:
        print(colored("✗ TikTok session ID missing", "red"))

if __name__ == "__main__":
    test_setup() 