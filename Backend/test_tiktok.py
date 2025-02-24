from tiktok_upload import TikTokUploader
from termcolor import colored
import os
from dotenv import load_dotenv

def test_tiktok_connection():
    load_dotenv()
    
    print(colored("\n=== Testing TikTok Connection ===", "blue"))
    
    session_id = os.getenv('TIKTOK_SESSION_ID')
    if not session_id:
        print(colored("[-] TikTok session ID not found in .env", "red"))
        return False
    
    print(colored(f"[+] Using session ID: {session_id[:10]}...", "blue"))
    print(colored(f"[+] Account name: {os.getenv('TIKTOK_ACCOUNT_NAME')}", "blue"))
    
    uploader = TikTokUploader(session_id)
    
    # Get CSRF token first
    csrf_token = uploader.get_csrf_token()
    if csrf_token:
        print(colored(f"[+] Got CSRF token: {csrf_token[:10]}...", "green"))
    else:
        print(colored("[-] Failed to get CSRF token", "red"))
    
    # Get user info
    user_info = uploader.get_user_info()
    if user_info:
        print(colored(f"[+] User ID: {user_info.get('id')}", "green"))
    
    return uploader.verify_session()

def test_create_playlists():
    load_dotenv()
    
    print(colored("\n=== Testing TikTok Playlists ===", "blue"))
    
    session_id = os.getenv('TIKTOK_SESSION_ID')
    if not session_id:
        return False
    
    uploader = TikTokUploader(session_id)
    
    # Test creating playlists for each series
    series = ['tech_humor', 'ai_money', 'ai_tech', 'quick_tips']
    
    for series_name in series:
        playlist_id = uploader.get_or_create_playlist(series_name)
        if playlist_id:
            print(colored(f"✓ Playlist ready: {series_name}", "green"))
        else:
            print(colored(f"✗ Failed to setup playlist: {series_name}", "red"))

if __name__ == "__main__":
    if test_tiktok_connection():
        test_create_playlists() 