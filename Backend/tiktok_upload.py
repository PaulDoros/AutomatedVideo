import os
import time
import requests
import json
from termcolor import colored
from urllib.parse import urlencode
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tiktok_session_manager import TikTokSessionManager

class TikTokAccountManager:
    def __init__(self):
        self.accounts = {}
        
    def add_account(self, account_name, session_id):
        self.accounts[account_name] = TikTokUploader(session_id)
        
    def upload_to_account(self, account_name, video_path, title, tags):
        if account_name in self.accounts:
            return self.accounts[account_name].upload_video(video_path, title, tags)
        else:
            print(colored(f"[-] Account {account_name} not found", "red"))
            return None

class TikTokUploader:
    def __init__(self, session_id=None):
        self.session_manager = TikTokSessionManager()
        
        # Get valid session
        session_data = self.session_manager.get_valid_session()
        if session_data:
            self.session_id = session_data['sessionid']
            self.csrf_token = session_data['csrf_token']
            self.cookies = session_data['cookies']
        else:
            raise Exception("Could not get valid TikTok session")
        
        self.base_url = "https://www.tiktok.com"
        self.headers = {
            'authority': 'www.tiktok.com',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': '; '.join([f'{k}={v}' for k,v in self.cookies.items()]),
            'referer': 'https://www.tiktok.com/',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def verify_session(self):
        """Verify TikTok session is valid"""
        try:
            # Try to access settings page to verify login
            response = requests.get(
                f"{self.base_url}/setting",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200 and 'setting' in response.url:
                print(colored(f"[+] Successfully connected to TikTok account: {os.getenv('TIKTOK_ACCOUNT_NAME')}", "green"))
                return True
            
            print(colored("[-] Invalid TikTok session", "red"))
            print(colored(f"[-] Status code: {response.status_code}", "yellow"))
            print(colored(f"[-] Response URL: {response.url}", "yellow"))
            return False
            
        except Exception as e:
            print(colored(f"[-] Error verifying TikTok session: {str(e)}", "red"))
            return False

    def create_playlist(self, name, description=""):
        """Create a new TikTok playlist/series"""
        try:
            # Updated API endpoint and payload
            data = {
                'type': 1,  # 1 for public playlist
                'title': name.replace('_', ' ').title(),
                'desc': description,
                'owner_id': self.get_user_id(),
                'is_public': True,
                'csrf_token': self.get_csrf_token()
            }
            
            # Updated endpoint
            response = requests.post(
                f"{self.base_url}/api/playlist/create/",
                headers={
                    **self.headers,
                    'Content-Type': 'application/json',
                    'X-Secsdk-Csrf-Token': self.get_csrf_token()
                },
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                print(colored(f"[+] Created playlist: {name}", "green"))
                return response.json().get('playlist_id')
            else:
                print(colored(f"[-] Failed to create playlist: {response.status_code}", "red"))
                print(colored(f"[-] Response: {response.text}", "yellow"))
                return None
                
        except Exception as e:
            print(colored(f"[-] Error creating playlist: {str(e)}", "red"))
            return None

    def get_user_id(self):
        """Get TikTok user ID"""
        try:
            response = requests.get(
                f"{self.base_url}/api/user/detail/",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'userInfo' in data:
                    return data['userInfo'].get('user', {}).get('id')
            return None
            
        except Exception as e:
            print(colored(f"[-] Error getting user ID: {str(e)}", "red"))
            return None

    def upload_video(self, video_path, title, tags, series=None):
        """Upload video to TikTok with series categorization"""
        try:
            print(colored("[+] Starting TikTok upload process...", "blue"))
            
            # Verify session first
            if not self.verify_session():
                return None
            
            # Get series hashtags
            series_tags = []
            if series:
                series_env_key = f"CONTENT_SERIES_{series.upper()}"
                series_hashtags = os.getenv(series_env_key, "")
                if series_hashtags:
                    series_tags = [tag.strip('#') for tag in series_hashtags.split()]
            
            # Combine title with series indicator
            series_indicator = f"[{series.replace('_', ' ').title()}] " if series else ""
            full_title = f"{series_indicator}{title}"
            
            # Combine all hashtags (limit to 15 most relevant)
            all_tags = (tags + series_tags)[:15]
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size > 150 * 1024 * 1024:  # 150MB limit
                print(colored("[-] Video file is too large for TikTok (max 150MB)", "red"))
                return None

            # Upload video using TikTok's web upload endpoint
            upload_url = f"{self.base_url}/upload/"
            
            with open(video_path, 'rb') as video_file:
                files = {
                    'video': ('video.mp4', video_file, 'video/mp4'),
                }
                
                data = {
                    'title': full_title[:150],
                    'description': f"{full_title}\n\n{' '.join(['#' + tag for tag in all_tags])}",
                    'privacy_level': 'public',
                    'allow_comment': '1',
                    'allow_duet': '1',
                    'allow_stitch': '1'
                }
                
                response = requests.post(
                    upload_url,
                    headers=self.headers,
                    files=files,
                    data=data,
                    timeout=300  # 5 minutes timeout for large uploads
                )
                
                if response.status_code == 200:
                    print(colored("[+] Video uploaded successfully to TikTok!", "green"))
                    return response.json()
                else:
                    print(colored(f"[-] Upload failed with status code: {response.status_code}", "red"))
                    if response.text:
                        print(colored(f"[-] Error message: {response.text}", "red"))
                    return None
                
        except Exception as e:
            print(colored(f"[-] Error uploading to TikTok: {str(e)}", "red"))
            return None

    def get_or_create_playlist(self, series_name):
        """Get existing playlist ID or create new one"""
        try:
            # Try to find existing playlist
            playlists = self.get_playlists()
            for playlist in playlists:
                if playlist['name'].lower() == series_name.replace('_', ' ').lower():
                    return playlist['id']
            
            # Create new playlist if not found
            return self.create_playlist(
                name=series_name.replace('_', ' ').title(),
                description=f"Videos about {series_name.replace('_', ' ').title()}"
            )
            
        except Exception as e:
            print(colored(f"[-] Error managing playlist: {str(e)}", "red"))
            return None

    def get_playlists(self):
        """Get list of user's playlists"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/user/playlists",  # Updated endpoint
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(colored(f"[+] Found {len(data.get('playlists', []))} playlists", "green"))
                return data.get('playlists', [])
            else:
                print(colored(f"[-] Failed to get playlists: {response.status_code}", "red"))
                print(colored(f"[-] Response: {response.text}", "yellow"))
            return []
            
        except Exception as e:
            print(colored(f"[-] Error getting playlists: {str(e)}", "red"))
            return []

    def generate_tiktok_tags(self, title, keywords):
        """Generate TikTok-appropriate tags for tech humor content"""
        # Base tags from title and keywords
        base_tags = [word.lower().replace(' ', '') for word in keywords.split(',')]
        
        # Add tech and humor specific viral tags
        tech_humor_tags = [
            'techtok',
            'programming',
            'coding',
            'techhumor',
            'programminghumor',
            'codingmemes',
            'developer',
            'computerscience',
            'tech',
            'fyp',
            'foryou',
            'viral',
            'trending'
        ]
        
        # Combine and limit to 15 tags (TikTok's limit)
        all_tags = list(set(base_tags + tech_humor_tags))[:15]
        return all_tags

    def get_csrf_token(self):
        """Get CSRF token from TikTok"""
        try:
            response = requests.get(
                self.base_url,
                headers=self.headers,
                timeout=10
            )
            
            # Extract CSRF token from cookies
            cookies = response.cookies
            for cookie in cookies:
                if cookie.name == 'tt_csrf_token':
                    return cookie.value
            return None
            
        except Exception as e:
            print(colored(f"[-] Error getting CSRF token: {str(e)}", "red"))
            return None

    def get_user_info(self):
        """Get user info including user ID"""
        try:
            response = requests.get(
                f"{self.base_url}/api/user/info/",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'user' in data:
                    print(colored(f"[+] Got user info for: {data['user'].get('uniqueId')}", "green"))
                    return data['user']
            return None
            
        except Exception as e:
            print(colored(f"[-] Error getting user info: {str(e)}", "red"))
            return None 