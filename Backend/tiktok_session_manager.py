from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from termcolor import colored
import time
import json
import os
from dotenv import load_dotenv

class TikTokSessionManager:
    def __init__(self):
        self.session_file = 'tiktok_session.json'
        
    def get_new_session(self):
        """Get new TikTok session using browser automation"""
        try:
            print(colored("\n=== Getting new TikTok session ===", "blue"))
            print(colored("1. Browser will open", "yellow"))
            print(colored("2. Please log in to TikTok", "yellow"))
            print(colored("3. After login, session will be saved automatically", "yellow"))
            
            # Configure Chrome options
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--window-size=1920,1080')
            
            # Use undetected-chromedriver to avoid detection
            driver = uc.Chrome(options=options)
            
            # Go to TikTok
            driver.get('https://www.tiktok.com/login')
            
            # Wait for login to complete (wait for /foryou page)
            WebDriverWait(driver, 300).until(
                lambda driver: 'foryou' in driver.current_url
            )
            
            # Get all cookies
            cookies = driver.get_cookies()
            
            # Extract session data
            session_data = {
                'sessionid': '',
                'csrf_token': '',
                'cookies': {}
            }
            
            for cookie in cookies:
                if cookie['name'] == 'sessionid':
                    session_data['sessionid'] = cookie['value']
                elif cookie['name'] == 'tt_csrf_token':
                    session_data['csrf_token'] = cookie['value']
                session_data['cookies'][cookie['name']] = cookie['value']
            
            # Save session data
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=4)
            
            # Update .env file
            self.update_env_session(session_data['sessionid'])
            
            print(colored("\n✓ Successfully saved new TikTok session!", "green"))
            
            driver.quit()
            return session_data
            
        except Exception as e:
            print(colored(f"\n✗ Error getting new session: {str(e)}", "red"))
            if 'driver' in locals():
                driver.quit()
            return None
    
    def update_env_session(self, session_id):
        """Update session ID in .env file"""
        try:
            # Read existing .env content
            with open('.env', 'r') as f:
                lines = f.readlines()
            
            # Update or add TIKTOK_SESSION_ID
            session_updated = False
            for i, line in enumerate(lines):
                if line.startswith('TIKTOK_SESSION_ID='):
                    lines[i] = f'TIKTOK_SESSION_ID="{session_id}"\n'
                    session_updated = True
                    break
            
            if not session_updated:
                lines.append(f'\nTIKTOK_SESSION_ID="{session_id}"\n')
            
            # Write back to .env
            with open('.env', 'r+') as f:
                f.writelines(lines)
            
            print(colored("✓ Updated session ID in .env file", "green"))
            
        except Exception as e:
            print(colored(f"✗ Error updating .env: {str(e)}", "red"))
    
    def verify_session(self, session_data):
        """Verify if session is valid"""
        try:
            import requests
            
            headers = {
                'authority': 'www.tiktok.com',
                'accept': 'application/json, text/plain, */*',
                'cookie': '; '.join([f'{k}={v}' for k,v in session_data['cookies'].items()]),
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get('https://www.tiktok.com/api/user/detail/', headers=headers)
            
            return response.status_code == 200 and 'userInfo' in response.json()
            
        except Exception:
            return False
    
    def get_valid_session(self):
        """Get valid session, refresh if needed"""
        try:
            # Check if we have saved session
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                
                # Verify session
                if self.verify_session(session_data):
                    print(colored("✓ Using existing TikTok session", "green"))
                    return session_data
            
            # Get new session if needed
            return self.get_new_session()
            
        except Exception as e:
            print(colored(f"✗ Error getting valid session: {str(e)}", "red"))
            return None
    
    def get_backup_account(self):
        """Create and set up backup account"""
        try:
            print(colored("\n=== Setting up backup TikTok account ===", "blue"))
            print(colored("1. Browser will open", "yellow"))
            print(colored("2. Please create a new TikTok account", "yellow"))
            print(colored("3. Use a different email/phone number", "yellow"))
            print(colored("4. After creation, session will be saved", "yellow"))
            
            # Use existing get_new_session but with different file
            self.session_file = 'tiktok_session_backup.json'
            return self.get_new_session()
            
        except Exception as e:
            print(colored(f"✗ Error creating backup account: {str(e)}", "red"))
            return None 