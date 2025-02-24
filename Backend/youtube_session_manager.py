from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from termcolor import colored
import os
import json
import pickle
import time
from datetime import datetime, timedelta

class YouTubeSessionManager:
    def __init__(self):
        # Define quota limits first
        self.DAILY_QUOTA_LIMIT = 10000  # YouTube's daily quota limit
        self.UPLOAD_COST = 1600  # Cost per video upload
        self.THUMBNAIL_COST = 50  # Cost per thumbnail update
        
        # Then initialize paths and scopes
        self.credentials_file = 'Backend/client_secret.json'
        self.token_dir = 'Backend/tokens'
        self.quota_file = f"{self.token_dir}/quotas.json"
        self.scopes = [
            'https://www.googleapis.com/auth/youtube.upload',
            'https://www.googleapis.com/auth/youtube'
        ]
        
        # Create tokens directory if it doesn't exist
        os.makedirs(self.token_dir, exist_ok=True)
        
        # Initialize quotas last
        self.quotas = self.load_quotas()

    def load_quotas(self):
        """Load quota information from file"""
        if os.path.exists(self.quota_file):
            try:
                with open(self.quota_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.initialize_quotas()

    def initialize_quotas(self):
        """Initialize quota tracking for all channels"""
        quotas = {
            'last_reset': datetime.now().strftime('%Y-%m-%d'),
            'channels': {
                'tech_humor': {'used': 0, 'remaining': self.DAILY_QUOTA_LIMIT},
                'ai_money': {'used': 0, 'remaining': self.DAILY_QUOTA_LIMIT},
                'baby_tips': {'used': 0, 'remaining': self.DAILY_QUOTA_LIMIT},
                'quick_meals': {'used': 0, 'remaining': self.DAILY_QUOTA_LIMIT},
                'fitness_motivation': {'used': 0, 'remaining': self.DAILY_QUOTA_LIMIT}
            }
        }
        self.save_quotas(quotas)
        return quotas

    def save_quotas(self, quotas):
        """Save quota information to file"""
        with open(self.quota_file, 'w') as f:
            json.dump(quotas, f, indent=2)

    def check_and_reset_quotas(self):
        """Reset quotas if it's a new day"""
        last_reset = datetime.strptime(self.quotas['last_reset'], '%Y-%m-%d')
        if datetime.now().date() > last_reset.date():
            self.quotas = self.initialize_quotas()

    def has_quota_available(self, channel_type, operation='upload'):
        """Check if channel has enough quota for operation"""
        self.check_and_reset_quotas()
        cost = self.UPLOAD_COST if operation == 'upload' else self.THUMBNAIL_COST
        return self.quotas['channels'][channel_type]['remaining'] >= cost

    def update_quota(self, channel_type, operation='upload'):
        """Update quota usage for channel"""
        cost = self.UPLOAD_COST if operation == 'upload' else self.THUMBNAIL_COST
        self.quotas['channels'][channel_type]['used'] += cost
        self.quotas['channels'][channel_type]['remaining'] -= cost
        self.save_quotas(self.quotas)

    def get_service(self, channel_type):
        """Get authenticated YouTube service with persistent credentials"""
        token_file = f"{self.token_dir}/token_{channel_type}.pickle"
        token_info_file = f"{self.token_dir}/token_info_{channel_type}.json"
        
        creds = None
        token_info = {}
        
        # Load token info if exists
        if os.path.exists(token_info_file):
            with open(token_info_file, 'r') as f:
                token_info = json.load(f)
        
        # Check if token will expire soon (within 1 hour)
        token_expires_soon = token_info.get('expires_at', 0) < time.time() + 3600
        
        # Load existing credentials if they exist
        if os.path.exists(token_file) and not token_expires_soon:
            try:
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)
            except Exception as e:
                print(colored(f"Error loading credentials: {e}", "red"))

        # If no valid credentials available, refresh or get new ones
        if not creds or not creds.valid or token_expires_soon:
            if creds and creds.expired and creds.refresh_token:
                print(colored(f"Refreshing credentials for {channel_type}...", "yellow"))
                creds.refresh(Request())
            else:
                print(colored(f"Getting new credentials for {channel_type}...", "yellow"))
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)
            
            # Update token info
            token_info = {
                'expires_at': time.time() + creds.expiry.timestamp(),
                'refresh_token': creds.refresh_token is not None,
                'last_refresh': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(token_info_file, 'w') as f:
                json.dump(token_info, f, indent=2)

        return build('youtube', 'v3', credentials=creds)

    def initialize_all_sessions(self):
        """Initialize sessions for all channels"""
        channel_types = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
        services = {}
        
        print(colored("\nInitializing YouTube sessions...", "blue"))
        for channel_type in channel_types:
            try:
                if self.has_quota_available(channel_type):
                    services[channel_type] = self.get_service(channel_type)
                    print(colored(f"✓ Initialized session for {channel_type}", "green"))
                    print(colored(f"  Remaining quota: {self.quotas['channels'][channel_type]['remaining']}", "cyan"))
                else:
                    print(colored(f"✗ No quota available for {channel_type}", "red"))
            except Exception as e:
                print(colored(f"✗ Failed to initialize {channel_type}: {e}", "red"))
        
        return services 