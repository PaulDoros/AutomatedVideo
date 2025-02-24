from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from termcolor import colored
import os

def get_channel_info():
    """Get YouTube channel ID and name for the authenticated user"""
    
    # OAuth 2.0 credentials
    credentials_file = 'Backend/client_secret.json'
    scopes = ['https://www.googleapis.com/auth/youtube.readonly']
    
    try:
        # Get credentials
        flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
        credentials = flow.run_local_server(port=0)
        
        # Build YouTube service
        youtube = build('youtube', 'v3', credentials=credentials)
        
        # Get channel info
        request = youtube.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True
        )
        response = request.execute()
        
        if response['items']:
            channel = response['items'][0]
            print(colored("\n=== Channel Information ===", "blue"))
            print(colored(f"Channel ID: {channel['id']}", "green"))
            print(colored(f"Channel Name: {channel['snippet']['title']}", "green"))
            print(colored(f"Channel Description: {channel['snippet']['description']}", "cyan"))
            print(colored(f"Subscriber Count: {channel['statistics']['subscriberCount']}", "yellow"))
            print(colored("\nAdd these to your .env file:", "blue"))
            print(colored(f"YOUTUBE_CHANNEL_XXX={channel['id']}", "yellow"))
            print(colored(f"YOUTUBE_CHANNEL_XXX_NAME=\"{channel['snippet']['title']}\"", "yellow"))
            return channel
        else:
            print(colored("No channel found!", "red"))
            return None
            
    except Exception as e:
        print(colored(f"Error getting channel info: {str(e)}", "red"))
        return None

if __name__ == "__main__":
    get_channel_info() 