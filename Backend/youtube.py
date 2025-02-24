import os
import sys
import time
import random
import httplib2
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from termcolor import colored
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
from oauth2client.client import flow_from_clientsecrets
from dotenv import load_dotenv

# Explicitly tell the underlying HTTP transport library not to retry, since
# we are handling retry logic ourselves.
httplib2.RETRIES = 1

# Maximum number of times to retry before giving up.
MAX_RETRIES = 10

# Always retry when these exceptions are raised.
RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, httplib2.ServerNotFoundError)

# Always retry when an apiclient.errors.HttpError with one of these status
# codes is raised.
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "./client_secret.json"

# This OAuth 2.0 access scope allows an application to upload files to the
# authenticated user's YouTube channel, but doesn't allow other types of access.
# YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
SCOPES = ['https://www.googleapis.com/auth/youtube.upload',
          'https://www.googleapis.com/auth/youtube',
          'https://www.googleapis.com/auth/youtubepartner']
YOUTUBE_API_SERVICE_NAME = "youtube"  
YOUTUBE_API_VERSION = "v3"  

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
MISSING_CLIENT_SECRETS_MESSAGE = f"""
WARNING: Please configure OAuth 2.0

To make this sample run you will need to populate the client_secrets.json file
found at:
  
{os.path.abspath(os.path.join(os.path.dirname(__file__), CLIENT_SECRETS_FILE))}

with information from the API Console
https://console.cloud.google.com/

For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
"""

VALID_PRIVACY_STATUSES = ("public", "private", "unlisted")  
  
  
def get_authenticated_service(channel='main'):
    """Get credentials and create an API client."""
    # Use the same client_secret.json for all channels
    client_secrets_file = 'client_secret.json'
    
    # Create unique token file for each channel
    token_file = f'token_{channel}.pickle'
    
    credentials = None
    # Check if we have valid token for this channel
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            credentials = pickle.load(token)

    # If no valid credentials, let user login with desired channel
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES)
            print(colored(f"[!] Please login with your {channel} channel...", "yellow"))
            credentials = flow.run_local_server(port=0)
        
        # Save credentials for this channel
        with open(token_file, 'wb') as token:
            pickle.dump(credentials, token)

    return build('youtube', 'v3', credentials=credentials)

def initialize_upload(youtube: any, options: dict):
    """
    This method uploads a video to YouTube.

    Args:
        youtube (any): The authenticated YouTube service.
        options (dict): The options to upload the video with.

    Returns:
        response: The response from the upload process.
    """

    tags = None
    if options['keywords']:
        tags = options['keywords'].split(",")

    body = {
        'snippet': {
            'title': options['title'],
            'description': options['description'],
            'tags': tags,
            'categoryId': options['category']
        },
        'status': {
            'privacyStatus': options['privacyStatus'],
            'madeForKids': False,  # Video is not made for kids
            'selfDeclaredMadeForKids': False  # You declare that the video is not made for kids
        }
    }

    # Call the API's videos.insert method to create and upload the video.
    insert_request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=MediaFileUpload(options['file'], chunksize=-1, resumable=True)
    )

    return resumable_upload(insert_request)

def resumable_upload(insert_request: MediaFileUpload):
    """
    This method implements an exponential backoff strategy to resume a  
    failed upload.

    Args:
        insert_request (MediaFileUpload): The request to insert the video.

    Returns:
        response: The response from the upload process.
    """
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            print(colored(" => Uploading file...", "magenta"))
            status, response = insert_request.next_chunk()
            if 'id' in response:
                print(f"Video id '{response['id']}' was successfully uploaded.")
                return response
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"A retriable HTTP error {e.resp.status} occurred:\n{e.content}"
            else:
                raise
        except RETRIABLE_EXCEPTIONS as e:
            error = f"A retriable error occurred: {e}"

        if error is not None:
            print(colored(error, "red"))
            retry += 1
            if retry > MAX_RETRIES:
                raise Exception("No longer attempting to retry.")

            max_sleep = 2 ** retry 
            sleep_seconds = random.random() * max_sleep
            print(colored(f" => Sleeping {sleep_seconds} seconds and then retrying...", "blue"))
            time.sleep(sleep_seconds)  
  
def upload_video(video_path, title, description, category, keywords, privacy_status, channel='main'):
    """Upload video to YouTube."""
    try:
        # Get the authenticated YouTube service for specific channel
        youtube = get_authenticated_service(channel)

        # Print which channel we're uploading to
        channels_response = youtube.channels().list(mine=True, part='snippet').execute()
        channel_title = channels_response['items'][0]['snippet']['title']
        print(colored(f"[+] Uploading to YouTube channel: {channel_title}", "blue"))

        # Initialize the upload process
        video_response = initialize_upload(youtube, {
            'file': video_path,
            'title': title,
            'description': description,
            'category': category,
            'keywords': keywords,
            'privacyStatus': privacy_status
        })
        return video_response
    except HttpError as e:
        print(colored(f"[-] An HTTP error {e.resp.status} occurred:\n{e.content}", "red"))
        raise e 

def test_channel_connection(channel_name):
    """Test connection to a YouTube channel"""
    try:
        print("\n=== Channel Connection Test ===")
        
        # Get the Backend directory path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        client_secret_path = os.path.join(backend_dir, 'client_secret.json')
        
        if not os.path.exists(client_secret_path):
            print(colored(f"[-] client_secret.json not found at: {client_secret_path}", "red"))
            print(colored("Please copy client_secret.json to the Backend directory", "yellow"))
            return False
            
        # OAuth 2.0 credentials
        SCOPES = [
            'https://www.googleapis.com/auth/youtube.upload',
            'https://www.googleapis.com/auth/youtube',
            'https://www.googleapis.com/auth/youtubepartner'
        ]
        
        # Create credentials flow
        flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
        
        # Get credentials
        print(colored(f"[!] Please login with your {channel_name} channel...", "yellow"))
        credentials = flow.run_local_server(port=0)
        
        # Build YouTube service
        youtube = build('youtube', 'v3', credentials=credentials)
        
        # Get channel info
        request = youtube.channels().list(
            part="snippet,statistics",
            mine=True
        )
        response = request.execute()
        
        if response['items']:
            channel = response['items'][0]
            print(colored(f"✓ Successfully connected to: {channel_name}", "green"))
            print(colored(f"Channel Title: {channel['snippet']['title']}", "blue"))
            print(colored(f"Subscriber Count: {channel['statistics']['subscriberCount']}", "blue"))
            print(colored(f"Video Count: {channel['statistics']['videoCount']}", "blue"))
            print(colored(f"Total Views: {channel['statistics']['viewCount']}", "blue"))
            return True
        else:
            print(colored(f"✗ Failed to get channel info for: {channel_name}", "red"))
            return False
            
    except Exception as e:
        print(colored(f"✗ Failed to connect to: {channel_name}", "red"))
        print(colored(f"Error: {str(e)}", "red"))
        return False
    finally:
        print("===========================\n")

def test_all_channels():
    """Test connection to all configured YouTube channels."""
    channels = ['main', 'business', 'gaming', 'tech']
    results = {}
    
    print(colored("\nTesting YouTube Channel Connections...", "blue"))
    
    for channel in channels:
        results[channel] = test_channel_connection(channel)
    
    print(colored("\nSummary:", "blue"))
    for channel, success in results.items():
        status = "✓" if success else "✗"
        color = "green" if success else "red"
        print(colored(f"{status} {channel.title()} Channel", color))
    
    return all(results.values()) 
