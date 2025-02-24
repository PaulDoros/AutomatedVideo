from youtube_session_manager import YouTubeSessionManager
from googleapiclient.http import MediaFileUpload
from termcolor import colored
import json
import os
from dotenv import load_dotenv

class YouTubeUploader:
    def __init__(self):
        load_dotenv()
        self.channel_credentials = {
            'tech_humor': {
                'channel_id': os.getenv('YOUTUBE_CHANNEL_TECH'),
                'channel_name': os.getenv('YOUTUBE_CHANNEL_TECH_NAME'),
            },
            'ai_money': {
                'channel_id': os.getenv('YOUTUBE_CHANNEL_AI'),
                'channel_name': os.getenv('YOUTUBE_CHANNEL_AI_NAME'),
            },
            'baby_tips': {
                'channel_id': os.getenv('YOUTUBE_CHANNEL_PARENTING'),
                'channel_name': os.getenv('YOUTUBE_CHANNEL_PARENTING_NAME'),
            },
            'quick_meals': {
                'channel_id': os.getenv('YOUTUBE_CHANNEL_MEALS'),
                'channel_name': os.getenv('YOUTUBE_CHANNEL_MEALS_NAME'),
            },
            'fitness_motivation': {
                'channel_id': os.getenv('YOUTUBE_CHANNEL_FITNESS'),
                'channel_name': os.getenv('YOUTUBE_CHANNEL_FITNESS_NAME'),
            }
        }
        
        # Initialize session manager and get services
        session_manager = YouTubeSessionManager()
        self.youtube_services = session_manager.initialize_all_sessions()

    async def upload_video(self, channel_type, video_path, title, description, tags):
        """Upload video to specific YouTube channel"""
        try:
            channel_info = self.channel_credentials[channel_type]
            youtube = self.youtube_services[channel_type]
            
            print(colored(f"\nUploading to channel: {channel_info['channel_name']}", "blue"))
            
            # Load cached script for description
            cache_file = f"cache/scripts/{channel_type}_latest.json"
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                script = cached.get('script', '')

            # Prepare video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': description or script,
                    'tags': tags,
                    'categoryId': '22'  # People & Blogs
                },
                'status': {
                    'privacyStatus': 'private',  # Start as private for review
                    'selfDeclaredMadeForKids': False
                }
            }

            # Upload video file
            media = MediaFileUpload(
                video_path,
                mimetype='video/mp4',
                resumable=True
            )

            # Execute upload
            upload_request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )

            response = upload_request.execute()
            video_id = response['id']
            print(colored(f"✓ Video uploaded successfully to {channel_info['channel_name']}!", "green"))
            print(colored(f"Video ID: {video_id}", "cyan"))

            # Set thumbnail
            thumbnail_path = f"test_thumbnails/{channel_type}.jpg"
            if os.path.exists(thumbnail_path):
                self.set_thumbnail(youtube, video_id, thumbnail_path)

            return True, video_id

        except Exception as e:
            print(colored(f"✗ Error uploading to {channel_type}: {str(e)}", "red"))
            return False, str(e)

    def set_thumbnail(self, youtube, video_id, thumbnail_path):
        """Set custom thumbnail for video"""
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path)
            ).execute()
            print(colored("✓ Thumbnail set successfully", "green"))
        except Exception as e:
            print(colored(f"✗ Error setting thumbnail: {str(e)}", "red")) 