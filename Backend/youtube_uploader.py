from youtube_session_manager import YouTubeSessionManager
from googleapiclient.http import MediaFileUpload
from termcolor import colored
import json
import os
import asyncio
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from video_database import VideoDatabase

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
        self.session_manager = YouTubeSessionManager()
        self.youtube_services = self.session_manager.initialize_all_sessions()
        
        # Initialize database
        self.db = VideoDatabase()
        
        # Maximum retry attempts for failed uploads
        self.max_retries = 3
        
        # Delay between retries (in seconds)
        self.retry_delay = 60
        
        # Log directory
        self.log_dir = "Backend/logs"
        os.makedirs(self.log_dir, exist_ok=True)

    async def upload_video(self, channel_type, video_path, title, description, tags, privacy_status='private'):
        """Upload video to specific YouTube channel"""
        try:
            channel_info = self.channel_credentials[channel_type]
            youtube = self.youtube_services[channel_type]
            
            print(colored(f"\nUploading to channel: {channel_info['channel_name']}", "blue"))
            
            # Check if video file exists
            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                print(colored(f"✗ {error_msg}", "red"))
                self._log_error(channel_type, error_msg)
                return False, error_msg
            
            # Load cached script for description
            cache_file = f"cache/scripts/{channel_type}_latest.json"
            script = ""
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                        script = cached.get('script', '')
                except Exception as e:
                    print(colored(f"Warning: Could not load script from cache: {str(e)}", "yellow"))
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': description or script,
                    'tags': tags,
                    'categoryId': '22'  # People & Blogs
                },
                'status': {
                    'privacyStatus': privacy_status,
                    'selfDeclaredMadeForKids': False
                }
            }

            # Upload video file
            media = MediaFileUpload(
                video_path,
                mimetype='video/mp4',
                resumable=True
            )

            # Execute upload with retry logic
            video_id = None
            retry_count = 0
            
            while retry_count < self.max_retries:
                try:
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
                    
                    # Log successful upload
                    self._log_success(channel_type, video_id, title)
                    
                    # Break out of retry loop on success
                    break
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = f"Upload attempt {retry_count} failed: {str(e)}"
                    print(colored(f"✗ {error_msg}", "red"))
                    
                    if retry_count < self.max_retries:
                        wait_time = self.retry_delay * retry_count
                        print(colored(f"Retrying in {wait_time} seconds...", "yellow"))
                        await asyncio.sleep(wait_time)
                    else:
                        error_msg = f"Upload failed after {self.max_retries} attempts: {str(e)}"
                        print(colored(f"✗ {error_msg}", "red"))
                        self._log_error(channel_type, error_msg)
                        return False, error_msg

            # If upload failed after all retries
            if not video_id:
                error_msg = f"Upload failed after {self.max_retries} attempts"
                print(colored(f"✗ {error_msg}", "red"))
                self._log_error(channel_type, error_msg)
                return False, error_msg

            # Set thumbnail
            thumbnail_path = f"test_thumbnails/{channel_type}.jpg"
            if os.path.exists(thumbnail_path):
                thumbnail_result = self.set_thumbnail(youtube, video_id, thumbnail_path)
                if not thumbnail_result:
                    print(colored("✗ Failed to set thumbnail, but video was uploaded", "yellow"))
            
            # Store video in database
            video_data = {
                'video_id': video_id,
                'channel_type': channel_type,
                'title': title,
                'description': description or script,
                'tags': tags,
                'upload_date': datetime.now().isoformat(),
                'status': 'published',
                'file_path': video_path,
                'thumbnail_path': thumbnail_path if os.path.exists(thumbnail_path) else '',
                'topic': title,
                'script': script
            }
            
            self.db.add_video(video_data)

            return True, video_id

        except Exception as e:
            error_msg = f"Error uploading to {channel_type}: {str(e)}"
            print(colored(f"✗ {error_msg}", "red"))
            self._log_error(channel_type, error_msg)
            return False, error_msg

    def set_thumbnail(self, youtube, video_id, thumbnail_path):
        """Set custom thumbnail for video"""
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path)
            ).execute()
            print(colored("✓ Thumbnail set successfully", "green"))
            return True
        except Exception as e:
            print(colored(f"✗ Error setting thumbnail: {str(e)}", "red"))
            return False
    
    def _log_success(self, channel_type, video_id, title):
        """Log successful upload"""
        log_file = os.path.join(self.log_dir, f"upload_success.log")
        
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] SUCCESS: {channel_type} - {video_id} - {title}\n")
    
    def _log_error(self, channel_type, error_msg):
        """Log upload error"""
        log_file = os.path.join(self.log_dir, f"upload_error.log")
        
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] ERROR: {channel_type} - {error_msg}\n")
    
    async def schedule_upload(self, schedule_id):
        """Process a scheduled upload"""
        print(colored(f"\nProcessing scheduled upload (ID: {schedule_id})...", "blue"))
        
        # Get schedule details
        self.db.connect()
        self.db.cursor.execute('''
        SELECT id, channel_type, scheduled_time, video_id
        FROM schedule
        WHERE id = ?
        ''', (schedule_id,))
        
        schedule = self.db.cursor.fetchone()
        self.db.disconnect()
        
        if not schedule:
            print(colored(f"✗ Schedule not found: {schedule_id}", "red"))
            return False
        
        schedule_id, channel_type, scheduled_time, video_id = schedule
        
        # Check if video is already assigned
        if video_id:
            # Get video details
            self.db.connect()
            self.db.cursor.execute('''
            SELECT video_id, title, file_path, description, tags
            FROM videos
            WHERE video_id = ?
            ''', (video_id,))
            
            video = self.db.cursor.fetchone()
            self.db.disconnect()
            
            if not video:
                print(colored(f"✗ Video not found: {video_id}", "red"))
                self.db.update_schedule_status(schedule_id, 'failed')
                return False
            
            video_id, title, file_path, description, tags_json = video
            
            # Parse tags
            try:
                tags = json.loads(tags_json) if tags_json else []
            except:
                tags = []
            
            # Upload video
            success, result = await self.upload_video(
                channel_type=channel_type,
                video_path=file_path,
                title=title,
                description=description,
                tags=tags
            )
            
            if success:
                # Update schedule status
                self.db.update_schedule_status(schedule_id, 'completed')
                print(colored(f"✓ Scheduled upload completed for {channel_type}", "green"))
                return True
            else:
                # Update schedule status
                self.db.update_schedule_status(schedule_id, 'failed')
                print(colored(f"✗ Scheduled upload failed for {channel_type}: {result}", "red"))
                return False
        else:
            # No video assigned yet
            print(colored(f"✗ No video assigned to schedule: {schedule_id}", "yellow"))
            return False
    
    async def process_upcoming_schedules(self, hours=1):
        """Process all upcoming scheduled uploads"""
        print(colored(f"\nProcessing upcoming scheduled uploads for the next {hours} hours...", "blue"))
        
        # Get upcoming schedules
        upcoming = self.db.get_upcoming_schedule(hours)
        
        if not upcoming:
            print(colored("No upcoming scheduled uploads", "yellow"))
            return []
        
        results = []
        
        for schedule in upcoming:
            if schedule['status'] == 'assigned':
                # Process schedule
                success = await self.schedule_upload(schedule['id'])
                
                results.append({
                    'schedule_id': schedule['id'],
                    'channel_type': schedule['channel_type'],
                    'scheduled_time': schedule['scheduled_time'],
                    'success': success
                })
        
        print(colored(f"✓ Processed {len(results)} scheduled uploads", "green"))
        return results
    
    async def update_video_status(self, video_id, privacy_status):
        """Update the privacy status of a video"""
        try:
            # Get video channel type
            self.db.connect()
            self.db.cursor.execute('''
            SELECT channel_type FROM videos WHERE video_id = ?
            ''', (video_id,))
            
            result = self.db.cursor.fetchone()
            self.db.disconnect()
            
            if not result:
                print(colored(f"✗ Video not found: {video_id}", "red"))
                return False
            
            channel_type = result[0]
            youtube = self.youtube_services[channel_type]
            
            # Update video status
            youtube.videos().update(
                part="status",
                body={
                    "id": video_id,
                    "status": {
                        "privacyStatus": privacy_status
                    }
                }
            ).execute()
            
            print(colored(f"✓ Updated video status to {privacy_status}: {video_id}", "green"))
            
            # Update in database
            self.db.update_video_status(video_id, f"status_updated_{privacy_status}")
            
            return True
        
        except Exception as e:
            print(colored(f"✗ Error updating video status: {str(e)}", "red"))
            return False

async def test_uploader():
    """Test the uploader functionality"""
    uploader = YouTubeUploader()
    
    # Test upload
    channel_type = 'tech_humor'
    video_path = 'output/videos/tech_humor_latest.mp4'
    title = 'Test Upload - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if os.path.exists(video_path):
        success, result = await uploader.upload_video(
            channel_type=channel_type,
            video_path=video_path,
            title=title,
            description=None,
            tags=['test', 'upload', 'shorts']
        )
        
        if success:
            print(colored(f"\nTest upload successful: {result}", "green"))
            
            # Test updating video status
            await asyncio.sleep(5)  # Wait a bit before updating
            await uploader.update_video_status(result, 'private')
        else:
            print(colored(f"\nTest upload failed: {result}", "red"))
    else:
        print(colored(f"\nTest video file not found: {video_path}", "yellow"))
    
    # Test scheduled uploads
    results = await uploader.process_upcoming_schedules(hours=24)
    
    if results:
        print(colored("\nScheduled upload results:", "blue"))
        for result in results:
            status = "✓ Success" if result['success'] else "✗ Failed"
            color = "green" if result['success'] else "red"
            print(colored(f"{status}: {result['channel_type']} scheduled for {result['scheduled_time']}", color))

if __name__ == "__main__":
    asyncio.run(test_uploader()) 