from main import generate_video
from youtube import upload_video
from tiktok_upload import TikTokUploader
import os
from termcolor import colored
from content_validator import ContentValidator, ScriptGenerator

class VideoGenerator:
    def __init__(self):
        self.validator = ContentValidator()
        self.script_generator = ScriptGenerator()

    def generate_video_for_channel(self, channel, topic, hashtags):
        """Generate and validate video content"""
        try:
            # Generate and validate script
            success, script = self.script_generator.generate_script(topic, channel)
            if not success:
                print(colored("Failed to generate valid script", "red"))
                return None

            # Validate video length before rendering
            estimated_duration = self.validator.estimate_video_length(script)
            if not self.is_duration_valid(estimated_duration, channel):
                print(colored(f"Estimated duration ({estimated_duration}s) not suitable", "red"))
                return None

            # Generate video content
            video_data = {
                'script': script,
                'title': f"{topic} | {' '.join(hashtags[:3])}",
                'description': self.generate_description(script, hashtags),
                'tags': hashtags,
                'thumbnail_text': self.generate_thumbnail_text(topic)
            }

            # Check content completeness
            is_complete, missing = self.validator.check_content_completeness(video_data)
            if not is_complete:
                print(colored(f"Missing content elements: {', '.join(missing)}", "red"))
                return None

            # Generate the actual video
            video_path = self.render_video(video_data)
            if not video_path:
                return None

            return {
                'video_path': video_path,
                'title': video_data['title'],
                'description': video_data['description'],
                'tags': video_data['tags']
            }

        except Exception as e:
            print(colored(f"Error generating video: {str(e)}", "red"))
            return None

    def is_duration_valid(self, duration, channel):
        # Implementation of is_duration_valid method
        pass

    def generate_description(self, script, hashtags):
        # Implementation of generate_description method
        pass

    def generate_thumbnail_text(self, topic):
        # Implementation of generate_thumbnail_text method
        pass

    def render_video(self, video_data):
        # Implementation of render_video method
        pass

def generate_video_for_channel(channel, topic, hashtags):
    """Generate and upload a video for a specific channel"""
    try:
        # Video generation parameters
        params = {
            'videoSubject': topic,
            'aiModel': 'gpt-4',  # or your preferred model
            'voice': 'en_us_001',
            'paragraphNumber': 1,
            'automateYoutubeUpload': True,
            'automateTikTokUpload': True,
            'youtubeAccount': channel,
            'tiktokAccount': channel,
            'useMusic': True,
            'threads': 2
        }
        
        # Generate video
        video_path = generate_video(params)
        
        if video_path:
            # Upload to YouTube
            youtube_response = upload_video(
                video_path=video_path,
                title=f"{topic} #{' #'.join(hashtags)}",
                description=f"Auto-generated content about {topic}\n\n#{' #'.join(hashtags)}",
                category="28",  # Tech
                keywords=",".join(hashtags),
                privacy_status="public",
                channel=channel
            )
            
            # Upload to TikTok
            tiktok_session = os.getenv(f"TIKTOK_SESSION_ID_{channel.upper()}")
            if tiktok_session:
                uploader = TikTokUploader(tiktok_session)
                tiktok_response = uploader.upload_video(
                    video_path=video_path,
                    title=f"{topic} #{' #'.join(hashtags)}"[:150],
                    tags=hashtags
                )
            
            print(colored(f"[+] Successfully generated and uploaded video for {channel}: {topic}", "green"))
            return True
            
    except Exception as e:
        print(colored(f"[-] Error generating video for {channel}: {str(e)}", "red"))
        return False 