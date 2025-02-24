from video_generator import generate_video_for_channel
from youtube import upload_video
from content_manager import CHANNEL_STRATEGIES
from termcolor import colored
import os
from dotenv import load_dotenv
import schedule
import time

class YouTubeContentManager:
    def __init__(self):
        load_dotenv()
        self.channels = {
            'tech_humor': {  # LOLRoboJAJA
                'name': os.getenv('YOUTUBE_CHANNEL_NAME_MAIN', 'LOLRoboJAJA'),
                'env_key': 'YOUTUBE_CHANNEL_MAIN',
                'strategy_key': 'tech_humor',
                'upload_times': os.getenv('UPLOAD_TIMES_TECH_HUMOR', '12:00,18:00').split(',')
            },
            'ai_money': {  # AI Money Hacks
                'name': os.getenv('YOUTUBE_CHANNEL_NAME_SECOND', 'AI Money Hacks'),
                'env_key': 'YOUTUBE_CHANNEL_SECOND',
                'strategy_key': 'ai_money',
                'upload_times': os.getenv('UPLOAD_TIMES_AI_MONEY', '10:00,16:00').split(',')
            },
            'baby_tips': {  # Smart Parenting Tips
                'name': os.getenv('YOUTUBE_CHANNEL_NAME_PARENTING', 'Smart Parenting Tips'),
                'env_key': 'YOUTUBE_CHANNEL_PARENTING',
                'strategy_key': 'baby_tips',
                'upload_times': os.getenv('UPLOAD_TIMES_BABY_TIPS', '09:00,15:00').split(',')
            },
            'quick_meals': {  # Quick & Healthy Meals
                'name': os.getenv('YOUTUBE_CHANNEL_NAME_MEALS', 'Quick & Healthy Meals'),
                'env_key': 'YOUTUBE_CHANNEL_MEALS',
                'strategy_key': 'quick_meals',
                'upload_times': os.getenv('UPLOAD_TIMES_QUICK_MEALS', '11:00,17:00').split(',')
            },
            'fitness_motivation': {  # Daily Fitness Motivation
                'name': os.getenv('YOUTUBE_CHANNEL_NAME_FITNESS', 'Daily Fitness Motivation'),
                'env_key': 'YOUTUBE_CHANNEL_FITNESS',
                'strategy_key': 'fitness_motivation',
                'upload_times': os.getenv('UPLOAD_TIMES_FITNESS', '08:00,14:00').split(',')
            }
        }

    def create_and_upload_video(self, channel_key):
        """Create and upload a video for a specific channel"""
        try:
            channel = self.channels[channel_key]
            strategy = CHANNEL_STRATEGIES[channel['strategy_key']]
            
            print(colored(f"\n=== Creating video for {channel['name']} ===", "blue"))
            
            # Generate video using channel strategy
            video_data = generate_video_for_channel(
                channel=channel['strategy_key'],
                topic=strategy['description'],
                hashtags=strategy['hashtags']
            )
            
            if video_data and video_data.get('video_path'):
                # Upload to YouTube
                response = upload_video(
                    video_path=video_data['video_path'],
                    title=video_data['title'],
                    description=f"{video_data['description']}\n\n#{' #'.join(strategy['hashtags'])}",
                    category="28",  # Education
                    keywords=",".join(strategy['hashtags']),
                    privacy_status="public",
                    channel=channel['env_key']
                )
                
                if response:
                    print(colored(f"✓ Successfully uploaded to {channel['name']}", "green"))
                    return True
            
            return False
            
        except Exception as e:
            print(colored(f"✗ Error creating/uploading video: {str(e)}", "red"))
            return False

    def create_videos_for_all_channels(self):
        """Create and upload videos for all channels"""
        results = {}
        
        print(colored("\n=== Starting Video Creation for All Channels ===", "blue"))
        print(colored("Channels to process:", "yellow"))
        for channel_key in self.channels:
            print(colored(f"- {self.channels[channel_key]['name']}", "yellow"))
        
        for channel_key in self.channels.keys():
            print(colored(f"\nProcessing {self.channels[channel_key]['name']}", "blue"))
            results[channel_key] = self.create_and_upload_video(channel_key)
        
        # Print summary
        print(colored("\n=== Upload Summary ===", "blue"))
        for channel_key, success in results.items():
            status = "✓" if success else "✗"
            color = "green" if success else "red"
            print(colored(f"{status} {self.channels[channel_key]['name']}", color))

def main():
    manager = YouTubeContentManager()
    manager.create_videos_for_all_channels()

if __name__ == "__main__":
    main() 