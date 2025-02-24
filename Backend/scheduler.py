import schedule
import time
import random
from datetime import datetime, timedelta
from termcolor import colored
import json
import os
from video_generator import generate_video_for_channel
from content_manager import ContentManager, CHANNEL_STRATEGIES

# Configuration for each channel
CHANNEL_CONFIG = {
    'main': {  # Tech Humor
        'topics': ['programming jokes', 'tech humor', 'developer life', 'coding fails'],
        'videos_per_day': 2,
        'best_times': ['14:00', '18:00'],  # Best posting times (24h format)
        'hashtags': ['programming', 'techtok', 'coding']
    },
    'business': {
        'topics': ['business tips', 'entrepreneurship', 'startup advice', 'marketing tips'],
        'videos_per_day': 2,
        'best_times': ['10:00', '16:00'],
        'hashtags': ['business', 'entrepreneur', 'marketing']
    },
    'gaming': {
        'topics': ['gaming tips', 'game reviews', 'gaming moments', 'esports'],
        'videos_per_day': 1,
        'best_times': ['20:00'],
        'hashtags': ['gaming', 'gamer', 'videogames']
    },
    'tech': {
        'topics': ['tech news', 'tech reviews', 'tech tips', 'future tech'],
        'videos_per_day': 1,
        'best_times': ['15:00'],
        'hashtags': ['technology', 'technews', 'gadgets']
    }
}

class VideoScheduler:
    def __init__(self):
        self.content_manager = ContentManager()
        self.schedule_file = 'video_schedule.json'

    def generate_daily_schedule(self):
        """Generate next day's video schedule with unique content"""
        tomorrow = datetime.now().date() + timedelta(days=1)
        
        schedule_data = {
            'date': tomorrow.strftime('%Y-%m-%d'),
            'videos': []
        }
        
        for channel, strategy in CHANNEL_STRATEGIES.items():
            # Get unique content for each time slot
            for time_slot in strategy['best_times']:
                video_idea = self.content_manager.get_next_video_idea(channel)
                
                schedule_data['videos'].append({
                    'channel': channel,
                    'content': video_idea['content'],
                    'post_time': time_slot,
                    'status': 'pending',
                    'hashtags': video_idea['strategy']['hashtags'],
                    'voice': video_idea['strategy']['voice']
                })
        
        # Save schedule
        with open(self.schedule_file, 'w') as f:
            json.dump(schedule_data, f, indent=4)
        
        print(colored(f"[+] Generated unique content schedule for {tomorrow}", "green"))

    def process_video_queue(self):
        """Process pending videos with unique content"""
        try:
            with open(self.schedule_file, 'r') as f:
                schedule_data = json.load(f)
            
            current_time = datetime.now().strftime('%H:%M')
            
            for video in schedule_data['videos']:
                if video['status'] == 'pending' and video['post_time'] == current_time:
                    print(colored(f"[+] Creating video for {video['channel']}", "blue"))
                    
                    success = generate_video_for_channel(
                        channel=video['channel'],
                        content=video['content'],
                        hashtags=video['hashtags'],
                        voice=video['voice']
                    )
                    
                    video['status'] = 'completed' if success else 'failed'
                    
                    # Save updated schedule
                    with open(self.schedule_file, 'w') as f:
                        json.dump(schedule_data, f, indent=4)
                    
        except Exception as e:
            print(colored(f"[-] Error processing video queue: {str(e)}", "red"))

def main():
    scheduler = VideoScheduler()
    
    # Generate tomorrow's schedule at 11 PM
    schedule.every().day.at("23:00").do(scheduler.generate_daily_schedule)
    
    # Check video queue every minute
    schedule.every(1).minutes.do(scheduler.process_video_queue)
    
    print(colored("[+] Video scheduler started with unique content generation", "green"))
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 