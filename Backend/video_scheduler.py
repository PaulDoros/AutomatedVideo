import asyncio
import pytz
from datetime import datetime, timedelta
import random
from termcolor import colored
import os
import json
from video_database import VideoDatabase

class VideoScheduler:
    def __init__(self):
        """Initialize the video scheduler"""
        self.db = VideoDatabase()
        self.channels = [
            'tech_humor', 
            'ai_money', 
            'baby_tips', 
            'quick_meals', 
            'fitness_motivation'
        ]
        
        # Default timezone (UTC)
        self.default_timezone = pytz.timezone('UTC')
        
        # Channel-specific timezones for global audience targeting
        self.channel_timezones = {
            'tech_humor': pytz.timezone('US/Pacific'),      # Tech audience in US West Coast
            'ai_money': pytz.timezone('US/Eastern'),        # Business audience in US East Coast
            'baby_tips': pytz.timezone('Europe/London'),    # Parents in Europe
            'quick_meals': pytz.timezone('Asia/Tokyo'),     # Food content for Asian audience
            'fitness_motivation': pytz.timezone('Australia/Sydney')  # Fitness for Australia/Asia
        }
        
        # Peak hours for each channel (in their local timezone)
        self.peak_hours = {
            'tech_humor': [9, 12, 15, 18, 20, 22],          # Tech audience active times
            'ai_money': [7, 10, 13, 16, 19, 21],            # Business hours + evening
            'baby_tips': [6, 9, 12, 15, 18, 21],            # Parent schedule (morning to evening)
            'quick_meals': [7, 10, 13, 16, 19, 22],         # Meal times + planning times
            'fitness_motivation': [5, 8, 12, 15, 18, 20]    # Workout times (early morning, lunch, evening)
        }
        
        # Load performance data if available
        self.performance_data_path = "Backend/data/performance_data.json"
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self):
        """Load performance data from file if it exists"""
        if os.path.exists(self.performance_data_path):
            try:
                with open(self.performance_data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(colored(f"Error loading performance data: {str(e)}", "red"))
        
        # Default performance data
        return {channel: {} for channel in self.channels}
    
    def _save_performance_data(self):
        """Save performance data to file"""
        os.makedirs(os.path.dirname(self.performance_data_path), exist_ok=True)
        
        try:
            with open(self.performance_data_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            print(colored(f"Error saving performance data: {str(e)}", "red"))
    
    def _get_optimal_hours(self, channel_type):
        """Get optimal posting hours for a channel based on performance data"""
        # If we have performance data, use it to adjust peak hours
        if channel_type in self.performance_data and self.performance_data[channel_type]:
            # Extract hour performance data
            hour_performance = self.performance_data[channel_type].get('hour_performance', {})
            
            if hour_performance:
                # Convert to list of (hour, performance) tuples and sort by performance
                sorted_hours = sorted(
                    [(int(hour), perf) for hour, perf in hour_performance.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Take top 6 performing hours
                return [hour for hour, _ in sorted_hours[:6]]
        
        # Fall back to default peak hours
        return self.peak_hours[channel_type]
    
    def _convert_to_utc(self, local_dt, timezone):
        """Convert local datetime to UTC"""
        # Check if the datetime is already timezone aware
        if local_dt.tzinfo is None:
            local_dt = timezone.localize(local_dt)
        else:
            # If it's already timezone aware, ensure it's in the correct timezone
            local_dt = local_dt.astimezone(timezone)
        return local_dt.astimezone(pytz.UTC)
    
    def _get_next_posting_times(self, channel_type, days=1):
        """Get next posting times for a channel"""
        optimal_hours = self._get_optimal_hours(channel_type)
        timezone = self.channel_timezones[channel_type]
        
        # Get current time in channel's timezone
        now_utc = datetime.now(pytz.UTC)
        now_local = now_utc.astimezone(timezone)
        
        posting_times = []
        
        # Generate posting times for the specified number of days
        for day in range(days):
            # Create a datetime for each optimal hour
            for hour in optimal_hours:
                # Create local datetime (timezone naive)
                post_time_local_naive = datetime(
                    now_local.year,
                    now_local.month,
                    now_local.day,
                    hour,
                    random.randint(0, 59)  # Random minute for natural distribution
                ) + timedelta(days=day)
                
                # Make it timezone aware
                post_time_local = timezone.localize(post_time_local_naive)
                
                # Skip times in the past
                if post_time_local <= now_local:
                    continue
                
                # Convert to UTC for storage
                post_time_utc = post_time_local.astimezone(pytz.UTC)
                
                posting_times.append({
                    'channel_type': channel_type,
                    'scheduled_time': post_time_utc,
                    'local_time': post_time_local
                })
        
        return posting_times
    
    def generate_schedule(self, days=7):
        """Generate a posting schedule for all channels for the specified number of days"""
        print(colored(f"\nGenerating posting schedule for {days} days...", "blue"))
        
        all_posting_times = []
        
        # Generate posting times for each channel
        for channel in self.channels:
            channel_times = self._get_next_posting_times(channel, days)
            all_posting_times.extend(channel_times)
            print(colored(f"✓ Generated {len(channel_times)} posting times for {channel}", "green"))
        
        # Sort by scheduled time
        all_posting_times.sort(key=lambda x: x['scheduled_time'])
        
        # Add to database
        added_count = 0
        for post_time in all_posting_times:
            success = self.db.add_to_schedule(
                post_time['channel_type'],
                post_time['scheduled_time']
            )
            if success:
                added_count += 1
        
        print(colored(f"✓ Added {added_count} posting times to schedule", "green"))
        return all_posting_times
    
    def get_upcoming_schedule(self, hours=24):
        """Get upcoming scheduled uploads"""
        return self.db.get_upcoming_schedule(hours)
    
    def update_performance_data(self, channel_type, hour, performance_score):
        """Update performance data for a specific hour"""
        if channel_type not in self.performance_data:
            self.performance_data[channel_type] = {}
        
        if 'hour_performance' not in self.performance_data[channel_type]:
            self.performance_data[channel_type]['hour_performance'] = {}
        
        hour_str = str(hour)
        
        # If we already have data for this hour, update with moving average
        if hour_str in self.performance_data[channel_type]['hour_performance']:
            current_score = self.performance_data[channel_type]['hour_performance'][hour_str]
            # 70% weight to historical data, 30% to new data
            new_score = (current_score * 0.7) + (performance_score * 0.3)
            self.performance_data[channel_type]['hour_performance'][hour_str] = new_score
        else:
            # First data point for this hour
            self.performance_data[channel_type]['hour_performance'][hour_str] = performance_score
        
        # Save updated performance data
        self._save_performance_data()
    
    def calculate_performance_score(self, metrics):
        """Calculate a performance score from metrics"""
        # Weighted score based on various metrics
        # You can adjust weights based on what's most important
        score = (
            (metrics.get('views', 0) * 0.4) +
            (metrics.get('likes', 0) * 0.3) +
            (metrics.get('comments', 0) * 0.2) +
            (metrics.get('shares', 0) * 0.1)
        )
        return score
    
    def update_schedule_from_analytics(self):
        """Update schedule based on analytics data"""
        print(colored("\nUpdating schedule based on analytics...", "blue"))
        
        # For each channel, get top performing content
        for channel in self.channels:
            top_content = self.db.get_top_performing_content(channel, limit=20)
            
            if not top_content:
                print(colored(f"No performance data available for {channel}", "yellow"))
                continue
            
            # Extract upload hours and calculate performance
            hour_performance = {}
            
            for content in top_content:
                # Get video details
                video_id = content['video_id']
                
                # Get upload time
                self.db.connect()
                self.db.cursor.execute('''
                SELECT upload_date FROM videos WHERE video_id = ?
                ''', (video_id,))
                result = self.db.cursor.fetchone()
                self.db.disconnect()
                
                if not result:
                    continue
                
                # Parse upload time
                try:
                    upload_time = datetime.fromisoformat(result[0])
                    
                    # Convert to channel's timezone
                    timezone = self.channel_timezones[channel]
                    local_time = upload_time.replace(tzinfo=pytz.UTC).astimezone(timezone)
                    
                    # Extract hour
                    hour = local_time.hour
                    
                    # Calculate performance score
                    performance_score = self.calculate_performance_score(content)
                    
                    # Update hour performance
                    if hour not in hour_performance:
                        hour_performance[hour] = []
                    
                    hour_performance[hour].append(performance_score)
                except Exception as e:
                    print(colored(f"Error processing upload time: {str(e)}", "red"))
                    continue
            
            # Calculate average performance for each hour
            for hour, scores in hour_performance.items():
                avg_score = sum(scores) / len(scores)
                self.update_performance_data(channel, hour, avg_score)
            
            print(colored(f"✓ Updated performance data for {channel}", "green"))
        
        # Regenerate schedule with updated performance data
        self.generate_schedule()
        
        print(colored("✓ Schedule updated based on analytics", "green"))
    
    def print_schedule(self, schedule):
        """Print the schedule in a readable format"""
        if not schedule:
            print(colored("No upcoming scheduled uploads", "yellow"))
            return
        
        print(colored("\nUpcoming scheduled uploads:", "blue"))
        
        for item in schedule:
            channel = item['channel_type']
            scheduled_time_utc = datetime.fromisoformat(item['scheduled_time'])
            
            # Convert to channel's timezone
            timezone = self.channel_timezones[channel]
            local_time = scheduled_time_utc.replace(tzinfo=pytz.UTC).astimezone(timezone)
            
            status = item['status']
            video_id = item['video_id'] or 'Not assigned'
            
            print(colored(f"Channel: {channel}", "cyan"))
            print(f"  Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})")
            print(f"  UTC time: {scheduled_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Status: {status}")
            print(f"  Video ID: {video_id}")
            print()

async def test_scheduler():
    """Test the scheduler functionality"""
    scheduler = VideoScheduler()
    
    # Generate a schedule for the next 7 days
    schedule = scheduler.generate_schedule(days=7)
    
    # Get upcoming schedule for the next 24 hours
    upcoming = scheduler.get_upcoming_schedule(hours=24)
    
    # Print the upcoming schedule
    scheduler.print_schedule(upcoming)
    
    # Simulate updating performance data
    for channel in scheduler.channels:
        for hour in range(24):
            # Random performance score for testing
            score = random.uniform(0, 100)
            scheduler.update_performance_data(channel, hour, score)
    
    # Update schedule based on simulated analytics
    scheduler.update_schedule_from_analytics()
    
    # Get updated upcoming schedule
    updated_upcoming = scheduler.get_upcoming_schedule(hours=24)
    
    # Print the updated upcoming schedule
    scheduler.print_schedule(updated_upcoming)

if __name__ == "__main__":
    asyncio.run(test_scheduler()) 