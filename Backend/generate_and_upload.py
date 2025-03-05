import asyncio
from content_validator import ScriptGenerator, ContentValidator
from video_generator import VideoGenerator
from thumbnail_generator import ThumbnailGenerator
from youtube_uploader import YouTubeUploader
from video_scheduler import VideoScheduler
from content_monitor import ContentMonitor
from performance_analyzer import PerformanceAnalyzer
from video_database import VideoDatabase
from termcolor import colored
import os
import json
import argparse
from datetime import datetime, timedelta
import random
import nltk
from joke_provider import JokeProvider

# Download wordnet if not already downloaded
nltk.download('wordnet')

async def generate_content(channel_type, topic=None, music_volume=0.3, no_upload=False):
    """Generate content for a specific channel"""
    print(colored(f"\n=== Generating content for {channel_type} ===", "blue"))
    
    # Initialize components
    script_gen = ScriptGenerator()
    video_gen = VideoGenerator()
    thumb_gen = ThumbnailGenerator()
    content_monitor = ContentMonitor()
    content_validator = ContentValidator()
    
    # Set music volume for better audibility
    video_gen.music_volume = music_volume
    print(colored(f"Setting music volume to: {video_gen.music_volume} for better audibility", "cyan"))
    
    # Check if topic is a duplicate
    if topic and content_monitor.is_topic_duplicate(channel_type, topic):
        print(colored(f"✗ Topic '{topic}' is a duplicate for {channel_type}", "yellow"))
        
        # Suggest alternative topic
        suggestion = content_monitor.suggest_alternative_topic(channel_type, topic)
        
        if suggestion:
            print(colored(f"Suggestion: {suggestion}", "cyan"))
            
            # Extract alternative topic from suggestion
            alt_topic = suggestion.split("Try '")[1].split("'")[0]
            topic = alt_topic
            
            print(colored(f"Using alternative topic: {topic}", "green"))
    
    try:
        # For tech_humor channel, use pre-written jokes or DeepSeek API
        if channel_type == "tech_humor":
            print(colored("\nUsing JokeProvider for tech_humor channel...", "blue"))
            joke_provider = JokeProvider()
            
            # Check if we should use AI-generated jokes
            # If we've used all pre-written jokes or explicitly requested AI
            use_ai = False
            if os.getenv("USE_AI_JOKES", "false").lower() == "true":
                use_ai = True
                print(colored("Using AI-generated jokes as requested in environment variables", "cyan"))
            
            # Get a joke (either pre-written or AI-generated)
            script = await joke_provider.get_joke(topic, use_ai)
            
            # Validate the joke script
            is_valid, analysis = content_validator.validate_script(script, channel_type)
            if not is_valid:
                print(colored(f"✗ Joke validation failed: {analysis.get('message', 'Unknown error')}", "yellow"))
                # Try another joke if the first one fails validation
                script = await joke_provider.get_joke(topic, True)  # Force AI generation for second attempt
                is_valid, analysis = content_validator.validate_script(script, channel_type)
                if not is_valid:
                    print(colored(f"✗ Alternative joke validation failed: {analysis.get('message', 'Unknown error')}", "red"))
                    return None
            
            # Save the script to a file for the video generator
            script_file = f"cache/scripts/{channel_type}_latest.json"
            os.makedirs(os.path.dirname(script_file), exist_ok=True)
            
            # Create a title from the first line of the joke
            title = script.split('\n')[0]
            
            # Save script data
            script_data = {
                "script": script,
                "title": title,
                "channel_type": channel_type,
                "topic": topic or "tech humor",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(script_file, 'w', encoding='utf-8') as f:
                json.dump(script_data, f, indent=2)
                
            print(colored("✓ Joke selected and saved", "green"))
            success = True
        else:
            # For other channels, generate script as usual
            print(colored("\nGenerating script...", "blue"))
            success, script = await script_gen.generate_script(
                topic=topic,
                channel_type=channel_type
            )
            
        if not success:
            print(colored(f"✗ Script generation failed for {channel_type}", "red"))
            return None
        print(colored("✓ Script generated", "green"))
        
        # Check if script content is a duplicate
        if content_monitor.is_content_duplicate(channel_type, script):
            print(colored(f"✗ Generated script is too similar to existing content", "yellow"))
            return None
        
        # 2. Generate Video
        print(colored("\nGenerating video...", "blue"))
        script_file = f"cache/scripts/{channel_type}_latest.json"
        
        # Create channel-specific output directory
        channel_output_dir = f"output/videos/{channel_type}"
        os.makedirs(channel_output_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{channel_type}_{timestamp}.mp4"
        output_path = os.path.join(channel_output_dir, output_filename)
        
        success = await video_gen.create_video(
            script_file=script_file,
            channel_type=channel_type,
            output_path=output_path
        )
        if not success:
            print(colored(f"✗ Video generation failed for {channel_type}", "red"))
            return None
        print(colored("✓ Video generated", "green"))
        print(colored(f"✓ Video saved to: {output_path}", "cyan"))
        
        # 3. Generate Thumbnail
        print(colored("\nGenerating thumbnail...", "blue"))
        thumb_gen.generate_test_thumbnails()
        print(colored("✓ Thumbnail generated", "green"))
        
        # Return video details
        video_details = {
            'channel_type': channel_type,
            'topic': topic,
            'script': script,
            'file_path': output_path,
            'thumbnail_path': f"test_thumbnails/{channel_type}.jpg"
        }
        
        return video_details
        
    except Exception as e:
        print(colored(f"✗ Error generating content for {channel_type}: {str(e)}", "red"))
        return None

async def upload_content(video_details, schedule_id=None):
    """Upload content to YouTube"""
    if not video_details:
        return False, "No video details provided"
    
    channel_type = video_details['channel_type']
    file_path = video_details['file_path']
    topic = video_details['topic']
    
    print(colored(f"\n=== Uploading content for {channel_type} ===", "blue"))
    
    # Initialize uploader
    uploader = YouTubeUploader()
    db = VideoDatabase()
    
    try:
        # Get thumbnail title from cache if available
        thumbnail_title = ""
        cache_file = f"cache/scripts/{channel_type}_latest.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    thumbnail_title = cached.get('thumbnail_title', '')
                    if thumbnail_title:
                        print(colored(f"Using thumbnail title as video title: {thumbnail_title}", "cyan"))
            except Exception as e:
                print(colored(f"Warning: Could not load thumbnail title from cache: {str(e)}", "yellow"))
        
        # Use thumbnail title if available, otherwise use topic
        video_title = thumbnail_title if thumbnail_title else topic
        
        # Upload to YouTube
        print(colored("\nUploading to YouTube...", "blue"))
        success, result = await uploader.upload_video(
            channel_type=channel_type,
            video_path=file_path,
            title=video_title,
            description=None,  # Will use professional description in uploader
            tags=[channel_type, 'shorts', 'youtube', 'content']
        )
        
        if success:
            print(colored(f"✓ Upload complete for {channel_type}", "green"))
            print(colored(f"Video ID: {result}", "cyan"))
            
            # If this is a scheduled upload, update the schedule
            if schedule_id:
                db.assign_video_to_schedule(schedule_id, result)
                db.update_schedule_status(schedule_id, 'completed')
            
            return True, result
        else:
            print(colored(f"✗ Upload failed for {channel_type}: {result}", "red"))
            
            # If this is a scheduled upload, update the schedule
            if schedule_id:
                db.update_schedule_status(schedule_id, 'failed')
            
            return False, result
            
    except Exception as e:
        error_msg = f"Error uploading content for {channel_type}: {str(e)}"
        print(colored(f"✗ {error_msg}", "red"))
        
        # If this is a scheduled upload, update the schedule
        if schedule_id:
            db.update_schedule_status(schedule_id, 'failed')
        
        return False, error_msg

async def process_scheduled_uploads(hours=1):
    """Process scheduled uploads for the next X hours"""
    print(colored(f"\n=== Processing scheduled uploads for next {hours} hours ===", "blue"))
    
    # Initialize components
    scheduler = VideoScheduler()
    db = VideoDatabase()
    
    # Get upcoming schedules
    upcoming = scheduler.get_upcoming_schedule(hours)
    
    if not upcoming:
        print(colored("No upcoming scheduled uploads", "yellow"))
        return []
    
    results = []
    
    for schedule in upcoming:
        schedule_id = schedule['id']
        channel_type = schedule['channel_type']
        scheduled_time = schedule['scheduled_time']
        status = schedule['status']
        
        print(colored(f"\nProcessing schedule: {channel_type} at {scheduled_time} (Status: {status})", "blue"))
        
        # Skip if not pending
        if status != 'pending':
            print(colored(f"Skipping: Status is {status}", "yellow"))
            continue
        
        # Generate content
        video_details = await generate_content(channel_type)
        
        if not video_details:
            print(colored(f"✗ Failed to generate content for {channel_type}", "red"))
            db.update_schedule_status(schedule_id, 'failed')
            
            results.append({
                'schedule_id': schedule_id,
                'channel_type': channel_type,
                'scheduled_time': scheduled_time,
                'success': False,
                'message': "Content generation failed"
            })
            
            continue
        
        # Assign video to schedule
        db.update_schedule_status(schedule_id, 'ready')
        
        # Upload content
        success, result = await upload_content(video_details, schedule_id)
        
        results.append({
            'schedule_id': schedule_id,
            'channel_type': channel_type,
            'scheduled_time': scheduled_time,
            'success': success,
            'message': result if not success else "Upload successful"
        })
    
    print(colored(f"\n=== Completed processing {len(results)} scheduled uploads ===", "green"))
    return results

async def analyze_performance():
    """Analyze performance of all channels"""
    print(colored("\n=== Analyzing channel performance ===", "blue"))
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    # Analyze all channels
    report_files = await analyzer.analyze_all_channels(days=30)
    
    if report_files:
        print(colored("\nGenerated performance reports:", "green"))
        for report_file in report_files:
            print(colored(f"- {report_file}", "cyan"))
    else:
        print(colored("\nNo performance reports generated", "yellow"))
    
    print(colored("\n=== Performance analysis complete ===", "green"))
    return report_files

async def monitor_content():
    """Monitor content across all channels"""
    print(colored("\n=== Monitoring channel content ===", "blue"))
    
    # Initialize monitor
    monitor = ContentMonitor()
    
    # Store all channel videos
    await monitor.store_all_channel_videos()
    
    print(colored("\n=== Content monitoring complete ===", "green"))

async def generate_schedule(days=7):
    """Generate a posting schedule for all channels"""
    print(colored(f"\n=== Generating posting schedule for {days} days ===", "blue"))
    
    # Initialize scheduler
    scheduler = VideoScheduler()
    
    # Generate schedule
    schedule = scheduler.generate_schedule(days=days)
    
    print(colored(f"\n=== Generated schedule with {len(schedule)} posting times ===", "green"))
    return schedule

async def generate_and_upload(music_volume=0.3, no_upload=False):
    """Generate and upload content for all channels"""
    
    # Create necessary directories
    os.makedirs("output/videos", exist_ok=True)
    os.makedirs("test_thumbnails", exist_ok=True)
    os.makedirs("cache/scripts", exist_ok=True)
    
    # Channel topics
    topics = {
        'tech_humor': 'When Your Code Works But You Don\'t Know Why',
        'ai_money': 'Make $100/Day with ChatGPT Automation',
        'baby_tips': 'Help Your Baby Sleep Through the Night',
        'quick_meals': '5-Minute Healthy Breakfast Ideas',
        'fitness_motivation': '10-Minute Morning Workout Routine'
    }
    
    for channel, topic in topics.items():
        print(colored(f"\n=== Processing {channel}: {topic} ===", "blue"))
        
        # Generate content
        video_details = await generate_content(channel, topic, music_volume, no_upload)
        
        if not video_details:
            continue
        
        # Upload content if not disabled
        if not no_upload:
            await upload_content(video_details)
        else:
            print(colored("Skipping upload as requested with --no-upload", "yellow"))
        
        print(colored(f"\n=== Completed {channel} ===", "green"))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate and upload content')
    
    # Action options
    parser.add_argument('--generate', action='store_true', help='Generate content')
    parser.add_argument('--schedule', type=int, help='Generate posting schedule for the next N days')
    parser.add_argument('--process', type=int, help='Process scheduled uploads for the next N hours')
    parser.add_argument('--analyze', action='store_true', help='Analyze performance of all channels')
    parser.add_argument('--monitor', action='store_true', help='Monitor content across all channels')
    parser.add_argument('--cleanup', action='store_true', help='Clean up video library')
    
    # Generation options
    parser.add_argument('--channel', type=str, help='Channel to generate content for')
    parser.add_argument('--topic', type=str, help='Topic to generate content about')
    parser.add_argument('--music-volume', type=float, default=0.2, help='Volume of background music (0.0 to 1.0)')
    
    # Cleanup options
    parser.add_argument('--max-videos', type=int, default=20, help='Maximum number of videos to keep per channel during cleanup')
    parser.add_argument('--days-to-keep', type=int, default=30, help='Keep videos newer than this many days during cleanup')
    
    # Upload options
    parser.add_argument('--no-upload', action='store_true', help='Generate video without uploading')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize database
    db = VideoDatabase()
    
    # Initialize YouTube uploader
    uploader = YouTubeUploader()
    
    # Initialize sessions for all channels
    CHANNELS = ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']
    
    if args.generate:
        # Generate content for a specific channel or all channels
        if args.channel:
            asyncio.run(generate_content(args.channel, args.topic, args.music_volume, args.no_upload))
        else:
            asyncio.run(generate_and_upload(args.music_volume, args.no_upload))
    
    elif args.schedule:
        # Generate posting schedule for the next N days
        scheduler = VideoScheduler(db)
        scheduler.generate_schedule(args.schedule)
        
    elif args.process:
        # Process scheduled uploads for the next N hours
        asyncio.run(process_scheduled_uploads(args.process))
        
    elif args.analyze:
        # Analyze performance of all channels
        analyzer = PerformanceAnalyzer(db)
        analyzer.analyze_all_channels()
        
    elif args.monitor:
        # Monitor content across all channels
        monitor = ContentMonitor(db)
        monitor.check_all_channels()
        
    elif args.cleanup:
        # Clean up video library
        print(colored(f"Cleaning up video library (max {args.max_videos} videos per channel, keeping videos newer than {args.days_to_keep} days)", "blue"))
        video_gen = VideoGenerator()
        for channel in CHANNELS:
            video_gen.cleanup_video_library(channel, max_videos=args.max_videos, days_to_keep=args.days_to_keep)
        print(colored("Video library cleanup complete", "green"))
        
    else:
        print("No action specified. Use --generate, --schedule, --process, --analyze, --monitor, or --cleanup")

if __name__ == "__main__":
    main() 