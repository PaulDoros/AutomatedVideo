import asyncio
from content_validator import ScriptGenerator
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

async def generate_content(channel_type, topic=None):
    """Generate content for a specific channel"""
    print(colored(f"\n=== Generating content for {channel_type} ===", "blue"))
    
    # Initialize components
    script_gen = ScriptGenerator()
    video_gen = VideoGenerator()
    thumb_gen = ThumbnailGenerator()
    content_monitor = ContentMonitor()
    
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
        # 1. Generate Script
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
        success = await video_gen.create_video(
            script_file=script_file,
            channel_type=channel_type
        )
        if not success:
            print(colored(f"✗ Video generation failed for {channel_type}", "red"))
            return None
        print(colored("✓ Video generated", "green"))
        
        # 3. Generate Thumbnail
        print(colored("\nGenerating thumbnail...", "blue"))
        thumb_gen.generate_test_thumbnails()
        print(colored("✓ Thumbnail generated", "green"))
        
        # Return video details
        video_details = {
            'channel_type': channel_type,
            'topic': topic,
            'script': script,
            'file_path': f"output/videos/{channel_type}_latest.mp4",
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
        # Upload to YouTube
        print(colored("\nUploading to YouTube...", "blue"))
        success, result = await uploader.upload_video(
            channel_type=channel_type,
            video_path=file_path,
            title=topic,
            description=None,  # Will use script as description
            tags=[channel_type, 'shorts', 'content']
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

async def generate_and_upload():
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
        video_details = await generate_content(channel, topic)
        
        if not video_details:
            continue
        
        # Upload content
        await upload_content(video_details)
        
        print(colored(f"\n=== Completed {channel} ===", "green"))

async def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='YouTube Shorts Automation System')
    
    # Add arguments
    parser.add_argument('--generate', action='store_true', help='Generate and upload content for all channels')
    parser.add_argument('--schedule', type=int, default=0, help='Generate posting schedule for X days')
    parser.add_argument('--process', type=int, default=0, help='Process scheduled uploads for next X hours')
    parser.add_argument('--analyze', action='store_true', help='Analyze performance of all channels')
    parser.add_argument('--monitor', action='store_true', help='Monitor content across all channels')
    parser.add_argument('--channel', type=str, help='Specify a single channel to process')
    parser.add_argument('--topic', type=str, help='Specify a topic for content generation')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Process arguments
    if args.generate:
        if args.channel:
            # Generate for a single channel
            topic = args.topic or None
            video_details = await generate_content(args.channel, topic)
            
            if video_details:
                await upload_content(video_details)
        else:
            # Generate for all channels
            await generate_and_upload()
    
    if args.schedule > 0:
        await generate_schedule(days=args.schedule)
    
    if args.process > 0:
        await process_scheduled_uploads(hours=args.process)
    
    if args.analyze:
        await analyze_performance()
    
    if args.monitor:
        await monitor_content()

if __name__ == "__main__":
    asyncio.run(main()) 