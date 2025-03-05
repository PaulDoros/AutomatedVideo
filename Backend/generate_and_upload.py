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
from content_learning_system import ContentLearningSystem

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
    learning_system = ContentLearningSystem()
    
    # Set music volume for better audibility
    video_gen.music_volume = music_volume
    print(colored(f"Setting music volume to: {video_gen.music_volume} for better audibility", "cyan"))
    
    # Check if topic is a duplicate using both systems
    is_duplicate = False
    if topic:
        # Check with content monitor
        if content_monitor.is_topic_duplicate(channel_type, topic):
            print(colored(f"✗ Topic '{topic}' is a duplicate for {channel_type}", "yellow"))
            is_duplicate = True
        
        # Check with learning system
        is_repetitive, details = learning_system.is_content_repetitive(channel_type, topic)
        if is_repetitive:
            print(colored(f"✗ Topic '{topic}' is too similar to recent content: {details['message']}", "yellow"))
            is_duplicate = True
        
        # Check if blacklisted
        is_blacklisted, reason = learning_system.is_content_blacklisted(channel_type, topic)
        if is_blacklisted:
            print(colored(f"✗ Topic '{topic}' is blacklisted: {reason}", "yellow"))
            is_duplicate = True
        
        if is_duplicate:
            # Suggest alternative topic
            suggestion = content_monitor.suggest_alternative_topic(channel_type, topic)
            
            if suggestion:
                print(colored(f"Suggestion: {suggestion}", "cyan"))
                
                # Extract alternative topic from suggestion
                alt_topic = suggestion.split("Try '")[1].split("'")[0]
                topic = alt_topic
                
                # Get content suggestions from learning system
                content_suggestions = learning_system.get_content_suggestions(channel_type, topic)
                if content_suggestions:
                    print(colored("Additional suggestions from learning system:", "cyan"))
                    for i, suggestion in enumerate(content_suggestions[:3], 1):
                        print(colored(f"  {i}. {suggestion['message']}", "cyan"))
            
    # For tech_humor channel, use JokeProvider
    if channel_type == 'tech_humor':
        print(colored(f"Using JokeProvider for {channel_type} channel...", "cyan"))
        joke_provider = JokeProvider()
        
        # Get a joke
        joke = await joke_provider.get_joke(topic, use_ai=True)
        
        # Save joke to file for reference
        os.makedirs("cache/scripts", exist_ok=True)
        with open(f"cache/scripts/{channel_type}_latest.txt", "w", encoding="utf-8") as f:
            f.write(joke)
        
        print(colored("✓ Joke selected and saved", "green"))
        
        # Validate joke
        is_valid, analysis = content_validator.validate_script(joke, channel_type)
        
        if not is_valid:
            print(colored(f"✗ Joke validation failed: {analysis['message']}", "red"))
            return None
        
        print(colored("✓ Script generated", "green"))
        
        # Generate video
        print(colored("\nGenerating video...", "blue"))
        video_path = await video_gen.generate_video(channel_type, f"cache/scripts/{channel_type}_latest.txt")
        
        if not video_path:
            print(colored("✗ Video generation failed", "red"))
            return None
        
        print(colored("✓ Video generated", "green"))
        
        # Generate thumbnail
        print(colored("\nGenerating thumbnail...", "blue"))
        thumbnail_path = thumb_gen.generate_thumbnail(channel_type, joke)
        
        if not thumbnail_path:
            print(colored("✗ Thumbnail generation failed", "red"))
            return None
        
        print(colored("✓ Thumbnail generated", "green"))
        
        # Return video details
        return {
            'channel_type': channel_type,
            'video_path': video_path,
            'thumbnail_path': thumbnail_path,
            'title': f"Tech Humor: {joke.splitlines()[0][:50]}",
            'description': joke + "\n\n#tech #programming #humor #shorts",
            'tags': ['tech', 'programming', 'humor', 'shorts', 'coding', 'developer', 'software'],
            'script': joke
        }
    
    # For other channels, generate script with GPT
    else:
        # Get content suggestions from learning system
        content_suggestions = learning_system.get_content_suggestions(channel_type, topic)
        if content_suggestions:
            print(colored("Content suggestions from learning system:", "cyan"))
            for i, suggestion in enumerate(content_suggestions[:3], 1):
                print(colored(f"  {i}. {suggestion['message']}", "cyan"))
        
        # Generate script
        is_valid, script = await script_gen.generate_script(topic, channel_type)
        
        if not is_valid or not script:
            print(colored("✗ Script generation failed", "red"))
            
            # Check if script is repetitive or blacklisted
            if script:
                is_repetitive, details = learning_system.is_content_repetitive(channel_type, script)
                if is_repetitive:
                    print(colored(f"✗ Generated script is too similar to recent content: {details['message']}", "yellow"))
                
                is_blacklisted, reason = learning_system.is_content_blacklisted(channel_type, script)
                if is_blacklisted:
                    print(colored(f"✗ Generated script is blacklisted: {reason}", "yellow"))
            
            return None
        
        # Save script to file for reference
        os.makedirs("cache/scripts", exist_ok=True)
        with open(f"cache/scripts/{channel_type}_latest.txt", "w", encoding="utf-8") as f:
            f.write(script)
        
        print(colored("✓ Script generated and saved", "green"))
        
        # Generate video
        print(colored("\nGenerating video...", "blue"))
        video_path = await video_gen.generate_video(channel_type, f"cache/scripts/{channel_type}_latest.txt")
        
        if not video_path:
            print(colored("✗ Video generation failed", "red"))
            return None
        
        print(colored("✓ Video generated", "green"))
        
        # Generate thumbnail
        print(colored("\nGenerating thumbnail...", "blue"))
        thumbnail_path = thumb_gen.generate_thumbnail(channel_type, script)
        
        if not thumbnail_path:
            print(colored("✗ Thumbnail generation failed", "red"))
            return None
        
        print(colored("✓ Thumbnail generated", "green"))
        
        # Extract title from script (first line)
        title = script.split('\n')[0]
        
        # Return video details
        return {
            'channel_type': channel_type,
            'video_path': video_path,
            'thumbnail_path': thumbnail_path,
            'title': title,
            'description': script + "\n\n#shorts",
            'tags': ['shorts'],
            'script': script
        }

async def upload_content(video_details, schedule_id=None):
    """Upload content to YouTube"""
    if not video_details:
        return False, "No video details provided"
    
    channel_type = video_details['channel_type']
    file_path = video_details['video_path']
    topic = video_details['topic']
    
    print(colored(f"\n=== Uploading content for {channel_type} ===", "blue"))
    
    # Initialize uploader
    uploader = YouTubeUploader()
    db = VideoDatabase()
    
    try:
        # Get thumbnail title from cache if available
        thumbnail_title = ""
        cache_file = f"cache/scripts/{channel_type}_latest.txt"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = f.read()
                    thumbnail_title = cached.split('\n')[0]
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
            description=video_details['description'],
            tags=video_details['tags']
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
    
    # Initialize learning system
    learning_system = ContentLearningSystem()
    
    # Analyze all channels
    report_files = await analyzer.analyze_all_channels(days=30)
    
    # Update learning system with performance data
    await learning_system.update_all_performance_data()
    
    if report_files:
        print(colored("\nGenerated performance reports:", "green"))
        for report_file in report_files:
            print(colored(f"- {report_file}", "cyan"))
    else:
        print(colored("\nNo performance reports generated", "yellow"))
    
    # Calculate content diversity scores
    print(colored("\nContent diversity analysis:", "blue"))
    for channel in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']:
        diversity_score = learning_system.get_content_diversity_score(channel)
        print(colored(f"- {channel}: {diversity_score:.2f}/1.0", "cyan"))
    
    print(colored("\n=== Performance analysis complete ===", "green"))
    return report_files

async def monitor_content():
    """Monitor content across all channels"""
    print(colored("\n=== Monitoring channel content ===", "blue"))
    
    # Initialize monitor
    monitor = ContentMonitor()
    
    # Initialize learning system
    learning_system = ContentLearningSystem()
    
    # Store all channel videos
    await monitor.store_all_channel_videos()
    
    # Analyze channel content with learning system
    print(colored("\nAnalyzing content patterns:", "blue"))
    for channel in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']:
        analysis = learning_system.analyze_channel_content(channel)
        if analysis['status'] == 'success':
            print(colored(f"\n{channel} content analysis:", "cyan"))
            print(colored(f"- Video count: {analysis['video_count']}", "cyan"))
            print(colored(f"- Average views: {analysis['avg_metrics']['views']:.0f}", "cyan"))
            print(colored(f"- Content clusters: {len(analysis['content_clusters'])}", "cyan"))
            print(colored(f"- Top keywords: {', '.join([kw for kw, _ in analysis['top_keywords'][:5]])}", "cyan"))
    
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
    
    # Initialize learning system
    learning_system = ContentLearningSystem()
    
    # Channel topics
    topics = {
        'tech_humor': 'When Your Code Works But You Don\'t Know Why',
        'ai_money': 'Make $100/Day with ChatGPT Automation',
        'baby_tips': 'Help Your Baby Sleep Through the Night',
        'quick_meals': '5-Minute Healthy Breakfast Ideas',
        'fitness_motivation': '10-Minute Morning Workout Routine'
    }
    
    # Get content suggestions for each channel
    for channel in topics.keys():
        suggestions = learning_system.get_content_suggestions(channel)
        if suggestions:
            # Find a high-performing suggestion
            for suggestion in suggestions:
                if suggestion['type'] == 'content' and suggestion.get('score', 0) > 0.7:
                    print(colored(f"Using high-performing topic for {channel} based on learning system", "cyan"))
                    topics[channel] = suggestion['title']
                    break
    
    for channel, topic in topics.items():
        print(colored(f"\n=== Processing {channel}: {topic} ===", "blue"))
        
        # Generate content
        video_details = await generate_content(channel, topic, music_volume, no_upload)
        
        if not video_details:
            continue
        
        # Upload content if not disabled
        if not no_upload:
            upload_result = await upload_content(video_details)
            
            # Record performance data if upload was successful
            if upload_result and 'video_id' in upload_result:
                # Initialize with zero metrics
                initial_metrics = {
                    'video_id': upload_result['video_id'],
                    'views': 0,
                    'likes': 0,
                    'comments': 0,
                    'ctr': 0.0
                }
                
                # Record in learning system
                learning_system.record_content_performance(
                    upload_result['video_id'],
                    channel,
                    video_details['script'],
                    initial_metrics
                )
                
                # For tech_humor, update joke performance
                if channel == 'tech_humor':
                    joke_provider = JokeProvider()
                    joke_provider.update_joke_performance(video_details['script'], initial_metrics)
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

async def main():
    """Main function"""
    args = parse_args()
    
    # Initialize learning system
    learning_system = ContentLearningSystem()
    
    # Process command line arguments
    if args.generate:
        if args.channel:
            # Generate for specific channel
            await generate_content(args.channel, args.topic, args.music_volume, args.no_upload)
        else:
            # Generate for all channels
            await generate_and_upload(args.music_volume, args.no_upload)
    elif args.schedule:
        # Generate posting schedule
        await generate_schedule(args.schedule)
    elif args.process:
        # Process scheduled uploads
        await process_scheduled_uploads(args.process)
    elif args.analyze:
        # Analyze performance
        await analyze_performance()
    elif args.monitor:
        # Monitor content
        await monitor_content()
    elif args.cleanup:
        # Clean up video library
        print(colored(f"Cleaning up video library (max {args.max_videos} videos per channel, keeping videos newer than {args.days_to_keep} days)", "blue"))
        video_gen = VideoGenerator()
        for channel in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']:
            video_gen.cleanup_video_library(channel, max_videos=args.max_videos, days_to_keep=args.days_to_keep)
        print(colored("Video library cleanup complete", "green"))
    else:
        # Show help
        print("No action specified. Use --help to see available options.")
        
    # Print content diversity scores
    print(colored("\nContent diversity scores:", "blue"))
    for channel in ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation']:
        diversity_score = learning_system.get_content_diversity_score(channel)
        print(colored(f"- {channel}: {diversity_score:.2f}/1.0", "cyan"))

if __name__ == "__main__":
    asyncio.run(main()) 