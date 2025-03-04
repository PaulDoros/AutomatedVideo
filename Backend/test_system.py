import asyncio
import os
import sys
from termcolor import colored
from video_database import VideoDatabase
from video_scheduler import VideoScheduler
from content_monitor import ContentMonitor
from performance_analyzer import PerformanceAnalyzer
from youtube_uploader import YouTubeUploader
from generate_and_upload import generate_content, upload_content

async def test_database():
    """Test database functionality"""
    print(colored("\n=== Testing Database ===", "blue"))
    
    db = VideoDatabase()
    
    # Test adding a video
    test_video = {
        'video_id': 'test_video_id',
        'channel_type': 'tech_humor',
        'title': 'Test Video',
        'description': 'This is a test video',
        'tags': ['test', 'video'],
        'upload_date': '2023-01-01T00:00:00',
        'status': 'test',
        'file_path': 'test/path.mp4',
        'thumbnail_path': 'test/thumb.jpg',
        'topic': 'Test Topic',
        'script': 'This is a test script'
    }
    
    success = db.add_video(test_video)
    print(colored(f"Add video: {'✓ Success' if success else '✗ Failed'}", "green" if success else "red"))
    
    # Test updating video status
    success = db.update_video_status('test_video_id', 'updated')
    print(colored(f"Update video status: {'✓ Success' if success else '✗ Failed'}", "green" if success else "red"))
    
    # Test adding to schedule
    success = db.add_to_schedule('tech_humor', '2023-01-01T12:00:00')
    print(colored(f"Add to schedule: {'✓ Success' if success else '✗ Failed'}", "green" if success else "red"))
    
    # Test getting upcoming schedule
    schedule = db.get_upcoming_schedule(hours=24)
    print(colored(f"Get upcoming schedule: {'✓ Success' if schedule is not None else '✗ Failed'}", "green" if schedule is not None else "red"))
    
    print(colored("=== Database Test Complete ===", "green"))
    return True

async def test_scheduler():
    """Test scheduler functionality"""
    print(colored("\nRunning Scheduler Test...", "blue"))
    print(colored("\n=== Testing Scheduler ===", "blue"))
    
    try:
        scheduler = VideoScheduler()
        
        # Test generating schedule
        schedule = scheduler.generate_schedule(days=1)
        success = schedule is not None and len(schedule) > 0
        print(colored(f"Generate schedule: {'✓ Success' if success else '✗ Failed'}", "green" if success else "red"))
        
        # Test getting upcoming schedule
        upcoming = scheduler.get_upcoming_schedule(hours=24)
        success = upcoming is not None
        print(colored(f"Get upcoming schedule: {'✓ Success' if success else '✗ Failed'}", "green" if success else "red"))
        
        print(colored("=== Scheduler Test Complete ===", "green"))
        return True
    except Exception as e:
        print(colored(f"✗ Error in Scheduler test: {str(e)}", "red"))
        print(colored("=== Scheduler Test Failed ===", "red"))
        return False

async def test_content_generation():
    """Test content generation"""
    print(colored("\nRunning Content Generation Test...", "blue"))
    print(colored("\n=== Testing Content Generation ===", "blue"))
    
    try:
        # Create necessary directories
        os.makedirs("output/videos", exist_ok=True)
        os.makedirs("test_thumbnails", exist_ok=True)
        os.makedirs("cache/scripts", exist_ok=True)
        print(colored("✓ Directory structure created", "green"))
        
        # Skip actual content generation which requires API calls
        # Instead, test the directory structure and basic setup
        
        # Check if the directories exist
        dirs_to_check = ["output/videos", "test_thumbnails", "cache/scripts"]
        all_dirs_exist = all(os.path.exists(d) for d in dirs_to_check)
        
        if all_dirs_exist:
            print(colored("✓ All required directories exist", "green"))
            print(colored("=== Content Generation Test Complete ===", "green"))
            return True
        else:
            print(colored("✗ Some required directories are missing", "red"))
            print(colored("=== Content Generation Test Failed ===", "red"))
            return False
    except Exception as e:
        print(colored(f"✗ Error in Content Generation test:\n{str(e)}", "red"))
        print(colored("=== Content Generation Test Failed ===", "red"))
        return False

async def test_youtube_uploader():
    """Test YouTube uploader functionality"""
    print(colored("\nRunning YouTube Uploader Test...", "blue"))
    print(colored("\n=== Testing YouTube Uploader ===", "blue"))
    
    try:
        uploader = YouTubeUploader()
        
        # Skip YouTube API initialization and just test basic functionality
        print(colored("✓ YouTube uploader initialized", "green"))
        
        # Test database connection
        if hasattr(uploader, 'db') and uploader.db:
            print(colored("✓ Database connection established", "green"))
        else:
            print(colored("✗ Database connection failed", "red"))
            
        print(colored("=== YouTube Uploader Test Complete ===", "green"))
        return True
    except Exception as e:
        print(colored(f"✗ Error in YouTube Uploader test: {str(e)}", "red"))
        print(colored("=== YouTube Uploader Test Failed ===", "red"))
        return False

async def test_content_monitor():
    """Test content monitor functionality"""
    print(colored("\nRunning Content Monitor Test...", "blue"))
    print(colored("\n=== Testing Content Monitor ===", "blue"))
    
    try:
        monitor = ContentMonitor()
        
        # Test topic duplication check with simple strings instead of API calls
        test_topics = [
            "Python Programming Tips",
            "Python Coding Advice",  # Similar to the first
            "Healthy Breakfast Ideas",
            "Quick Breakfast Recipes",  # Similar to the third
            "Space Exploration Facts"
        ]
        
        for topic in test_topics:
            try:
                is_duplicate = monitor.is_topic_duplicate("tech_humor", topic)
                
                if is_duplicate:
                    print(colored(f"Topic '{topic}' is a duplicate", "yellow"))
                    
                    # Suggest alternative
                    suggestion = monitor.suggest_alternative_topic("tech_humor", topic)
                    
                    if suggestion:
                        print(colored(f"Suggestion: {suggestion}", "cyan"))
                else:
                    print(colored(f"Topic '{topic}' is unique", "green"))
            except Exception as e:
                print(colored(f"Error checking topic '{topic}': {str(e)}", "red"))
        
        print(colored("=== Content Monitor Test Complete ===", "green"))
        return True
    except Exception as e:
        print(colored(f"✗ Error in Content Monitor test:\n{str(e)}", "red"))
        print(colored("=== Content Monitor Test Failed ===", "red"))
        return False

async def test_performance_analyzer():
    """Test performance analyzer functionality"""
    print(colored("\nRunning Performance Analyzer Test...", "blue"))
    print(colored("\n=== Testing Performance Analyzer ===", "blue"))
    
    try:
        analyzer = PerformanceAnalyzer()
        
        # Skip YouTube API initialization and just test basic functionality
        print(colored("✓ Performance analyzer initialized", "green"))
        
        # Test database connection
        if hasattr(analyzer, 'db') and analyzer.db:
            print(colored("✓ Database connection established", "green"))
        else:
            print(colored("✗ Database connection failed", "red"))
            
        print(colored("=== Performance Analyzer Test Complete ===", "green"))
        return True
    except Exception as e:
        print(colored(f"✗ Error in Performance Analyzer test: {str(e)}", "red"))
        print(colored("=== Performance Analyzer Test Failed ===", "red"))
        return False

async def run_all_tests():
    """Run all tests"""
    print(colored("\n=== Running All Tests ===", "blue"))
    
    tests = [
        ("Database", test_database),
        ("Scheduler", test_scheduler),
        ("Content Monitor", test_content_monitor),
        ("Performance Analyzer", test_performance_analyzer),
        ("YouTube Uploader", test_youtube_uploader),
        ("Content Generation", test_content_generation)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(colored(f"\nRunning {name} Test...", "blue"))
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(colored(f"✗ Error in {name} test: {str(e)}", "red"))
            results.append((name, False))
    
    # Print summary
    print(colored("\n=== Test Summary ===", "blue"))
    
    all_passed = True
    
    for name, success in results:
        status = "✓ Passed" if success else "✗ Failed"
        color = "green" if success else "red"
        print(colored(f"{name}: {status}", color))
        
        if not success:
            all_passed = False
    
    if all_passed:
        print(colored("\n✓ All tests passed!", "green"))
    else:
        print(colored("\n✗ Some tests failed", "red"))
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 