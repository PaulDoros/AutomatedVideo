from youtube import test_channel_connection
from termcolor import colored

def test_all_channels():
    """Test connection to all configured YouTube channels"""
    channels = [
        {
            'key': 'tech_humor',
            'name': 'LOLRoboJAJA',
            'env_key': 'YOUTUBE_CHANNEL_MAIN'
        },
        {
            'key': 'ai_money',
            'name': 'AI Money Hacks',
            'env_key': 'YOUTUBE_CHANNEL_SECOND'
        },
        {
            'key': 'baby_tips',
            'name': 'Smart Parenting Tips',
            'env_key': 'YOUTUBE_CHANNEL_PARENTING'
        },
        {
            'key': 'quick_meals',
            'name': 'Quick & Healthy Meals',
            'env_key': 'YOUTUBE_CHANNEL_MEALS'
        },
        {
            'key': 'fitness_motivation',
            'name': 'Daily Fitness Motivation',
            'env_key': 'YOUTUBE_CHANNEL_FITNESS'
        }
    ]
    
    results = {}
    
    print(colored("\nTesting YouTube Channel Connections...", "blue"))
    
    for channel in channels:
        print(colored(f"\n=== Testing {channel['name']} ===", "blue"))
        results[channel['key']] = test_channel_connection(channel['key'])
    
    print(colored("\nSummary:", "blue"))
    for channel in channels:
        status = "✓" if results[channel['key']] else "✗"
        color = "green" if results[channel['key']] else "red"
        print(colored(f"{status} {channel['name']}", color))
    
    all_success = all(results.values())
    if all_success:
        print(colored("\n✓ All channels connected successfully!", "green"))
    else:
        print(colored("\n✗ Some channels failed to connect.", "red"))
    
    return all_success

def main():
    test_all_channels()

if __name__ == "__main__":
    main() 