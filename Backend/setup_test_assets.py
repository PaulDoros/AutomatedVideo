import os
import shutil
from termcolor import colored

def setup_test_assets():
    """Setup test assets for video generation"""
    try:
        # Create directories
        dirs = [
            'cache/scripts',
            'temp/videos',
            'temp/tts',
            'temp/subtitles',
            'output/videos',
            'assets/music/tech_humor',
            'assets/fonts'
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            
        # Copy test font if not exists
        if not os.path.exists('assets/fonts/bold_font.ttf'):
            # Using Arial as fallback
            system_font = 'C:/Windows/Fonts/Arial.ttf'
            if os.path.exists(system_font):
                shutil.copy2(system_font, 'assets/fonts/bold_font.ttf')
                print(colored("✓ Test font copied", "green"))
            else:
                print(colored("✗ Could not find system font", "yellow"))
        
        print(colored("✓ Test assets setup complete", "green"))
        
    except Exception as e:
        print(colored(f"✗ Error setting up test assets: {str(e)}", "red"))

if __name__ == "__main__":
    setup_test_assets() 