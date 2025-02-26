import os
import subprocess
from moviepy.config import change_settings

def find_imagemagick():
    """Find ImageMagick binary path"""
    possible_paths = [
        r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe",
        r"C:\\Program Files\\ImageMagick-7.1.1-Q16\\magick.exe",
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
        r"C:\\ProgramData\\chocolatey\\lib\\imagemagick.app\\tools\\magick.exe",  # Chocolatey path
        "magick.exe"  # Will use system PATH
    ]
    
    # First try the which/where command
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'magick'], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip().split('\n')[0]
                print("Found ImageMagick via where command:", path)
                return path
    except Exception as e:
        print("Error finding ImageMagick in PATH:", str(e))

    # Then try the possible paths
    for path in possible_paths:
        if os.path.exists(path):
            print("Found ImageMagick at:", path)
            return path
            
    return None

# Find ImageMagick binary
IMAGEMAGICK_BINARY = find_imagemagick()

if not IMAGEMAGICK_BINARY:
    print("WARNING: ImageMagick not found. Please ensure it's installed correctly.")
    print("You can install it using: choco install imagemagick.app -y --params '/NoUpdate /InstallDevelopmentHeaders /InstallLegacyTools'")

# Configure MoviePy settings
MOVIEPY_SETTINGS = {
    "IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY,
    "FFMPEG_BINARY": "ffmpeg",
    "IMAGEMAGICK_PARAMS": ["-quiet"],
}

def configure_moviepy():
    try:
        if not IMAGEMAGICK_BINARY:
            raise FileNotFoundError("ImageMagick not found. Please install it and add to PATH")
            
        # Test ImageMagick
        try:
            result = subprocess.run([IMAGEMAGICK_BINARY, '-version'], 
                                  capture_output=True, 
                                  text=True)
            print("ImageMagick version:", result.stdout.split('\n')[0])
        except Exception as e:
            print("Warning: Could not verify ImageMagick version:", str(e))
            
        # Apply settings
        change_settings(MOVIEPY_SETTINGS)
        print("✓ MoviePy configured successfully")
        
        return True
    except Exception as e:
        print("Error configuring MoviePy:", str(e))
        print("\nTroubleshooting steps:")
        print("1. Install Chocolatey (https://chocolatey.org/install)")
        print("2. Run: choco install imagemagick.app -y --params '/NoUpdate /InstallDevelopmentHeaders /InstallLegacyTools'")
        print("3. Restart your terminal/IDE")
        print("4. Ensure the font 'Arial' is installed")
        return False 