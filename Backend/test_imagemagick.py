import os

# Set ImageMagick path before importing moviepy
os.environ['IMAGEMAGICK_BINARY'] = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"

# Now import moviepy
from moviepy.editor import *

def test_imagemagick():
    # Print current working directory
    print("Current working directory:", os.getcwd())
    
    # Print ImageMagick path from environment
    print("ImageMagick path from env:", os.environ['IMAGEMAGICK_BINARY'])
    
    # Print actual file existence
    path = os.environ['IMAGEMAGICK_BINARY']
    print("Does ImageMagick exist at path?:", os.path.exists(path))
    
    # Try to create a simple text clip
    try:
        text_clip = TextClip("Test", fontsize=70, color='white', size=(640, 480))
        print("Successfully created text clip!")
        text_clip.close()
    except Exception as e:
        print("Error creating text clip:", str(e))

if __name__ == "__main__":
    test_imagemagick() 