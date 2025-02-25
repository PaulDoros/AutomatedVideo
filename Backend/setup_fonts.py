import os
import requests
from termcolor import colored
from PIL import ImageFont

def download_fonts():
    """Download required fonts"""
    fonts = {
        'Montserrat-Bold': 'https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf',
        'Roboto-Regular': 'https://github.com/google/fonts/raw/main/ofl/roboto/static/Roboto-Regular.ttf'
    }

    fonts_dir = 'assets/fonts'
    os.makedirs(fonts_dir, exist_ok=True)

    for font_name, url in fonts.items():
        file_path = f'{fonts_dir}/{font_name}.ttf'
        if not os.path.exists(file_path):
            try:
                print(colored(f"Downloading {font_name}...", "blue"))
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad status codes
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(colored(f"✓ Successfully downloaded {font_name}", "green"))
            except Exception as e:
                print(colored(f"✗ Error downloading {font_name}: {str(e)}", "red"))
                print(colored("Attempting alternate download URL...", "yellow"))
                # Try alternate URLs
                alt_urls = {
                    'Montserrat-Bold': 'https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf',
                    'Roboto-Regular': 'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf'
                }
                try:
                    response = requests.get(alt_urls[font_name])
                    response.raise_for_status()
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    print(colored(f"✓ Successfully downloaded {font_name} from alternate source", "green"))
                except Exception as e2:
                    print(colored(f"✗ Error downloading from alternate source: {str(e2)}", "red"))
        else:
            print(colored(f"✓ {font_name} already exists", "green"))

    # Verify downloaded fonts
    for font_name in fonts.keys():
        file_path = f'{fonts_dir}/{font_name}.ttf'
        if os.path.exists(file_path):
            try:
                # Try to load the font to verify it's valid
                test_font = ImageFont.truetype(file_path, 12)
                print(colored(f"✓ Verified {font_name} is valid", "green"))
            except Exception as e:
                print(colored(f"✗ Font file exists but is invalid: {font_name}", "red"))
                os.remove(file_path)  # Remove invalid font file
                print(colored(f"Removed invalid font file: {font_name}", "yellow"))

if __name__ == "__main__":
    download_fonts() 