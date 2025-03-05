import os
import requests
from termcolor import colored
import zipfile
import shutil
from io import BytesIO

def download_fonts():
    """Download required fonts for thumbnail generation"""
    try:
        fonts_dir = "assets/fonts"
        os.makedirs(fonts_dir, exist_ok=True)
        
        # Check if fonts already exist
        montserrat_bold = f"{fonts_dir}/Montserrat-Bold.ttf"
        roboto_regular = f"{fonts_dir}/Roboto-Regular.ttf"
        
        if os.path.exists(montserrat_bold) and os.path.exists(roboto_regular):
            print(colored("✓ Required fonts already exist", "green"))
            return True
            
        print(colored("Downloading required fonts...", "blue"))
        
        # Download Montserrat Bold
        montserrat_url = "https://fonts.google.com/download?family=Montserrat"
        
        try:
            response = requests.get(montserrat_url)
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                # Extract only the Bold font
                for file in zip_ref.namelist():
                    if "Montserrat-Bold.ttf" in file:
                        with zip_ref.open(file) as source, open(montserrat_bold, "wb") as target:
                            shutil.copyfileobj(source, target)
                        print(colored(f"✓ Downloaded Montserrat Bold font", "green"))
                        break
        except Exception as e:
            print(colored(f"Error downloading Montserrat font: {str(e)}", "red"))
            # Use fallback URL if Google Fonts fails
            fallback_url = "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf"
            try:
                response = requests.get(fallback_url)
                with open(montserrat_bold, "wb") as f:
                    f.write(response.content)
                print(colored(f"✓ Downloaded Montserrat Bold font from fallback", "green"))
            except Exception as e:
                print(colored(f"Error downloading from fallback: {str(e)}", "red"))
                return False
        
        # Download Roboto Regular
        roboto_url = "https://fonts.google.com/download?family=Roboto"
        
        try:
            response = requests.get(roboto_url)
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                # Extract only the Regular font
                for file in zip_ref.namelist():
                    if "Roboto-Regular.ttf" in file:
                        with zip_ref.open(file) as source, open(roboto_regular, "wb") as target:
                            shutil.copyfileobj(source, target)
                        print(colored(f"✓ Downloaded Roboto Regular font", "green"))
                        break
        except Exception as e:
            print(colored(f"Error downloading Roboto font: {str(e)}", "red"))
            # Use fallback URL if Google Fonts fails
            fallback_url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
            try:
                response = requests.get(fallback_url)
                with open(roboto_regular, "wb") as f:
                    f.write(response.content)
                print(colored(f"✓ Downloaded Roboto Regular font from fallback", "green"))
            except Exception as e:
                print(colored(f"Error downloading from fallback: {str(e)}", "red"))
                return False
        
        return True
        
    except Exception as e:
        print(colored(f"Error setting up fonts: {str(e)}", "red"))
        return False

if __name__ == "__main__":
    download_fonts() 