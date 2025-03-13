import subprocess
import sys
import pkg_resources

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("🔍 Checking and fixing TTS installation...")
    
    # First uninstall existing TTS if present
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "TTS", "-y"])
        print("✓ Removed existing TTS installation")
    except:
        print("No existing TTS installation found")
    
    # Install TTS with all required dependencies
    print("\n📦 Installing TTS with required dependencies...")
    install_package("TTS")
    
    # Verify installation
    try:
        from TTS.api import TTS
        print("\n✅ TTS installation successful!")
        print("\nAvailable models:")
        tts = TTS()
        print(tts.list_models())
    except Exception as e:
        print("\n❌ Error during verification:", str(e))
        print("Please try running: pip install TTS --upgrade")

if __name__ == "__main__":
    main() 