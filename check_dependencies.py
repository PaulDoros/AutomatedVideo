import pkg_resources
import subprocess
import sys

def check_and_install_dependencies():
    # Core dependencies with specific versions that we know work
    required_packages = {
        'TTS': '0.8.0',  # Using a stable version of TTS
        'torch': '2.0.1',  # TTS usually works well with this version
        'google-api-python-client': '2.100.0',
        'google-auth-httplib2': '0.1.0',
        'google-auth-oauthlib': '1.0.0',
        'openai': '1.0.0',
        'python-dotenv': '1.0.0',
        'termcolor': '2.3.0',
        'asyncio': '3.4.3',
        'pytz': '2023.3'
    }

    def install_package(package_name, version):
        print(f"Installing {package_name}=={version}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={version}"])

    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}=={version}")
            print(f"✓ {package} {version} is already installed")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"✗ {package} {version} needs to be installed")
            try:
                install_package(package, version)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")

if __name__ == "__main__":
    print("Checking and installing dependencies...")
    check_and_install_dependencies()
    print("\nDependency check complete. If TTS still doesn't work, try running:")
    print("pip uninstall TTS -y && pip install TTS==0.8.0") 