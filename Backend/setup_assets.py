import os
from termcolor import colored
from TTS.api import TTS
import torch
import shutil

def setup_assets():
    """One-time setup of assets"""
    print(colored("\n=== Setting up MoneyPrinter assets ===\n", "blue"))
    
    # Create directory structure
    dirs = {
        'assets': {
            'videos': ['tech_humor', 'ai_money', 'baby_tips', 'quick_meals', 'fitness_motivation'],
            'tts': ['model'],
            'music': ['background']
        },
        'temp': {
            'videos': [],
            'tts': [],
            'subtitles': []
        },
        'output': {
            'videos': []
        }
    }
    
    for main_dir, sub_dirs in dirs.items():
        for sub_dir, channels in sub_dirs.items():
            base_path = f"{main_dir}/{sub_dir}"
            os.makedirs(base_path, exist_ok=True)
            for channel in channels:
                os.makedirs(f"{base_path}/{channel}", exist_ok=True)
    
    print(colored("\n✓ Assets setup complete!", "green"))

if __name__ == "__main__":
    setup_assets() 