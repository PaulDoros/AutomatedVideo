#!/usr/bin/env python3
"""
Speaker Analysis Tool for Coqui TTS
Lists all available speakers in the XTTS v2 model and categorizes them by language
"""

import os
import sys
import torch
from termcolor import colored
import pandas as pd

def log_info(message):
    print(colored(f"ℹ️ {message}", "cyan"))

def log_success(message):
    print(colored(f"✅ {message}", "green"))

def log_error(message):
    print(colored(f"❌ {message}", "red"))

def analyze_speakers():
    try:
        log_info("Initializing Coqui TTS XTTS model...")
        
        # Import TTS library
        try:
            from TTS.api import TTS
        except ImportError:
            log_error("Failed to import TTS. Make sure Coqui TTS is installed.")
            log_info("Try: pip install TTS")
            return
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_info(f"Using device: {device}")
        
        # Initialize the model
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        # Get all speakers
        if hasattr(model, "speakers") and model.speakers:
            speakers = model.speakers
            log_success(f"Found {len(speakers)} speakers in model")
        else:
            log_error("No speakers found in model")
            return
        
        # Get all languages
        if hasattr(model, "languages") and model.languages:
            languages = model.languages
            log_success(f"Found {len(languages)} languages in model: {', '.join(languages)}")
        else:
            log_error("No languages found in model")
            return
        
        # Analyze speakers and their possible languages
        log_info("Analyzing speakers...")
        
        # Create a dictionary to hold speakers by language patterns
        speakers_by_pattern = {
            "en_": [],  # Starts with 'en_'
            "en-": [],  # Contains 'en-'
            "english": [],  # Contains 'english'
            "p3": [],  # VCTK dataset (p326, etc.)
            "other": []  # Other speakers
        }
        
        # Categorize speakers by naming pattern
        for speaker in speakers:
            speaker_lower = speaker.lower()
            if speaker_lower.startswith("en_"):
                speakers_by_pattern["en_"].append(speaker)
            elif "en-" in speaker_lower:
                speakers_by_pattern["en-"].append(speaker)
            elif "english" in speaker_lower:
                speakers_by_pattern["english"].append(speaker)
            elif speaker_lower.startswith("p") and any(c.isdigit() for c in speaker_lower):
                speakers_by_pattern["p3"].append(speaker)
            else:
                speakers_by_pattern["other"].append(speaker)
        
        # Print summary of speaker patterns
        for pattern, speaker_list in speakers_by_pattern.items():
            if speaker_list:
                log_info(f"Pattern '{pattern}': {len(speaker_list)} speakers")
                # Print first 5 examples
                for i, speaker in enumerate(speaker_list[:5]):
                    print(f"  - {speaker}")
                if len(speaker_list) > 5:
                    print(f"  - ... and {len(speaker_list) - 5} more")
        
        # Try to get language information for each speaker if possible
        log_info("Attempting to determine language for each speaker...")
        
        speaker_info = []
        for speaker in speakers:
            # Extract language from speaker name if possible
            language = "unknown"
            gender = "unknown"
            
            # Check for language code in speaker name
            for lang in languages:
                if lang in speaker.lower():
                    language = lang
                    break
            
            # Check for gender in speaker name
            if "female" in speaker.lower() or "woman" in speaker.lower():
                gender = "female"
            elif "male" in speaker.lower() or "man" in speaker.lower():
                gender = "male"
            
            speaker_info.append({
                "speaker": speaker,
                "language": language,
                "gender": gender,
                "is_english": language == "en" or speaker.lower().startswith("en_") or "english" in speaker.lower()
            })
        
        # Create a DataFrame for better viewing
        df = pd.DataFrame(speaker_info)
        
        # Save to CSV for reference
        csv_path = "temp/speaker_analysis.csv"
        os.makedirs("temp", exist_ok=True)
        df.to_csv(csv_path, index=False)
        log_success(f"Saved speaker analysis to {csv_path}")
        
        # Print summary statistics
        english_speakers = df[df["is_english"] == True]
        log_info(f"Found {len(english_speakers)} potential English speakers")
        
        # Print recommended English speakers
        if not english_speakers.empty:
            log_success("Recommended English speakers:")
            for i, (_, row) in enumerate(english_speakers.head(10).iterrows()):
                print(f"  {i+1}. {row['speaker']} (Gender: {row['gender']})")
        
        # Suggest improved speaker selection code
        log_info("\nRecommended English speaker selection code:")
        print(colored("""
# Best way to select English speakers from XTTS v2:
english_speakers = []
for speaker in available_speakers:
    # Check for English language indicators
    if (speaker.lower().startswith("en_") or 
        "en-" in speaker.lower() or 
        "english" in speaker.lower() or
        (speaker.lower().startswith("p") and any(c.isdigit() for c in speaker))):
        english_speakers.append(speaker)

# If no English speakers found, try to use any speaker with the English language
if not english_speakers and hasattr(model, "languages") and "en" in model.languages:
    log_info("No specific English speakers found, using any speaker with English language")
    english_speakers = available_speakers  # All speakers with English language
        """, "yellow"))
        
        return english_speakers
        
    except Exception as e:
        log_error(f"Error analyzing speakers: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    print(colored("\n=== Coqui TTS Speaker Analysis ===\n", "blue"))
    analyze_speakers()
    print(colored("\n=== Analysis Complete ===\n", "blue")) 