import asyncio
import os
import random
import time
from typing import List, Dict, Optional
from coqui_integration import CoquiTTSAPI
from termcolor import colored

# Voice classification
MALE_VOICE_KEYWORDS = ['andrew', 'damien', 'luis', 'viktor', 'badr', 'craig', 'aaron', 'marcos', 
                       'royston', 'baldur', 'gilberto', 'ilkin', 'kazuhiko', 'ludvig', 'suad', 
                       'torcull', 'zacharie', 'filip', 'damjan', 'wulf', 'kumar', 'eugenio', 'ferran', 
                       'xavier']

FEMALE_VOICE_KEYWORDS = ['daisy', 'sofia', 'alma', 'gracie', 'alison', 'tammy', 'brenda', 'claribel',
                         'tammie', 'ana', 'annmarie', 'asya', 'gitta', 'henriette', 'tanja', 'vjollca',
                         'nova', 'maja', 'uta', 'lidiya', 'chandra', 'szofi', 'camilla', 'lilya', 
                         'zofija', 'narelle', 'barbora', 'alexandra', 'rosemary']

# Sample texts for testing
SAMPLE_TEXTS = {
    "intro": "Welcome to our channel! Today we're going to explore an exciting topic that will change the way you think about technology.",
    "tutorial": "In this tutorial, I'll show you step by step how to implement this solution. It's easier than you might think.",
    "explanation": "Let me explain how this works. The concept is based on fundamental principles that have been proven effective.",
    "conclusion": "Thanks for watching! If you found this video helpful, don't forget to like and subscribe for more content like this.",
    "joke": "Why don't scientists trust atoms? Because they make up everything! And speaking of making things up, my code works on the first try.",
    "fact": "Did you know that artificial intelligence is transforming industries across the globe? The potential applications are virtually limitless."
}

# Emotions for testing
EMOTIONS = ["neutral", "cheerful", "professional", "friendly", "serious"]

class VoiceDiversification:
    """Voice diversification system for video voiceovers"""
    
    def __init__(self):
        """Initialize the voice diversification system"""
        self.api = CoquiTTSAPI()
        self.male_voices = []
        self.female_voices = []
        self.all_voices = []
        self.has_speaker_support = hasattr(self.api, 'speakers') and self.api.speakers
        self._classify_voices()
        
    def _classify_voices(self):
        """Classify available voices into male and female categories"""
        if not self.has_speaker_support:
            print(colored("Current TTS model doesn't support multiple speakers", "yellow"))
            print(colored("Using default voice with emotion variations", "yellow"))
            return
            
        # Classify voices
        self.male_voices = [s for s in self.api.speakers if any(name in s.lower() for name in MALE_VOICE_KEYWORDS)]
        self.female_voices = [s for s in self.api.speakers if any(name in s.lower() for name in FEMALE_VOICE_KEYWORDS)]
        
        # Add remaining voices to all_voices
        self.all_voices = self.api.speakers.copy()
        
        print(colored(f"Classified {len(self.male_voices)} male voices and {len(self.female_voices)} female voices", "green"))
        print(colored(f"Total available voices: {len(self.all_voices)}", "green"))
        
    def get_random_voice(self, gender: Optional[str] = None) -> str:
        """Get a random voice based on gender preference"""
        if not self.has_speaker_support:
            return None
            
        if gender == "male" and self.male_voices:
            return random.choice(self.male_voices)
        elif gender == "female" and self.female_voices:
            return random.choice(self.female_voices)
        else:
            return random.choice(self.all_voices) if self.all_voices else None
            
    def get_random_emotion(self) -> str:
        """Get a random emotion"""
        return random.choice(EMOTIONS)
        
    def get_random_text(self) -> str:
        """Get a random sample text"""
        return random.choice(list(SAMPLE_TEXTS.values()))
        
    async def generate_diverse_samples(self, output_dir: str = "temp/tts/diverse_samples"):
        """Generate diverse voice samples for all voices"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate samples for all voices
        results = []
        
        print(colored("\n=== Generating samples for all voices ===", "cyan"))
        
        # Use a consistent text for comparison
        text = SAMPLE_TEXTS["intro"]
        emotion = "professional"
        
        if not self.has_speaker_support:
            print(colored("Current model doesn't support multiple speakers. Generating single sample.", "yellow"))
            output_path = f"{output_dir}/default_voice_{int(time.time())}.wav"
            result = await self.api.generate_voice(
                text=text,
                language="en",
                emotion=emotion,
                speed=1.0,
                output_path=output_path
            )
            if result:
                results.append({
                    "voice": "default",
                    "gender": "unknown",
                    "file": result,
                    "abs_path": os.path.abspath(result)
                })
                print(colored(f"✓ Generated: {result}", "green"))
            else:
                print(colored(f"✗ Failed to generate sample", "red"))
            return results
        
        for i, voice in enumerate(self.all_voices):
            print(colored(f"\nGenerating sample {i+1}/{len(self.all_voices)}: {voice}", "yellow"))
            
            # Create unique filename
            safe_voice = voice.replace(" ", "_").replace(".", "").lower()
            filename = f"{safe_voice}_{int(time.time())}.wav"
            output_path = f"{output_dir}/{filename}"
            
            # Generate voice
            result = await self.api.generate_voice(
                text=text,
                speaker=voice,
                language="en",
                emotion=emotion,
                speed=1.0,
                output_path=output_path
            )
            
            if result:
                gender = "male" if voice in self.male_voices else "female" if voice in self.female_voices else "unknown"
                results.append({
                    "voice": voice,
                    "gender": gender,
                    "file": result,
                    "abs_path": os.path.abspath(result)
                })
                print(colored(f"✓ Generated: {result}", "green"))
            else:
                print(colored(f"✗ Failed to generate sample for {voice}", "red"))
                
        return results
        
    async def generate_random_voiceover(self, text: str = None, gender: str = None, emotion: str = None):
        """Generate a random voiceover for a video"""
        # Select random parameters if not provided
        if text is None:
            text = self.get_random_text()
        if emotion is None:
            emotion = self.get_random_emotion()
        
        # Create output directory
        output_dir = "temp/tts/random_voiceovers"
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle models without speaker support
        if not self.has_speaker_support:
            print(colored(f"\n=== Generating voiceover with default voice ===", "cyan"))
            print(colored(f"Emotion: {emotion}", "yellow"))
            print(colored(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}", "yellow"))
            
            # Create unique filename
            filename = f"default_voice_{emotion}_{int(time.time())}.wav"
            output_path = f"{output_dir}/{filename}"
            
            # Generate voice
            result = await self.api.generate_voice(
                text=text,
                language="en",
                emotion=emotion,
                speed=1.0,
                output_path=output_path
            )
            
            if result:
                print(colored(f"✓ Generated voiceover: {result}", "green"))
                print(colored(f"Absolute path: {os.path.abspath(result)}", "green"))
                return result
            else:
                print(colored("✗ Failed to generate voiceover", "red"))
                return None
        
        # Get random voice based on gender preference
        voice = self.get_random_voice(gender)
        
        if not voice:
            print(colored("No voices available", "red"))
            return None
            
        print(colored(f"\n=== Generating random voiceover ===", "cyan"))
        print(colored(f"Voice: {voice}", "yellow"))
        print(colored(f"Emotion: {emotion}", "yellow"))
        print(colored(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}", "yellow"))
        
        # Create unique filename
        safe_voice = voice.replace(" ", "_").replace(".", "").lower()
        filename = f"random_{safe_voice}_{emotion}_{int(time.time())}.wav"
        output_path = f"{output_dir}/{filename}"
        
        # Generate voice
        result = await self.api.generate_voice(
            text=text,
            speaker=voice,
            language="en",
            emotion=emotion,
            speed=1.0,
            output_path=output_path
        )
        
        if result:
            print(colored(f"✓ Generated random voiceover: {result}", "green"))
            print(colored(f"Absolute path: {os.path.abspath(result)}", "green"))
            return result
        else:
            print(colored("✗ Failed to generate random voiceover", "red"))
            return None

    def select_voice(self, channel_type, gender=None):
        """Select a voice based on channel type and gender preference with consistent results"""
        try:
            # Determine gender preference based on content type if not specified
            if gender is None:
                if channel_type == 'tech_humor':
                    gender = 'male'  # Use male voices for tech humor
                elif channel_type == 'baby_tips':
                    gender = 'female'  # Prefer female voices for baby tips
                elif channel_type == 'fitness_motivation':
                    gender = 'male'    # Prefer male voices for fitness motivation
                elif channel_type == 'quick_meals':
                    gender = 'female'  # Prefer female voices for cooking content
                elif channel_type == 'ai_money':
                    gender = 'male'    # Prefer male voices for finance content
            
            # Use a deterministic approach to select a voice based on channel type
            # This ensures we get the same voice each time for a given channel type
            if gender == 'male':
                voices = self.male_voices
            elif gender == 'female':
                voices = self.female_voices
            else:
                voices = self.all_voices
            
            if not voices:
                return "default"
            
            # Use a hash of the channel type to select a consistent voice
            # This is deterministic - same channel type will always get the same voice
            index = hash(channel_type) % len(voices)
            return voices[index]
            
        except Exception as e:
            print(f"Error selecting voice: {str(e)}")
            return "default"

async def test_voice_diversification():
    """Test the voice diversification system"""
    diversifier = VoiceDiversification()
    
    # Print available voices
    print("\n=== Available Male Voices ===")
    for voice in diversifier.male_voices:
        print(f"- {voice}")
        
    print("\n=== Available Female Voices ===")
    for voice in diversifier.female_voices:
        print(f"- {voice}")
    
    # Generate random voiceovers
    print("\n=== Testing Random Voiceovers ===")
    
    # Random male voice
    await diversifier.generate_random_voiceover(
        text=SAMPLE_TEXTS["tutorial"],
        gender="male",
        emotion="professional"
    )
    
    # Random female voice
    await diversifier.generate_random_voiceover(
        text=SAMPLE_TEXTS["explanation"],
        gender="female",
        emotion="friendly"
    )
    
    # Completely random
    await diversifier.generate_random_voiceover()
    
    # Generate samples for all voices (commented out to avoid generating too many files)
    # Uncomment to generate samples for all voices
    print("\nDo you want to generate samples for all voices? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        results = await diversifier.generate_diverse_samples()
        print(f"\nGenerated {len(results)} voice samples")
    else:
        print("\nSkipping generation of all voice samples")
    
    print("\n✓ Voice diversification test complete!")

if __name__ == "__main__":
    asyncio.run(test_voice_diversification()) 