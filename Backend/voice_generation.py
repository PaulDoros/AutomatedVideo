import os
import asyncio
import logging
from typing import Optional, Dict, List, Tuple
from coqui_integration import CoquiTTSAPI
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VoiceGeneration")

class VoiceGenerator:
    """Voice generation system that uses Coqui TTS as primary with GPT as fallback"""
    
    def __init__(self):
        """Initialize the voice generation system"""
        self.coqui_api = CoquiTTSAPI()
        self.available_voices = {}
        self.preferred_voices = {
            "male": {
                "default": "Viktor Menelaos",
                "professional": "Damien Black",
                "accent": "Luis Moray"
            },
            "female": {
                "default": "Daisy Studious",
                "warm": "Sofia Hellen",
                "accent": "Alma María"
            }
        }
        
        # Initialize voice cache
        self._init_voice_cache()
        
    def _init_voice_cache(self):
        """Initialize the voice cache with available voices"""
        try:
            self.available_voices = self.coqui_api.get_available_voices()
            logger.info(f"Loaded {sum(len(voices) for voices in self.available_voices.values())} voices from Coqui TTS")
        except Exception as e:
            logger.error(f"Failed to initialize voice cache: {str(e)}")
            self.available_voices = {}
    
    def get_voice_options(self) -> Dict[str, List[str]]:
        """Get available voice options grouped by gender"""
        male_voices = []
        female_voices = []
        
        # Add preferred voices first
        for voice_type, voices in self.preferred_voices.items():
            for _, voice in voices.items():
                if voice_type == "male":
                    male_voices.append(voice)
                else:
                    female_voices.append(voice)
        
        # Add remaining voices from XTTS model
        if "xtts_v2" in self.available_voices:
            for voice in self.available_voices["xtts_v2"]:
                if voice not in male_voices and voice not in female_voices:
                    # Simple heuristic - this could be improved with a proper gender classification
                    if voice in self.coqui_api.speakers:
                        if voice not in male_voices and voice not in female_voices:
                            # Add to appropriate list based on our best guess
                            if any(name in voice.lower() for name in ["viktor", "damien", "luis", "aaron", "craig", "badr"]):
                                male_voices.append(voice)
                            elif any(name in voice.lower() for name in ["daisy", "sofia", "alma", "gracie", "alison"]):
                                female_voices.append(voice)
        
        return {
            "male": male_voices,
            "female": female_voices
        }
    
    def get_recommended_voice(self, gender: str = "female", style: str = "default") -> str:
        """Get a recommended voice based on gender and style preferences"""
        if gender.lower() in self.preferred_voices:
            voices = self.preferred_voices[gender.lower()]
            if style.lower() in voices:
                return voices[style.lower()]
            return voices["default"]
        return self.preferred_voices["female"]["default"]  # Default fallback
    
    async def generate_voice(self, 
                           text: str, 
                           voice: str = None,
                           gender: str = "female", 
                           style: str = "default",
                           emotion: str = "neutral", 
                           language: str = "en",
                           speed: float = 1.0) -> Optional[str]:
        """Generate voice audio for the given text
        
        Args:
            text: The text to convert to speech
            voice: Specific voice to use (if None, will use recommended voice)
            gender: Gender preference if no specific voice provided
            style: Voice style preference if no specific voice provided
            emotion: Emotion to apply to the voice
            language: Language code
            speed: Speech speed multiplier
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        # Select voice if not specified
        if not voice:
            voice = self.get_recommended_voice(gender, style)
            
        logger.info(f"Generating voice with Coqui TTS: {voice}, {emotion}")
        
        try:
            # Create unique output path
            os.makedirs("temp/tts/generated", exist_ok=True)
            safe_voice = voice.replace(" ", "_").replace(".", "").lower()
            output_path = f"temp/tts/generated/{safe_voice}_{emotion}_{int(asyncio.get_event_loop().time() * 1000)}.wav"
            
            # Try with Coqui TTS
            result = await self.coqui_api.generate_voice(
                text=text,
                speaker=voice,
                language=language,
                emotion=emotion,
                speed=speed,
                output_path=output_path
            )
            
            if result:
                logger.info(f"Successfully generated voice with Coqui TTS: {result}")
                return result
                
            # If Coqui TTS failed, log the error and try fallback
            logger.warning("Coqui TTS generation failed, falling back to GPT TTS")
            
            # TODO: Implement GPT TTS fallback
            # This would call the OpenAI TTS API or another fallback system
            logger.error("GPT TTS fallback not implemented yet")
            return None
            
        except Exception as e:
            logger.error(f"Voice generation failed: {str(e)}")
            return None
    
    async def list_all_voices(self) -> Dict[str, List[Tuple[str, str]]]:
        """List all available voices with gender classification
        
        Returns:
            Dictionary with gender as key and list of (voice_name, description) tuples as value
        """
        result = {"male": [], "female": []}
        
        # Add preferred voices with descriptions
        for gender, voices in self.preferred_voices.items():
            for style, voice in voices.items():
                description = f"{style.capitalize()} voice"
                result[gender].append((voice, description))
        
        # Add remaining voices from XTTS model
        if "xtts_v2" in self.available_voices:
            for voice in self.available_voices["xtts_v2"]:
                # Skip voices already in preferred list
                if any(voice == v[0] for v in result["male"] + result["female"]):
                    continue
                    
                # Simple gender classification heuristic
                if any(name in voice.lower() for name in ["viktor", "damien", "luis", "aaron", "craig", "badr"]):
                    result["male"].append((voice, "Additional voice"))
                elif any(name in voice.lower() for name in ["daisy", "sofia", "alma", "gracie", "alison"]):
                    result["female"].append((voice, "Additional voice"))
                else:
                    # If we can't determine gender, add to both lists
                    result["male"].append((voice, "Unclassified voice"))
                    result["female"].append((voice, "Unclassified voice"))
        
        return result

# Example usage
async def test_voice_generator():
    generator = VoiceGenerator()
    
    print("Available voice options:")
    options = generator.get_voice_options()
    print(f"Male voices: {', '.join(options['male'][:5])}...")
    print(f"Female voices: {', '.join(options['female'][:5])}...")
    
    print("\nRecommended voices:")
    print(f"Default female: {generator.get_recommended_voice('female', 'default')}")
    print(f"Professional male: {generator.get_recommended_voice('male', 'professional')}")
    
    print("\nGenerating sample voices:")
    text = "This is a test of the integrated voice generation system with Coqui TTS as primary and GPT as fallback."
    
    # Test with default female voice
    result = await generator.generate_voice(
        text=text,
        gender="female",
        emotion="professional"
    )
    
    if result:
        print(f"✓ Generated female voice: {result}")
    
    # Test with specific male voice
    result = await generator.generate_voice(
        text=text,
        voice="Damien Black",
        emotion="serious"
    )
    
    if result:
        print(f"✓ Generated male voice: {result}")
    
    print("\nVoice generation test complete!")

if __name__ == "__main__":
    asyncio.run(test_voice_generator()) 