import os
import torch
import asyncio
import re
import multiprocessing
from typing import Optional, Dict
from TTS.api import TTS
from termcolor import colored
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from contextlib import contextmanager

class CoquiTTSAPI:
    def __init__(self):
        """Initialize Coqui TTS with XTTS-v2 model"""
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Set CUDA device to 0 (first GPU)
            torch.cuda.set_device(0)
            print(colored(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}", "green"))
            # Use XTTS-v2 model with GPU
            self.default_model = "xtts_v2"
            # Set GPU optimization level - higher values prioritize quality over speed
            self.gpu_optimization_level = "high_quality"
        else:
            # Optimize for CPU performance
            print(colored("GPU not available, optimizing for CPU performance", "yellow"))
            # Set number of threads for CPU optimization
            num_threads = min(multiprocessing.cpu_count(), 16)  # Use up to 16 CPU threads
            torch.set_num_threads(num_threads)
            print(colored(f"Using {num_threads} CPU threads for processing", "yellow"))
            # Use faster model for CPU
            self.default_model = "fast"
            print(colored("Using faster model for CPU processing", "yellow"))
            
        self.models = {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/vctk/vits",
            "fast": "tts_models/en/ljspeech/fast_pitch"
        }
        
        # Create cache directory for models
        os.makedirs("assets/tts/model_cache", exist_ok=True)
        
        # Initialize with default model
        self._init_tts(self.default_model)
        
    @contextmanager
    def _torch_load_context(self):
        """Context manager for PyTorch loading compatibility"""
        try:
            # Add XttsConfig to safe globals for PyTorch 2.6+ compatibility
            torch.serialization.add_safe_globals([XttsConfig])
            yield
        finally:
            pass

    @contextmanager
    def _gpu_context(self):
        """Context manager for GPU operations"""
        try:
            # If using CUDA, ensure we're using the right device
            if self.device == "cuda" and torch.cuda.is_available():
                # Set CUDA device to 0 (first GPU)
                torch.cuda.set_device(0)
                # Clear CUDA cache to free up memory
                torch.cuda.empty_cache()
            yield
        finally:
            # Clean up after TTS generation
            if self.device == "cuda" and torch.cuda.is_available():
                # Clear CUDA cache again
                torch.cuda.empty_cache()

    def _init_tts(self, model_key: str = None):
        """Initialize TTS with specified model"""
        try:
            if model_key is None:
                model_key = self.default_model
                
            model_name = self.models.get(model_key)
            if not model_name:
                print(colored(f"Unknown model key: {model_key}, falling back to fast model", "yellow"))
                model_key = "fast"
                model_name = self.models.get("fast")
                
            print(colored(f"Initializing Coqui TTS with model: {model_name}", "cyan"))
            
            # Use context manager for PyTorch loading
            with self._torch_load_context():
                try:
                    # Initialize TTS without gpu parameter (which is deprecated)
                    self.tts = TTS(model_name=model_name, progress_bar=True)
                    
                    # Move model to appropriate device after initialization
                    if self.device == "cuda" and hasattr(self.tts, "to"):
                        self.tts.to(self.device)
                except Exception as model_error:
                    print(colored(f"Error loading model {model_name}: {str(model_error)}", "red"))
                    
                    # If this wasn't already the fast model, try that as fallback
                    if model_key != "fast":
                        print(colored("Trying fast model as fallback...", "yellow"))
                        model_name = self.models.get("fast")
                        try:
                            self.tts = TTS(model_name=model_name, progress_bar=True)
                        except Exception as fallback_error:
                            print(colored(f"Fallback model also failed: {str(fallback_error)}", "red"))
                            return False
                    else:
                        # If we're already trying the fast model and it failed, return False
                        return False
            
            # Store model capabilities with safe checks
            self.is_multi_speaker = hasattr(self.tts, "speakers") and self.tts.speakers is not None
            self.is_multilingual = hasattr(self.tts, "languages") and self.tts.languages is not None
            
            # Safely get speakers and languages
            try:
                self.speakers = self.tts.speakers if self.is_multi_speaker else []
            except Exception:
                self.speakers = []
                self.is_multi_speaker = False
                
            try:
                self.languages = self.tts.languages if self.is_multilingual else ["en"]
            except Exception:
                self.languages = ["en"]
                self.is_multilingual = False
            
            # Ensure speakers and languages are not None
            if self.speakers is None:
                self.speakers = []
            if self.languages is None:
                self.languages = ["en"]
            
            if self.is_multi_speaker:
                print(colored(f"Available speakers: {len(self.speakers)}", "green"))
            if self.is_multilingual:
                print(colored(f"Available languages: {self.languages}", "green"))
            
            return True
            
        except Exception as e:
            print(colored(f"Error initializing TTS model: {str(e)}", "red"))
            # Reset TTS to avoid using a partially initialized model
            self.tts = None
            self.speakers = []
            self.languages = ["en"]
            self.is_multi_speaker = False
            self.is_multilingual = False
            return False
    
    def _clean_text(self, text: str, emotion: str = None) -> str:
        """Clean text by removing emojis and emotion tags, and improving pronunciation of technical terms"""
        # Remove emoji characters
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
        
        text = emoji_pattern.sub(r'', text)
        
        # Remove emotion tags like [happy], [sad], etc.
        text = re.sub(r'\[(.*?)\]', '', text)
        
        # Improve pronunciation of technical terms
        # Replace acronyms and technical terms with spelled-out versions for better pronunciation
        tech_terms = {
            r'\bGPU\b': 'G P U',
            r'\bCPU\b': 'C P U',
            r'\bRAM\b': 'RAM',
            r'\bNVIDIA\b': 'N-VIDIA',
            r'\bRTX\b': 'R T X',
            r'\bSUPER\b': 'SUPER',
            r'\bi9\b': 'i 9',
            r'\bKF\b': 'K F',
            r'\bGHz\b': 'gigahertz',
        }
        
        for pattern, replacement in tech_terms.items():
            text = re.sub(pattern, replacement, text)
        
        # Add slight pauses for better phrasing
        text = text.replace('. ', '. <break time="0.3s"/> ')
        text = text.replace('! ', '! <break time="0.3s"/> ')
        text = text.replace('? ', '? <break time="0.3s"/> ')
        text = text.replace(', ', ', <break time="0.1s"/> ')
        
        return text
            
    async def generate_voice(self, text: str, speaker: str = None, language: str = "en", 
                           emotion: str = "neutral", speed: float = 1.0, output_path: str = None) -> Optional[str]:
        """Generate voice using Coqui TTS"""
        try:
            # Create temp directory if it doesn't exist
            os.makedirs("temp/tts", exist_ok=True)
            
            # Use custom output path if provided, otherwise generate one
            if not output_path:
                output_path = f"temp/tts/coqui_{hash(text)}.wav"
            else:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Force English language to avoid non-English TTS
            language = "en"
            
            # Clean the text (remove emojis and emotion tags)
            clean_text = self._clean_text(text, emotion)
            
            # Generate speech
            print(colored(f"Generating speech with Coqui TTS ({self.device})", "cyan"))
            print(colored(f"Text: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}", "cyan"))
            
            # Ensure TTS is initialized
            if not hasattr(self, 'tts') or self.tts is None:
                if not self._init_tts():
                    print(colored("Failed to initialize TTS model, trying OpenAI fallback", "yellow"))
                    return await self._try_openai_tts(clean_text, output_path)
            
            # Prepare TTS parameters
            tts_kwargs = {"text": clean_text, "file_path": output_path}
            
            # Apply speed adjustment based on emotion and device
            if self.device == "cuda":
                # On GPU, we can use higher quality settings
                # Adjust speed based on content type in the output path
                if "tech_humor" in output_path:
                    # Faster for tech humor content - increase speed for jokes
                    tts_kwargs["speed"] = speed * 1.3
                    print(colored(f"Using speed multiplier of 1.3 for tech_humor content", "blue"))
                elif emotion in ["energetic", "enthusiastic", "playful"]:
                    # Faster for energetic emotions
                    tts_kwargs["speed"] = speed * 1.25
                    print(colored(f"Using speed multiplier of 1.25 for {emotion} emotion", "blue"))
                elif emotion == "cheerful":
                    # Slightly faster for cheerful emotion
                    tts_kwargs["speed"] = speed * 1.15
                    print(colored(f"Using speed multiplier of 1.15 for cheerful emotion", "blue"))
                else:
                    # Normal speed for other emotions
                    tts_kwargs["speed"] = speed
                    print(colored(f"Using normal speed for {emotion} emotion", "blue"))
            else:
                # On CPU, prioritize speed
                tts_kwargs["speed"] = speed * 1.05  # Slightly faster by default on CPU
            
            # Verify speaker is valid and in English
            if speaker:
                # Check if this is a non-English speaker name
                non_english_indicators = ['zh', 'cn', 'jp', 'ru', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ko']
                if any(indicator in speaker.lower() for indicator in non_english_indicators):
                    print(colored(f"Warning: Speaker '{speaker}' appears to be non-English. Switching to default English voice.", "yellow"))
                    # Try to find an English speaker
                    if hasattr(self.tts, 'speakers') and self.tts.speakers is not None:
                        english_speakers = [s for s in self.tts.speakers if 'en' in s.lower() or not any(indicator in s.lower() for indicator in non_english_indicators)]
                        if english_speakers:
                            speaker = english_speakers[0]
                            print(colored(f"Switched to English speaker: {speaker}", "green"))
                        else:
                            speaker = None
                            print(colored("No English speakers found, using default voice", "yellow"))
                    else:
                        speaker = None
                
                # Add speaker if provided and supported
                if speaker and hasattr(self.tts, 'speakers') and self.tts.speakers is not None and speaker in self.tts.speakers:
                    tts_kwargs["speaker"] = speaker
                    print(colored(f"Using speaker: {speaker}", "blue"))
            
            # Force English language
            if hasattr(self.tts, 'languages') and self.tts.languages is not None:
                if "en" in self.tts.languages:
                    tts_kwargs["language"] = "en"
                    print(colored("Forcing English language for TTS", "blue"))
                else:
                    print(colored("Warning: English not in available languages. Using default language.", "yellow"))
            
            # Generate speech
            with self._gpu_context():
                try:
                    self.tts.tts_to_file(**tts_kwargs)
                except Exception as tts_error:
                    print(colored(f"Error in TTS generation: {str(tts_error)}", "red"))
                    # Try with a simpler model if this fails
                    if self.default_model != "fast":
                        print(colored("Trying with a simpler model...", "yellow"))
                        self._init_tts("fast")
                        # Try again with the new model, but without speaker/language parameters
                        simple_kwargs = {"text": clean_text, "file_path": output_path}
                        try:
                            self.tts.tts_to_file(**simple_kwargs)
                        except Exception as simple_error:
                            print(colored(f"Simple model also failed: {str(simple_error)}", "red"))
                            # Try OpenAI TTS as a last resort
                            return await self._try_openai_tts(clean_text, output_path)
                    else:
                        # Try OpenAI TTS as a last resort
                        return await self._try_openai_tts(clean_text, output_path)
                
            # Verify the file was created
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                print(colored(f"TTS generation failed or produced an empty file: {output_path}", "red"))
                # Try OpenAI TTS as a last resort
                return await self._try_openai_tts(clean_text, output_path)
            
            # Verify the audio is valid by checking its duration
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(output_path)
                duration = len(audio) / 1000.0  # Duration in seconds
                
                # Check if duration is reasonable (at least 1 second per 10 characters)
                expected_min_duration = len(clean_text) / 30  # Rough estimate
                if duration < expected_min_duration and len(clean_text) > 100:
                    print(colored(f"Warning: Audio duration ({duration:.2f}s) is shorter than expected ({expected_min_duration:.2f}s). This might indicate TTS issues.", "yellow"))
                    
                    # If the duration is extremely short, try again with a different model
                    if duration < expected_min_duration / 3:
                        print(colored("Audio duration is extremely short. Trying with a different model...", "yellow"))
                        # Try with a different model
                        if self.default_model != "fast":
                            self._init_tts("fast")
                            # Recursive call with the new model
                            return await self.generate_voice(text, None, "en", emotion, speed, output_path)
                
                print(colored(f"Generated audio duration: {duration:.2f}s for {len(clean_text)} characters", "green"))
            except Exception as audio_check_error:
                print(colored(f"Warning: Could not verify audio duration: {str(audio_check_error)}", "yellow"))
                
            print(colored(f"✓ Generated voice file: {output_path}", "green"))
            return output_path
            
        except Exception as e:
            print(colored(f"Error generating voice: {str(e)}", "red"))
            
            # Try with a different model if this one failed
            if hasattr(self, 'default_model') and self.default_model != "fast":
                print(colored("Trying with a simpler model...", "yellow"))
                try:
                    self._init_tts("fast")
                    # Try again with the new model
                    return await self.generate_voice(text, None, "en", emotion, speed, output_path)
                except Exception as fallback_error:
                    print(colored(f"Fallback model also failed: {str(fallback_error)}", "red"))
            
            # Try OpenAI TTS as a last resort
            return await self._try_openai_tts(clean_text, output_path)
            
    def get_available_voices(self) -> Dict[str, list]:
        """Get available voices grouped by model"""
        voices = {
            'speakers': [],
            'languages': ['en']  # Default to English
        }
        
        # First try to get voices from the current TTS instance
        if hasattr(self, 'tts') and self.tts is not None:
            try:
                # Get speakers if available
                if hasattr(self.tts, 'speakers') and self.tts.speakers is not None:
                    voices['speakers'] = self.tts.speakers
                
                # Get languages if available
                if hasattr(self.tts, 'languages') and self.tts.languages is not None:
                    voices['languages'] = self.tts.languages
            except Exception as e:
                print(colored(f"Error getting voices from current TTS: {str(e)}", "yellow"))
        
        # If we don't have speakers yet, try to initialize models and get their speakers
        if not voices['speakers']:
            # Add default English voices as fallback
            voices['speakers'] = [
                "en_US_female", "en_US_male", "en_UK_female", "en_UK_male",
                "default", "English", "en"
            ]
            
            # Try to get voices from each model
            for model_key, model_name in self.models.items():
                try:
                    # Skip if it's the current model (already tried)
                    if hasattr(self, 'tts') and self.tts is not None and self.tts.model_name == model_name:
                        continue
                        
                    # Try to initialize the model
                    try:
                        temp_tts = TTS(model_name=model_name, progress_bar=False)
                        
                        # Get speakers if available
                        if hasattr(temp_tts, 'speakers') and temp_tts.speakers is not None:
                            # Add to our list of speakers
                            for speaker in temp_tts.speakers:
                                if speaker not in voices['speakers']:
                                    voices['speakers'].append(speaker)
                        
                        # Get languages if available
                        if hasattr(temp_tts, 'languages') and temp_tts.languages is not None:
                            # Add to our list of languages
                            for language in temp_tts.languages:
                                if language not in voices['languages']:
                                    voices['languages'].append(language)
                    except Exception as model_error:
                        print(colored(f"Error initializing model {model_name}: {str(model_error)}", "yellow"))
                        
                except Exception as e:
                    print(colored(f"Error getting voices for {model_key}: {str(e)}", "yellow"))
        
        # Ensure we have at least some voices
        if not voices['speakers']:
            print(colored("No voices found, using default fallback voices", "yellow"))
            voices['speakers'] = ["default", "en"]
            
        # Ensure English is in languages
        if 'en' not in voices['languages']:
            voices['languages'].append('en')
            
        return voices
        
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        try:
            if hasattr(self.tts, "languages"):
                return self.tts.languages
            return ["en"]  # Default to English if no language info available
        except:
            return ["en"]
            
    async def clone_voice(self, reference_audio: str, text: str) -> Optional[str]:
        """Clone a voice from reference audio (XTTS-v2 feature)"""
        try:
            if not hasattr(self, 'tts') or "xtts" not in self.tts.model_name.lower():
                if not self._init_tts("xtts_v2"):
                    raise ValueError("XTTS-v2 model required for voice cloning")
                    
            # Create output directory
            os.makedirs("temp/tts/cloned", exist_ok=True)
            output_path = f"temp/tts/cloned/cloned_{hash(text)}.wav"
            
            # Clean the text (remove emojis)
            clean_text = self._clean_text(text)
            
            # Generate speech with cloned voice
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.tts.tts_to_file(
                    text=clean_text,
                    file_path=output_path,
                    speaker_wav=reference_audio,
                )
            )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(colored(f"✓ Generated cloned voice: {output_path}", "green"))
                return output_path
            else:
                raise ValueError("Generated file is empty or does not exist")
                
        except Exception as e:
            print(colored(f"Error cloning voice: {str(e)}", "red"))
            return None

    async def _try_openai_tts(self, text, output_path):
        """Try to use OpenAI TTS as a fallback"""
        try:
            print(colored("Attempting to use OpenAI TTS as a last resort...", "yellow"))
            
            # Import OpenAI here to avoid dependency issues
            import openai
            from dotenv import load_dotenv
            
            # Load environment variables
            load_dotenv()
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(colored("No OpenAI API key found in environment", "red"))
                return None
                
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Generate MP3 path
            mp3_path = output_path.replace('.wav', '.mp3')
            
            # Generate speech
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",  # Use a neutral voice
                input=text
            )
            
            # Save to file
            response.stream_to_file(mp3_path)
            
            # Verify file exists
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 1000:
                print(colored(f"✓ Generated voice with OpenAI TTS: {mp3_path}", "green"))
                return mp3_path
            else:
                print(colored("OpenAI TTS generated an empty or invalid file", "red"))
                return None
                
        except Exception as openai_error:
            print(colored(f"OpenAI TTS failed: {str(openai_error)}", "red"))
            return None 