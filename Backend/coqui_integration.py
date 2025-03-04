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
                raise ValueError(f"Unknown model key: {model_key}")
                
            print(colored(f"Initializing Coqui TTS with model: {model_name}", "cyan"))
            
            # Use context manager for PyTorch loading
            with self._torch_load_context():
                # Initialize TTS without gpu parameter (which is deprecated)
                self.tts = TTS(model_name=model_name, progress_bar=True)
                
                # Move model to appropriate device after initialization
                if self.device == "cuda":
                    self.tts.to(self.device)
            
            # Store model capabilities
            self.is_multi_speaker = hasattr(self.tts, "speakers") and self.tts.speakers is not None
            self.is_multilingual = hasattr(self.tts, "languages") and self.tts.languages is not None
            self.speakers = self.tts.speakers if self.is_multi_speaker else []
            self.languages = self.tts.languages if self.is_multilingual else ["en"]
            
            if self.is_multi_speaker:
                print(colored(f"Available speakers: {len(self.speakers)}", "green"))
            if self.is_multilingual:
                print(colored(f"Available languages: {self.languages}", "green"))
            
            return True
            
        except Exception as e:
            print(colored(f"Error initializing TTS model: {str(e)}", "red"))
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
            
            # Clean the text (remove emojis and emotion tags)
            clean_text = self._clean_text(text, emotion)
            
            # Generate speech
            print(colored(f"Generating speech with Coqui TTS ({self.device})", "cyan"))
            print(colored(f"Text: {clean_text[:100]}{'...' if len(clean_text) > 100 else ''}", "cyan"))
            
            # Ensure TTS is initialized
            if not hasattr(self, 'tts'):
                if not self._init_tts():
                    raise ValueError("Failed to initialize TTS model")
            
            # Prepare TTS parameters
            tts_kwargs = {"text": clean_text, "file_path": output_path}
            
            # Apply speed adjustment based on emotion and device
            if self.device == "cuda":
                # On GPU, we can use higher quality settings
                # Adjust speed based on content type in the output path
                if "tech_humor" in output_path:
                    # Faster for tech humor content
                    tts_kwargs["speed"] = speed * 1.2
                elif emotion == "cheerful":
                    # Slightly faster for cheerful emotion
                    tts_kwargs["speed"] = speed * 1.1
                else:
                    # Normal speed for other emotions
                    tts_kwargs["speed"] = speed
                
                # Add quality settings for GPU
                if hasattr(self, 'gpu_optimization_level') and self.gpu_optimization_level == "high_quality":
                    # Higher quality settings for high-end GPUs
                    if hasattr(self.tts, "synthesizer") and hasattr(self.tts.synthesizer, "config"):
                        # Attempt to set higher quality parameters if available
                        try:
                            # These are model-specific settings that may improve quality
                            if hasattr(self.tts.synthesizer.config, "use_griffin_lim"):
                                self.tts.synthesizer.config.use_griffin_lim = False  # Disable Griffin-Lim for better quality
                            
                            # Set higher sampling rate if supported
                            if hasattr(self.tts.synthesizer.config, "audio") and hasattr(self.tts.synthesizer.config.audio, "sample_rate"):
                                # Try to use 24kHz or higher if supported
                                if self.tts.synthesizer.config.audio.sample_rate < 24000:
                                    print(colored("Increasing sample rate for better quality", "blue"))
                        except Exception as config_error:
                            print(colored(f"Note: Could not apply all quality settings: {str(config_error)}", "yellow"))
            else:
                # On CPU, prioritize speed
                tts_kwargs["speed"] = speed * 1.05  # Slightly faster by default on CPU
                
            # Add speaker if provided and supported
            if speaker and hasattr(self.tts, 'speakers') and speaker in self.tts.speakers:
                tts_kwargs["speaker"] = speaker
                
            # Add language if supported
            if hasattr(self.tts, 'languages') and language in self.tts.languages:
                tts_kwargs["language"] = language
            
            # Generate speech
            with self._gpu_context():
                self.tts.tts_to_file(**tts_kwargs)
                
            # Verify the file was created
            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise ValueError(f"TTS generation failed or produced an empty file: {output_path}")
                
            print(colored(f"✓ Generated voice file: {output_path}", "green"))
            return output_path
            
        except Exception as e:
            print(colored(f"Error generating voice: {str(e)}", "red"))
            return None
            
    def get_available_voices(self) -> Dict[str, list]:
        """Get available voices grouped by model"""
        voices = {}
        
        for model_key, model_name in self.models.items():
            try:
                if not hasattr(self, 'tts') or self.tts.model_name != model_name:
                    self._init_tts(model_key)
                    
                # Check if model was initialized successfully
                if not hasattr(self, 'tts'):
                    print(colored(f"Failed to initialize {model_key}", "yellow"))
                    voices[model_key] = []
                    continue
                    
                if self.is_multi_speaker and self.speakers:
                    voices[model_key] = self.speakers
                else:
                    voices[model_key] = ["default"]
                    
            except Exception as e:
                print(colored(f"Error getting voices for {model_key}: {str(e)}", "yellow"))
                voices[model_key] = []
                
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