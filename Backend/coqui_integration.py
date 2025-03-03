import os
import torch
import asyncio
from typing import Optional, Dict
from TTS.api import TTS
from termcolor import colored
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from contextlib import contextmanager

class CoquiTTSAPI:
    def __init__(self):
        """Initialize Coqui TTS with XTTS-v2 model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/vctk/vits",
            "fast": "tts_models/en/ljspeech/fast_pitch"
        }
        
        # Create cache directory for models
        os.makedirs("assets/tts/model_cache", exist_ok=True)
        
        # Initialize with default model
        self._init_tts()
        
    @contextmanager
    def _torch_load_context(self):
        """Context manager for PyTorch loading compatibility"""
        try:
            # Add XttsConfig to safe globals for PyTorch 2.6+ compatibility
            torch.serialization.add_safe_globals([XttsConfig])
            yield
        finally:
            pass

    def _init_tts(self, model_key: str = "xtts_v2"):
        """Initialize TTS with specified model"""
        try:
            model_name = self.models.get(model_key)
            if not model_name:
                raise ValueError(f"Unknown model key: {model_key}")
                
            print(colored(f"Initializing Coqui TTS with model: {model_name}", "cyan"))
            
            # Use context manager for PyTorch loading
            with self._torch_load_context():
                self.tts = TTS(
                    model_name=model_name,
                    progress_bar=True,
                    gpu=(self.device == "cuda")
                )
            
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
            
    async def generate_voice(self, text: str, speaker: str = None, language: str = "en", 
                           emotion: str = "neutral", speed: float = 1.0) -> Optional[str]:
        """Generate voice using Coqui TTS"""
        try:
            # Create output directory
            os.makedirs("temp/tts", exist_ok=True)
            output_path = f"temp/tts/coqui_{hash(text)}.wav"
            
            # Add SSML-like markup for emotion and speed control
            if emotion != "neutral":
                text = f"[{emotion}] {text}"
            
            # Generate speech
            print(colored(f"Generating speech with Coqui TTS ({self.device})", "cyan"))
            print(colored(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}", "cyan"))
            
            # Ensure TTS is initialized
            if not hasattr(self, 'tts'):
                if not self._init_tts():
                    raise ValueError("Failed to initialize TTS model")
            
            # Prepare TTS parameters
            tts_kwargs = {"text": text, "file_path": output_path, "speed": speed}
            
            # Only add speaker if model is multi-speaker
            if self.is_multi_speaker:
                if not speaker and self.speakers:
                    speaker = self.speakers[0]  # Use first available speaker as default
                tts_kwargs["speaker"] = speaker
                
            # Only add language if model is multilingual
            if self.is_multilingual:
                if language not in self.languages:
                    language = "en"  # Fallback to English if language not supported
                tts_kwargs["language"] = language
            
            # Use asyncio to run TTS in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.tts.tts_to_file(**tts_kwargs)
            )
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(colored(f"✓ Generated voice file: {output_path}", "green"))
                return output_path
            else:
                raise ValueError("Generated file is empty or does not exist")
                
        except Exception as e:
            print(colored(f"Error generating voice: {str(e)}", "red"))
            
            # Try fallback model if primary fails
            if hasattr(self, 'tts') and self.tts.model_name == self.models["xtts_v2"]:
                print(colored("Attempting fallback to faster model...", "yellow"))
                if self._init_tts("fast"):
                    try:
                        return await self.generate_voice(text, speaker=None, language="en", emotion=emotion, speed=speed)
                    except Exception as fallback_error:
                        print(colored(f"Fallback also failed: {str(fallback_error)}", "red"))
                        
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
            
            # Generate speech with cloned voice
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.tts.tts_to_file(
                    text=text,
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