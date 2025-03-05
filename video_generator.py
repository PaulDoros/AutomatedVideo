class VideoGenerator:
    def __init__(self):
        # ... existing initialization code ...

        # Initialize TTS engines
        self.tts_engines = {
            'openai': self._generate_tts_openai,
            'coqui': self._generate_tts_coqui,
            'tiktok': self._generate_tts_tiktok
        }
        
        # Initialize Coqui TTS
        self._init_coqui_tts()

    def _init_coqui_tts(self):
        """Initialize Coqui TTS with optimal models"""
        try:
            # Initialize with the best model for YouTube Shorts
            self.coqui_models = {
                'tech_humor': {
                    'model': "tts_models/en/jenny/jenny",  # Natural female voice
                    'description': 'Energetic female voice perfect for tech content'
                },
                'ai_money': {
                    'model': "tts_models/en/vctk/vits",  # Multi-speaker model
                    'speaker': "p273",  # Professional male voice
                    'description': 'Professional male voice for business content'
                },
                'default': {
                    'model': "tts_models/en/vctk/vits",
                    'speaker': "p225",  # Clear female voice
                    'description': 'Clear, engaging voice for general content'
                }
            }
            
            # Load the default model first
            self.coqui_tts = TTS(
                model_name=self.coqui_models['default']['model'],
                progress_bar=True,
                gpu=torch.cuda.is_available()
            )
            print(colored("✓ Coqui TTS initialized successfully", "green"))
            
        except Exception as e:
            print(colored(f"! Warning: Could not initialize Coqui TTS: {str(e)}", "yellow"))
            self.coqui_tts = None

    async def _generate_tts_coqui(self, script, channel_type):
        """Generate TTS using Coqui"""
        try:
            if not self.coqui_tts:
                raise ValueError("Coqui TTS not initialized")

            # Get the appropriate model for this channel
            model_config = self.coqui_models.get(channel_type, self.coqui_models['default'])
            
            # Load the specific model if different from current
            if self.coqui_tts.model_name != model_config['model']:
                self.coqui_tts = TTS(
                    model_name=model_config['model'],
                    progress_bar=True,
                    gpu=torch.cuda.is_available()
                )
            
            # Clean up the script
            clean_script = '\n'.join(
                line.strip().strip('"') 
                for line in script.split('\n') 
                if line.strip() and not line[0].isdigit()
            )
            
            # Set up the output paths
            temp_path = f"temp/tts/{channel_type}_raw.wav"
            final_path = f"temp/tts/{channel_type}_latest.mp3"
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            # Generate the speech
            speaker = model_config.get('speaker', None)
            self.coqui_tts.tts_to_file(
                text=clean_script,
                speaker=speaker,
                file_path=temp_path
            )
            
            # Convert to MP3 and apply audio enhancements
            audio_clip = AudioFileClip(temp_path)
            enhanced_audio = audio_clip.audio_fadeout(0.5)  # Add fade out
            enhanced_audio.write_audiofile(final_path)
            
            # Clean up
            audio_clip.close()
            enhanced_audio.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(colored(f"✓ Generated Coqui TTS audio using {model_config['description']}", "green"))
            return final_path
            
        except Exception as e:
            print(colored(f"Coqui TTS generation failed: {str(e)}", "red"))
            return None

    async def _generate_tts(self, script, channel_type):
        """Generate TTS using the preferred engine"""
        # Try Coqui first, fall back to OpenAI if it fails
        tts_path = await self._generate_tts_coqui(script, channel_type)
        if not tts_path:
            print(colored("Falling back to OpenAI TTS...", "yellow"))
            tts_path = await self._generate_tts_openai(script, channel_type)
        return tts_path

    async def _generate_tts_openai(self, script, channel_type):
        """Original OpenAI TTS implementation"""
        # ... existing OpenAI TTS code ... 