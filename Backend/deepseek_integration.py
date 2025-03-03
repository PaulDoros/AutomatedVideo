import os
import json
import requests
from typing import Tuple, Optional
from termcolor import colored
from dotenv import load_dotenv

class DeepSeekAPI:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate_script(self, topic: str, paragraph_number: int, custom_prompt: str = None) -> Tuple[bool, Optional[str]]:
        """Generate a script using DeepSeek's API"""
        try:
            # Build the prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = f"""
                Generate a script for a short video about {topic}.
                The script should be {paragraph_number} paragraphs long.
                
                Guidelines:
                - Get straight to the point
                - No unnecessary introductions
                - Make it engaging and conversational
                - Keep each paragraph concise
                - No formatting or markdown
                - No titles or headers
                """

            # Make API call to DeepSeek
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                script = result["choices"][0]["message"]["content"].strip()
                
                # Clean up the script
                script = script.replace("*", "")
                script = script.replace("#", "")
                script = script.split("\n\n")[:paragraph_number]
                script = "\n\n".join(script)
                
                return True, script
            else:
                print(colored("[-] DeepSeek returned an empty response.", "red"))
                return False, None

        except Exception as e:
            print(colored(f"[-] Error generating script with DeepSeek: {str(e)}", "red"))
            return False, None

    async def generate_voice(self, text: str, voice_id: str = "en-US-1") -> Optional[str]:
        """Generate voice using DeepSeek's text-to-speech API"""
        try:
            # Create temp directory if it doesn't exist
            os.makedirs("temp/tts", exist_ok=True)
            output_path = f"temp/tts/deepseek_tts_{hash(text)}.mp3"

            # Make API call to DeepSeek's TTS endpoint
            response = requests.post(
                f"{self.base_url}/audio/speech",
                headers=self.headers,
                json={
                    "text": text,
                    "voice_id": voice_id,
                    "speed": 1.0,
                    "format": "mp3"
                }
            )
            
            response.raise_for_status()
            
            # Save the audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return output_path

        except Exception as e:
            print(colored(f"[-] Error generating voice with DeepSeek: {str(e)}", "red"))
            return None

    def get_available_voices(self) -> list:
        """Get list of available voices from DeepSeek"""
        try:
            response = requests.get(
                f"{self.base_url}/audio/voices",
                headers=self.headers
            )
            
            response.raise_for_status()
            return response.json().get("voices", [])

        except Exception as e:
            print(colored(f"[-] Error getting available voices: {str(e)}", "red"))
            return [] 