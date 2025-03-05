import os
import random
import json
import aiohttp
import asyncio
from pathlib import Path
from termcolor import colored

class JokeProvider:
    """Provides pre-written jokes for tech_humor channel"""
    
    def __init__(self):
        self.jokes_file = Path(os.path.dirname(os.path.abspath(__file__))) / "resources" / "jokes" / "tech_jokes.txt"
        self.jokes = self._load_jokes()
        self.used_jokes_file = Path(os.path.dirname(os.path.abspath(__file__))) / "resources" / "jokes" / "used_jokes.json"
        self.used_jokes = self._load_used_jokes()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
    def _load_jokes(self):
        """Load jokes from the jokes file"""
        if not self.jokes_file.exists():
            print(f"Warning: Jokes file not found at {self.jokes_file}")
            return []
            
        with open(self.jokes_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split by the separator (---) and filter out empty jokes
        jokes = [joke.strip() for joke in content.split('---') if joke.strip()]
        print(f"Loaded {len(jokes)} jokes from {self.jokes_file}")
        return jokes
    
    def _load_used_jokes(self):
        """Load record of used jokes"""
        if not self.used_jokes_file.exists():
            return []
            
        try:
            with open(self.used_jokes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse used jokes file at {self.used_jokes_file}")
            return []
    
    def _save_used_joke(self, joke):
        """Save a record that a joke has been used"""
        if joke not in self.used_jokes:
            self.used_jokes.append(joke)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.used_jokes_file), exist_ok=True)
            
            with open(self.used_jokes_file, 'w', encoding='utf-8') as f:
                json.dump(self.used_jokes, f, indent=2)
        
    def get_random_joke(self):
        """Get a random joke from the collection"""
        if not self.jokes:
            return "Why did the programmer quit his job? He didn't get arrays! 😂\nFollow for more tech humor! 🚀"
        
        # Filter out jokes that have been used
        available_jokes = [joke for joke in self.jokes if joke not in self.used_jokes]
        
        # If all jokes have been used, reset the used jokes list
        if not available_jokes:
            print(colored("All pre-written jokes have been used. Resetting used jokes list.", "yellow"))
            self.used_jokes = []
            available_jokes = self.jokes
            
        # Get a random joke
        joke = random.choice(available_jokes)
        self._save_used_joke(joke)
        return joke
        
    def get_joke_for_topic(self, topic=None):
        """Get a joke that might be related to the topic if possible"""
        if not self.jokes:
            return "Why did the programmer quit his job? He didn't get arrays! 😂\nFollow for more tech humor! 🚀"
            
        # Filter out jokes that have been used
        available_jokes = [joke for joke in self.jokes if joke not in self.used_jokes]
        
        # If all jokes have been used, reset the used jokes list
        if not available_jokes:
            print(colored("All pre-written jokes have been used. Resetting used jokes list.", "yellow"))
            self.used_jokes = []
            available_jokes = self.jokes
            
        # If we have a topic, try to find a joke that contains the topic
        if topic:
            topic_lower = topic.lower()
            matching_jokes = [joke for joke in available_jokes if topic_lower in joke.lower()]
            if matching_jokes:
                joke = random.choice(matching_jokes)
                self._save_used_joke(joke)
                return joke
                
        # If no topic or no matching jokes, return a random one
        joke = random.choice(available_jokes)
        self._save_used_joke(joke)
        return joke
    
    async def generate_joke_with_deepseek(self, topic=None):
        """Generate a new joke using DeepSeek API"""
        if not self.deepseek_api_key:
            print(colored("DeepSeek API key not found. Using pre-written jokes instead.", "yellow"))
            return self.get_random_joke()
            
        print(colored("Generating new joke with DeepSeek API...", "blue"))
        
        # Prepare the prompt
        topic_text = f" about {topic}" if topic else ""
        prompt = f"""You are a professional comedy writer for tech humor. 
Create a short, witty tech joke{topic_text} for a YouTube Short.
The joke should be:
1. Genuinely funny and clever
2. Related to programming, technology, or developer culture
3. Short (3-5 lines maximum)
4. Include emojis for visual appeal
5. End with "Follow for more tech humor! 🚀"

Format the joke with each sentence on a new line.
DO NOT include any explanations, just the joke itself.
"""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.deepseek_api_key}"
                }
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                }
                
                async with session.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        joke = result["choices"][0]["message"]["content"].strip()
                        print(colored("Successfully generated joke with DeepSeek API", "green"))
                        return joke
                    else:
                        error_text = await response.text()
                        print(colored(f"Error from DeepSeek API: {error_text}", "red"))
                        return self.get_random_joke()
                        
        except Exception as e:
            print(colored(f"Error generating joke with DeepSeek API: {str(e)}", "red"))
            return self.get_random_joke()
    
    async def get_joke(self, topic=None, use_ai=False):
        """Get a joke, either from pre-written collection or generated with AI"""
        # If we have jokes available and not explicitly asked to use AI, use pre-written jokes
        if self.jokes and not use_ai:
            # Check if we've used all jokes
            available_jokes = [joke for joke in self.jokes if joke not in self.used_jokes]
            if available_jokes:
                return self.get_joke_for_topic(topic)
        
        # If we've used all jokes or explicitly asked to use AI, generate with DeepSeek
        print(colored("Using DeepSeek API to generate a fresh joke", "blue"))
        return await self.generate_joke_with_deepseek(topic) 