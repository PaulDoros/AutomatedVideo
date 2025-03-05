import os
import random
import json
import aiohttp
import asyncio
from pathlib import Path
from termcolor import colored
from content_learning_system import ContentLearningSystem

class JokeProvider:
    """Provides pre-written jokes for tech_humor channel"""
    
    def __init__(self):
        self.jokes_file = Path(os.path.dirname(os.path.abspath(__file__))) / "resources" / "jokes" / "tech_jokes.txt"
        self.jokes = self._load_jokes()
        self.used_jokes_file = Path(os.path.dirname(os.path.abspath(__file__))) / "resources" / "jokes" / "used_jokes.json"
        self.used_jokes = self._load_used_jokes()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        
        # Initialize content learning system
        self.learning_system = ContentLearningSystem()
        
        # Track joke performance
        self.joke_performance_file = Path(os.path.dirname(os.path.abspath(__file__))) / "resources" / "jokes" / "joke_performance.json"
        self.joke_performance = self._load_joke_performance()
        
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
    
    def _load_joke_performance(self):
        """Load joke performance data"""
        if not self.joke_performance_file.exists():
            return {}
            
        try:
            with open(self.joke_performance_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse joke performance file at {self.joke_performance_file}")
            return {}
    
    def _save_used_joke(self, joke):
        """Save a record that a joke has been used"""
        if joke not in self.used_jokes:
            self.used_jokes.append(joke)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.used_jokes_file), exist_ok=True)
            
            with open(self.used_jokes_file, 'w', encoding='utf-8') as f:
                json.dump(self.used_jokes, f, indent=2)
    
    def _save_joke_performance(self):
        """Save joke performance data"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.joke_performance_file), exist_ok=True)
        
        with open(self.joke_performance_file, 'w', encoding='utf-8') as f:
            json.dump(self.joke_performance, f, indent=2)
    
    def update_joke_performance(self, joke, metrics):
        """Update performance metrics for a joke"""
        # Create a hash of the joke for consistent identification
        import hashlib
        joke_hash = hashlib.md5(joke.encode()).hexdigest()
        
        # Update or create performance record
        if joke_hash in self.joke_performance:
            self.joke_performance[joke_hash]['metrics'] = metrics
            self.joke_performance[joke_hash]['updated_at'] = asyncio.get_event_loop().time()
        else:
            self.joke_performance[joke_hash] = {
                'joke': joke,
                'metrics': metrics,
                'created_at': asyncio.get_event_loop().time(),
                'updated_at': asyncio.get_event_loop().time()
            }
        
        # Save performance data
        self._save_joke_performance()
        
        # Record in learning system
        if 'video_id' in metrics:
            self.learning_system.record_content_performance(
                metrics['video_id'],
                'tech_humor',
                joke,
                {
                    'views': metrics.get('views', 0),
                    'likes': metrics.get('likes', 0),
                    'comments': metrics.get('comments', 0),
                    'ctr': metrics.get('ctr', 0.0)
                }
            )
        
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
        
        # Check if joke is blacklisted
        is_blacklisted, reason = self.learning_system.is_content_blacklisted('tech_humor', joke)
        if is_blacklisted:
            print(colored(f"Joke is blacklisted: {reason}. Selecting another joke.", "yellow"))
            # Remove from available jokes and try again
            available_jokes.remove(joke)
            if not available_jokes:
                return "Why did the programmer quit his job? He didn't get arrays! 😂\nFollow for more tech humor! 🚀"
            joke = random.choice(available_jokes)
        
        # Check if joke is too similar to recent content
        is_repetitive, details = self.learning_system.is_content_repetitive('tech_humor', joke)
        if is_repetitive:
            print(colored(f"Joke is too similar to recent content: {details['message']}. Selecting another joke.", "yellow"))
            # Remove from available jokes and try again
            available_jokes.remove(joke)
            if not available_jokes:
                return "Why did the programmer quit his job? He didn't get arrays! 😂\nFollow for more tech humor! 🚀"
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
            
            # Filter out blacklisted jokes
            filtered_matching_jokes = []
            for joke in matching_jokes:
                is_blacklisted, _ = self.learning_system.is_content_blacklisted('tech_humor', joke)
                is_repetitive, _ = self.learning_system.is_content_repetitive('tech_humor', joke)
                if not is_blacklisted and not is_repetitive:
                    filtered_matching_jokes.append(joke)
            
            if filtered_matching_jokes:
                joke = random.choice(filtered_matching_jokes)
                self._save_used_joke(joke)
                return joke
                
        # If no topic or no matching jokes, return a random one
        return self.get_random_joke()
    
    async def generate_joke_with_deepseek(self, topic=None):
        """Generate a new joke using DeepSeek API"""
        if not self.deepseek_api_key:
            print(colored("DeepSeek API key not found. Using pre-written jokes instead.", "yellow"))
            return self.get_random_joke()
            
        print(colored("Generating new joke with DeepSeek API...", "blue"))
        
        # Get content suggestions from learning system
        suggestions = self.learning_system.get_content_suggestions('tech_humor', topic)
        
        # Extract high-performing keywords
        high_performing_keywords = []
        for suggestion in suggestions:
            if suggestion['type'] == 'keywords':
                high_performing_keywords.extend(suggestion.get('data', [])[:5])
        
        # Prepare the prompt
        topic_text = f" about {topic}" if topic else ""
        keyword_text = ""
        if high_performing_keywords:
            keyword_text = f"\n6. Consider using some of these high-performing keywords if relevant: {', '.join(high_performing_keywords)}"
            
        prompt = f"""You are a professional comedy writer for tech humor. 
Create a short, witty tech joke{topic_text} for a YouTube Short.
The joke should be:
1. Genuinely funny and clever
2. Related to programming, technology, or developer culture
3. Short (3-5 lines maximum)
4. Include emojis for visual appeal
5. End with "Follow for more tech humor! 🚀"{keyword_text}

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
                        
                        # Check if joke is blacklisted
                        is_blacklisted, reason = self.learning_system.is_content_blacklisted('tech_humor', joke)
                        if is_blacklisted:
                            print(colored(f"Generated joke is blacklisted: {reason}. Using pre-written joke instead.", "yellow"))
                            return self.get_random_joke()
                        
                        # Check if joke is too similar to recent content
                        is_repetitive, details = self.learning_system.is_content_repetitive('tech_humor', joke)
                        if is_repetitive:
                            print(colored(f"Generated joke is too similar to recent content: {details['message']}. Using pre-written joke instead.", "yellow"))
                            return self.get_random_joke()
                        
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