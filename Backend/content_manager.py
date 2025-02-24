import json
import sqlite3
from datetime import datetime
import openai
from termcolor import colored
import os
from dotenv import load_dotenv

load_dotenv()

# Channel Definitions with researched content strategies
CHANNEL_STRATEGIES = {
    'tech_humor': {  # Existing LOLRoboJAJA
        'name': 'Tech Humor',
        'description': 'Funny takes on tech life, AI fails, and programmer humor',
        'target_audience': ['developers', 'tech enthusiasts', 'students'],
        'content_types': [
            {'type': 'AI Fails', 'weight': 30},
            {'type': 'Programming Jokes', 'weight': 25},
            {'type': 'Tech Life Struggles', 'weight': 25},
            {'type': 'Funny AI Interactions', 'weight': 20}
        ],
        'prompt_template': """
        Create a FUNNY and entertaining script about {topic}.
        Think stand-up comedy style, not technical explanation.
        Include:
        - A relatable tech/programming struggle
        - Exaggerated situations
        - Punchlines and comedic timing
        Make it genuinely humorous!
        """,
        'hashtags': ['techhumor', 'programmerhumor', 'aifails', 'coding', 'techtok'],
        'voice': 'en_us_006'  # Cheerful female voice
    },
    'ai_money': {  # Existing AI Money Hacks
        'name': 'AI Money Hacks',
        'description': 'AI tools and strategies for making money',
        'target_audience': ['entrepreneurs', 'side-hustlers', 'tech enthusiasts'],
        'content_types': [
            {'type': 'AI Money Tools', 'weight': 30},
            {'type': 'Side Hustle Tips', 'weight': 25},
            {'type': 'AI Business Ideas', 'weight': 25},
            {'type': 'Success Stories', 'weight': 20}
        ],
        'prompt_template': """
        Create an engaging video about making money with AI.
        Focus on practical, actionable tips.
        Include:
        - A specific AI tool or strategy
        - How to get started
        - Potential earnings
        Keep it motivational but realistic!
        """,
        'hashtags': ['aimoney', 'sidehustle', 'makemoney', 'aitools', 'moneytips'],
        'voice': 'en_us_001'  # Professional male voice
    },
    'baby_tips': {  # New channel
        'name': 'Smart Parenting Tips',
        'description': 'Daily tips for new parents and baby development',
        'target_audience': ['new parents', 'expecting parents', 'caregivers'],
        'content_types': [
            {'type': 'Baby Development', 'weight': 30},
            {'type': 'Parenting Hacks', 'weight': 25},
            {'type': 'Baby Care Tips', 'weight': 25},
            {'type': 'Parent Life Hacks', 'weight': 20}
        ],
        'prompt_template': """
        Create a helpful parenting tip video about {topic}.
        Make it practical and easy to follow.
        Include:
        - The main parenting challenge
        - A simple solution or tip
        - Why it works
        Keep it supportive and encouraging!
        """,
        'hashtags': ['babytips', 'parenting', 'newborn', 'momtips', 'babylife'],
        'voice': 'en_us_002'  # Warm female voice
    },
    'quick_meals': {  # New channel
        'name': 'Quick & Healthy Meals',
        'description': '15-minute healthy meal ideas and recipes',
        'target_audience': ['busy parents', 'students', 'working professionals'],
        'content_types': [
            {'type': 'Quick Recipes', 'weight': 30},
            {'type': 'Meal Prep', 'weight': 25},
            {'type': 'Healthy Snacks', 'weight': 25},
            {'type': 'Budget Meals', 'weight': 20}
        ],
        'prompt_template': """
        Create a quick recipe video about {topic}.
        Focus on speed and simplicity.
        Include:
        - Ingredients needed
        - Step-by-step instructions
        - Time-saving tips
        Keep it under 15 minutes!
        """,
        'hashtags': ['quickmeals', 'easyrecipes', 'healthyfood', 'mealprep', 'cooking'],
        'voice': 'en_us_002'  # Warm female voice
    },
    'fitness_motivation': {  # New channel
        'name': 'Daily Fitness Motivation',
        'description': 'Quick workouts and fitness motivation',
        'target_audience': ['fitness beginners', 'busy professionals', 'home workout enthusiasts'],
        'content_types': [
            {'type': 'Quick Workouts', 'weight': 30},
            {'type': 'Fitness Tips', 'weight': 25},
            {'type': 'Motivation', 'weight': 25},
            {'type': 'Form Guides', 'weight': 20}
        ],
        'prompt_template': """
        Create an motivational fitness video about {topic}.
        Make it accessible for beginners.
        Include:
        - The main exercise or tip
        - Proper form explanation
        - Motivation element
        Keep it energetic and encouraging!
        """,
        'hashtags': ['fitness', 'workout', 'motivation', 'healthylifestyle', 'exercise'],
        'voice': 'en_us_001'  # Professional male voice
    }
}

class ContentManager:
    def __init__(self):
        self.db_path = 'content_database.db'
        self.setup_database()
        self.openai = openai
        self.openai.api_key = os.getenv('OPENAI_API_KEY')

    def setup_database(self):
        """Initialize SQLite database for content tracking"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables for content tracking
        c.execute('''CREATE TABLE IF NOT EXISTS generated_content
                    (id INTEGER PRIMARY KEY,
                     channel TEXT,
                     topic TEXT,
                     title TEXT,
                     content_type TEXT,
                     created_date DATE,
                     posted_date DATE,
                     performance_score REAL,
                     hashtags TEXT)''')
        
        # Create table for topic tracking
        c.execute('''CREATE TABLE IF NOT EXISTS used_topics
                    (id INTEGER PRIMARY KEY,
                     channel TEXT,
                     topic TEXT,
                     used_date DATE,
                     success_score REAL)''')
        
        conn.commit()
        conn.close()

    def generate_unique_content(self, channel):
        """Generate unique content ideas based on channel strategy"""
        try:
            strategy = CHANNEL_STRATEGIES[channel]
            
            # Check recently used topics
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''SELECT topic FROM used_topics 
                        WHERE channel = ? 
                        ORDER BY used_date DESC LIMIT 20''', (channel,))
            used_topics = [row[0] for row in c.fetchall()]
            conn.close()

            # Generate new content ideas using GPT-4
            prompt = f"""
            Generate a unique video idea for a {strategy['name']} channel.
            Target audience: {', '.join(strategy['target_audience'])}
            Channel description: {strategy['description']}
            
            The idea should NOT be similar to these recent topics:
            {', '.join(used_topics)}
            
            Please provide:
            1. Title
            2. Brief description (2-3 sentences)
            3. Key points to cover (3-5 points)
            4. Attention-grabbing opening line
            5. Suggested hashtags
            
            Make it engaging, informative, and suitable for short-form video.
            """

            response = self.openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )

            content = response.choices[0].message.content
            
            # Save to database
            self.save_content_idea(channel, content)
            
            return content

        except Exception as e:
            print(colored(f"Error generating content: {str(e)}", "red"))
            return None

    def save_content_idea(self, channel, content):
        """Save generated content to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        now = datetime.now().date()
        
        c.execute('''INSERT INTO generated_content 
                    (channel, topic, created_date) 
                    VALUES (?, ?, ?)''', 
                    (channel, content, now))
        
        conn.commit()
        conn.close()

    def get_next_video_idea(self, channel):
        """Get the next video idea to produce"""
        strategy = CHANNEL_STRATEGIES[channel]
        content = self.generate_unique_content(channel)
        
        return {
            'channel': channel,
            'strategy': strategy,
            'content': content,
            'voice': strategy['voice'],
            'hashtags': strategy['hashtags']
        }

    def track_performance(self, video_id, metrics):
        """Track video performance for better content planning"""
        # Add performance tracking logic here
        pass 