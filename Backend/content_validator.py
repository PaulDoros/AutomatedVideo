from termcolor import colored
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from typing import Tuple

class ContentValidator:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Content quality standards
        self.standards = {
            'min_words': 150,
            'max_words': 500,
            'min_sections': 3,
            'ideal_duration': {
                'shorts': (30, 70),    # Updated to allow up to 70 seconds
                'regular': (8, 12)      # 8-12 minutes
            },
            'required_elements': [
                'hook',
                'main_points',
                'conclusion',
                'call_to_action'
            ]
        }

    def validate_script(self, script, channel_type):
        """Validate script content and structure"""
        try:
            # First clean up the script - handle line breaks properly
            lines = []
            for line in script.split('\n'):
                line = line.strip()
                if line and not line.startswith('['):  # Skip scene directions
                    # Keep emojis with their lines
                    if not any(line.startswith(c) for c in ['🏆', '☕️', '💻', '🚀', '✨']):
                        lines.append(line)
                    else:
                        if lines:
                            lines[-1] = lines[-1] + ' ' + line
                        else:
                            lines.append(line)

            # Calculate metrics
            total_words = sum(len(re.findall(r'\w+', line)) for line in lines)
            estimated_duration = total_words * 0.4  # 0.4 seconds per word

            # Print analysis
            print(colored("\nScript Analysis:", "blue"))
            print(colored(f"Lines: {len(lines)}", "cyan"))
            print(colored(f"Words: {total_words}", "cyan"))
            print(colored(f"Estimated Duration: {estimated_duration:.1f}s", "cyan"))
            
            # Print the processed script
            print(colored("\nProcessed Script:", "blue"))
            for i, line in enumerate(lines, 1):
                print(colored(f"{i}. {line}", "cyan"))

            # More flexible validation
            if estimated_duration > 120:  # Allow up to 2 minutes but warn
                print(colored(f"\nWarning: Script duration ({estimated_duration:.1f}s) is longer than ideal for shorts", "yellow"))
            
            # Check for minimum content and emojis
            if total_words < 30:
                return False, {"message": "Script too short, need at least 30 words"}

            emoji_count = sum(1 for line in lines if any(emoji in line for emoji in 
                ['☕️', '💻', '🚀', '✨', '🔥', '🎯', '💡', '🤖', '👨‍💻', '👩‍💻', '🎉', '🌟']))
            if emoji_count < 2:
                return False, {"message": "Script must include at least 2 emojis"}

            # Save successful scripts for learning
            if 25 <= estimated_duration <= 65:  # This is our ideal range
                self._save_successful_script(channel_type, script, {
                    "duration": estimated_duration,
                    "words": total_words,
                    "lines": len(lines),
                    "emoji_count": emoji_count
                })

            return True, {
                "lines": len(lines),
                "words": total_words,
                "estimated_duration": estimated_duration,
                "is_ideal_length": 25 <= estimated_duration <= 65
            }

        except Exception as e:
            print(colored(f"Error validating script: {str(e)}", "red"))
            return False, str(e)

    def _save_successful_script(self, channel_type: str, script: str, metrics: dict):
        """Save successful scripts to learn from"""
        try:
            cache_file = f"cache/scripts/{channel_type}_successful.json"
            os.makedirs("cache/scripts", exist_ok=True)
            
            successful_scripts = []
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    successful_scripts = json.load(f)
            
            successful_scripts.append({
                'script': script,
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            # Keep last 20 successful scripts
            successful_scripts = successful_scripts[-20:]
            
            with open(cache_file, 'w') as f:
                json.dump(successful_scripts, f, indent=2)
                
        except Exception as e:
            print(colored(f"Error saving successful script: {str(e)}", "yellow"))

    def estimate_video_length(self, script):
        """Estimate video length based on word count and pacing"""
        words = len(script.split())
        # Average speaking rate (words per minute)
        speaking_rate = 150
        # Add time for visuals, transitions, etc.
        base_duration = (words / speaking_rate) * 60
        total_duration = base_duration * 1.2  # 20% buffer for pacing
        return total_duration

    def check_content_completeness(self, content):
        """Check if all necessary content elements are present"""
        required_elements = {
            'script': True,
            'title': True,
            'description': True,
            'tags': True,
            'thumbnail_text': True
        }
        
        missing_elements = []
        for element, required in required_elements.items():
            if required and not content.get(element):
                missing_elements.append(element)
        
        return len(missing_elements) == 0, missing_elements

    def get_channel_standards(self, channel_type):
        """Get channel-specific quality standards"""
        standards = {
            'tech_humor': {
                'min_jokes': 3,
                'max_technical_terms': 5,
                'tone': 'humorous',
                'required_sections': ['setup', 'punchline', 'tech_explanation'],
                'thumbnail_style': 'funny_tech'
            },
            'ai_money': {
                'min_steps': 3,
                'required_data': ['earnings_example', 'proof', 'timeline'],
                'tone': 'professional',
                'required_sections': ['opportunity', 'method', 'results'],
                'thumbnail_style': 'business'
            },
            'baby_tips': {
                'safety_check': True,
                'medical_disclaimer': True,
                'tone': 'warm_supportive',
                'required_sections': ['problem', 'solution', 'safety_notes'],
                'thumbnail_style': 'parenting'
            },
            'quick_meals': {
                'ingredient_count': '5-10',
                'prep_time': '15min',
                'nutrition_info': True,
                'required_sections': ['ingredients', 'steps', 'tips'],
                'thumbnail_style': 'food'
            },
            'fitness_motivation': {
                'exercise_safety': True,
                'difficulty_level': True,
                'modifications': True,
                'required_sections': ['warmup', 'workout', 'cooldown'],
                'thumbnail_style': 'fitness'
            }
        }
        return standards.get(channel_type, {})

class ScriptGenerator:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.models = {
            'primary': 'gpt-4-turbo-preview',  # Latest GPT-4 Turbo model
            'fallback': 'gpt-3.5-turbo'  # Cost-effective fallback
        }
        self.cost_per_token = {
            'gpt-4-turbo-preview': {
                'input': 0.00001,   # $0.01 per 1K tokens
                'output': 0.00003   # $0.03 per 1K tokens
            },
            'gpt-3.5-turbo': {
                'input': 0.000001,  # $0.001 per 1K tokens
                'output': 0.000002  # $0.002 per 1K tokens
            }
        }
        self.total_cost = 0

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost of API call"""
        model_costs = self.cost_per_token.get(model, self.cost_per_token['gpt-3.5-turbo'])
        input_cost = input_tokens * model_costs['input']
        output_cost = output_tokens * model_costs['output']
        total = input_cost + output_cost
        self.total_cost += total
        return total

    async def generate_script(self, topic: str, channel_type: str) -> Tuple[bool, str]:
        """Generate an engaging script for short-form video"""
        try:
            # Get enhanced system prompt for O1
            system_prompt = self.get_system_prompt(channel_type)
            
            # Get channel-specific prompt
            content_prompt = self.get_channel_prompt(channel_type, topic)
            
            response = self.client.chat.completions.create(
                model=self.models['primary'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_prompt}
                ],
                temperature=0.7
            )
            
            # Calculate and log cost
            cost = self._calculate_cost(
                response.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            print(colored(f"\nAPI Cost: ${cost:.4f}", "yellow"))
            print(colored(f"Total Cost: ${self.total_cost:.4f}", "yellow"))
            
            script = response.choices[0].message.content.strip()
            
            # Validate script
            validator = ContentValidator()
            is_valid, analysis = validator.validate_script(script, channel_type)
            
            if is_valid:
                print(colored("\nGenerated Script:", "green"))
                print(colored(script, "cyan"))
                print(colored(f"\nEstimated Duration: {analysis.get('estimated_duration', 0):.1f}s", "blue"))
                return True, script
            
            # If primary model fails, try fallback
            print(colored("\nTrying fallback model...", "yellow"))
            response = self.client.chat.completions.create(
                model=self.models['fallback'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_prompt}
                ],
                temperature=0.7
            )
            
            script = response.choices[0].message.content.strip()
            is_valid, analysis = validator.validate_script(script, channel_type)
            
            if is_valid:
                return True, script
                
            print(colored(f"Script validation failed: {analysis.get('message', 'Unknown error')}", "yellow"))
            return False, None

        except Exception as e:
            print(colored(f"Error generating script: {str(e)}", "red"))
            return False, None

    def track_usage(self, script, is_premium):
        """Track token usage and estimated costs"""
        token_count = len(script.split()) * 1.3  # Rough token estimate
        model = self.primary_model if is_premium else self.fallback_model
        estimated_cost = (token_count / 1000) * model['cost']
        
        print(colored(f"\nUsage Statistics:", "blue"))
        print(colored(f"- Model: {model['name']}", "cyan"))
        print(colored(f"- Estimated tokens: {int(token_count)}", "cyan"))
        print(colored(f"- Estimated cost: ${estimated_cost:.4f}", "cyan"))

    def get_system_prompt(self, channel_type):
        """Get enhanced system prompt for O1's advanced reasoning"""
        prompts = {
            'tech_humor': """You are an expert comedy writer specializing in tech humor with a deep understanding of both technology and comedic timing.

Key Capabilities:
- Craft relatable tech jokes that resonate with both beginners and experts
- Balance technical accuracy with accessible humor
- Create memorable analogies that simplify complex concepts
- Maintain perfect comedic timing in script structure
- Incorporate current tech trends and common developer experiences

Style Guidelines:
- Use conversational, natural language
- Include self-deprecating humor when appropriate
- Reference popular tech culture and memes tastefully
- Create visual humor through descriptive scenarios
- Build tension and release through technical storytelling""",

            'ai_money': """You are a strategic AI business consultant with expertise in monetization and practical implementation.

Key Capabilities:
- Transform complex AI concepts into actionable business opportunities
- Provide realistic earning potential with detailed breakdowns
- Identify market gaps and emerging AI trends
- Balance optimism with practical limitations
- Create step-by-step implementation plans

Style Guidelines:
- Use data-driven examples with specific numbers
- Include risk assessment and mitigation strategies
- Focus on sustainable, ethical AI applications
- Address common obstacles and solutions
- Maintain professional yet accessible language""",

            'baby_tips': """You are a compassionate parenting expert with extensive knowledge in child development and family dynamics.

Key Capabilities:
- Provide evidence-based parenting advice
- Balance expert knowledge with emotional support
- Address common parenting anxieties
- Incorporate latest pediatric research
- Create safe, practical solutions

Style Guidelines:
- Use warm, supportive language
- Include expert citations when needed
- Address parent guilt and concerns proactively
- Balance ideals with realistic expectations
- Maintain safety as the top priority""",

            'quick_meals': """You are an innovative culinary expert specializing in efficient, healthy meal preparation.

Key Capabilities:
- Design time-optimized cooking processes
- Balance nutrition with convenience
- Create flavor-packed simple recipes
- Incorporate practical kitchen hacks
- Adapt recipes for different skill levels

Style Guidelines:
- Use clear, sequential instructions
- Include time-saving techniques
- Focus on accessible ingredients
- Address common cooking mistakes
- Balance health with taste appeal""",

            'fitness_motivation': """You are an inspiring fitness coach with expertise in exercise science and behavioral psychology.

Key Capabilities:
- Create engaging workout narratives
- Balance motivation with proper form
- Design adaptable exercise routines
- Incorporate progressive challenge levels
- Blend physical and mental wellness

Style Guidelines:
- Use energetic, encouraging language
- Include form cues and safety checks
- Address common fitness fears
- Create inclusive, adaptable content
- Balance intensity with accessibility"""
        }
        
        base_prompt = prompts.get(channel_type, "You are a professional content creator.")
        
        advanced_elements = """
        Leverage advanced reasoning to:
        1. Anticipate and address viewer questions before they arise
        2. Create multi-layered content that rewards repeated viewing
        3. Incorporate subtle psychological triggers for engagement
        4. Build complex narrative structures in simple language
        5. Use advanced persuasion techniques ethically
        
        Content Enhancement:
        - Layer information for different expertise levels
        - Create memorable "aha" moments
        - Plant hooks for future content
        - Build parasocial connection through shared experiences
        - Use advanced storytelling techniques
        
        Engagement Optimization:
        - Craft precise emotional triggers
        - Use linguistic patterns for better retention
        - Create strategic information gaps
        - Implement advanced pacing techniques
        - Design multi-platform content hooks
        
        Format your response with clear section markers, timing, and visual cues.
        Focus on creating content that feels authentic while maintaining high production value.
        """
        
        return f"{base_prompt}\n\n{advanced_elements}"

    def get_enhanced_prompt(self, topic, channel_type):
        """Get enhanced prompt with specific instructions"""
        base_prompt = self.get_channel_prompt(channel_type, topic)
        
        # Add quality-focused instructions
        enhanced_prompt = f"""
        Create a high-quality, engaging script about {topic}.
        
        Key requirements:
        1. Strong hook in first 5 seconds
        2. Clear, concise explanations
        3. Engaging storytelling elements
        4. Actionable takeaways
        5. Natural, conversational tone
        6. Appropriate pacing for short-form video
        
        {base_prompt}
        
        Additional guidelines:
        - Include emotional triggers and relatable examples
        - Use power words and engaging language
        - Create clear visual descriptions
        - End with a strong call-to-action
        
        Format the script with clear section markers and timing.
        """
        
        return enhanced_prompt

    def get_channel_prompt(self, channel_type, topic):
        """Get channel-specific prompt template with examples from successful scripts"""
        base_prompt = """
            Create an engaging script about {topic} for short-form video.

            GUIDELINES (not strict rules):
            - Aim for 50-100 words total
            - Use 7-10 clear points/lines
            - Add relevant emojis to enhance key points
            - Focus on engagement over exact timing
            - Keep each point clear and impactful

            Structure:
            1. Hook question/statement
            2. Quick engaging answer
            3-6. Main points with humor/value
            7. Surprising twist or revelation
            8. Clear call to action

            Example Format:
            Ever wondered why programmers debug with rubber ducks? 🦆
            These squeaky friends are our secret debugging tool! 💻
            Explaining code to ducks makes bugs magically appear! ✨
            Ducks never judge your messy code or variable names! 🤫
            They're the only ones who understand our loops! 🔄
            Plot twist: The ducks are actually coding experts! ��
            Follow for more dev secrets and duck debugging! 🚀
        """

        # Add successful examples if available
        successful_scripts = self._load_successful_scripts(channel_type)
        if successful_scripts:
            best_script = successful_scripts[-1]['script']
            base_prompt += f"\n\nHere's another successful example:\n{best_script}"

        return base_prompt.format(topic=topic)

    def _load_successful_scripts(self, channel_type: str) -> list:
        """Load previously successful scripts"""
        try:
            cache_file = f"cache/scripts/{channel_type}_successful.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(colored(f"Error loading successful scripts: {str(e)}", "yellow"))
            return [] 