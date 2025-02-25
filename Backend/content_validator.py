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
            # Clean up the script
            lines = []
            for line in script.split('\n'):
                line = line.strip()
                if line and not line.startswith('['):
                    lines.append(line)

            # Calculate metrics
            total_words = sum(len(re.findall(r'\w+', line)) for line in lines)
            estimated_duration = total_words * 0.4  # 0.4 seconds per word

            # Strict length validation
            if total_words > 100:  # Hard limit on words
                return False, {"message": f"Script too long ({total_words} words). Maximum is 100 words."}
            
            if estimated_duration > 45:  # Hard limit on duration
                return False, {"message": f"Script too long ({estimated_duration:.1f}s). Maximum is 45 seconds."}
            
            if len(lines) > 8:  # Hard limit on lines
                return False, {"message": f"Too many lines ({len(lines)}). Maximum is 8 lines."}

            # Print analysis
            print(colored("\nScript Analysis:", "blue"))
            print(colored(f"Lines: {len(lines)}/{8} max", "cyan"))
            print(colored(f"Words: {total_words}/100 max", "cyan"))
            print(colored(f"Estimated Duration: {estimated_duration:.1f}/45s max", "cyan"))
            
            # Print the processed script
            print(colored("\nProcessed Script:", "blue"))
            for i, line in enumerate(lines, 1):
                print(colored(f"{i}. {line}", "cyan"))

            # Check for minimum content
            if total_words < 30:
                return False, {"message": "Script too short, need at least 30 words"}

            # Save successful scripts for learning
            if 25 <= estimated_duration <= 45:  # This is our ideal range
                self._save_successful_script(channel_type, script, {
                    "duration": estimated_duration,
                    "words": total_words,
                    "lines": len(lines)
                })

            return True, {
                "lines": len(lines),
                "words": total_words,
                "estimated_duration": estimated_duration,
                "is_ideal_length": 25 <= estimated_duration <= 45
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
        """Get channel-specific prompt template"""
        base_length_guide = """
        CRITICAL LENGTH REQUIREMENTS:
        - Target length: 20-30 seconds
        - Maximum word count: 50-60 words
        - Aim for 4-5 lines total
        - Each line: 10-15 words maximum
        
        Your script MUST fit these requirements for short-form video.
        """
        
        prompts = {
            'tech_humor': f"""
            {base_length_guide}
            
            Create a SINGLE, FOCUSED tech joke/story about {topic} for short-form video.
            Focus on ONE punchline and build up to it perfectly!
            Remember: ONE JOKE, make it count!

            Style Guide:
            - Start with a relatable tech situation
            - Build tension with one clear setup
            - Deliver one strong punchline
            - Keep energy high and delivery snappy
            
            Structure (4-5 lines ONLY):
            1. Hook: Relatable tech situation (10-12 words)
            2. Setup: Build the context (10-15 words)
            3. Tension: Add detail or twist (10-15 words)
            4. Punchline: The big payoff (8-12 words)
            5. Quick call to action (5-8 words)
            
            Example Perfect Length:
            "Ever wonder why programmers can't fix printers? 🖨️
            They keep saying 'It works on my machine!' 💻
            But when they try to print their code...
            All they get is 404: Paper not found! 😅
            Follow for more tech fails! 🚀"
            
            Focus on ONE strong joke - no multiple punchlines!
            Make it punchy, clear, and KEEP IT SHORT!
            """,

            'ai_money': """
            Create an INSIGHTFUL and ACTIONABLE script about {topic} for short-form video.
            Focus on real AI opportunities and practical implementation.

            Style Guide:
            - Professional but accessible tone
            - Data-driven insights
            - Clear step-by-step explanations
            - Realistic expectations
            - Specific numbers and examples
            - Engaging storytelling
            
            Content Elements:
            - Current AI trend or opportunity
            - Market gap or pain point
            - Practical implementation steps
            - Real earning potential
            - Risk considerations
            - Resource requirements
            
            Structure:
            1. Hook: Attention-grabbing AI stat/fact
            2. Problem: Market gap or opportunity
            3. Solution: AI-based approach
            4. Implementation: Step-by-step guide
            5. Results: Expected outcomes
            6. Validation: Proof or case study
            7. Action steps: How to start
            8. Engaging call to action
            
            Key Topics to Cover:
            - AI tools and platforms
            - Market research
            - Implementation strategy
            - Cost analysis
            - Revenue streams
            - Scaling potential
            - Common pitfalls
            - Success metrics
            """,

            'baby_tips': """
            Create a WARM and SUPPORTIVE script about {topic} for short-form video.
            Focus on practical, evidence-based parenting advice.

            Style Guide:
            - Nurturing and encouraging tone
            - Clear, actionable advice
            - Evidence-based information
            - Empathetic understanding
            - Realistic expectations
            - Safety-first approach
            
            Content Elements:
            - Common parenting challenge
            - Expert-backed solution
            - Step-by-step guidance
            - Safety considerations
            - Age-appropriate variations
            - Real parent experiences
            
            Structure:
            1. Hook: Relatable parenting moment
            2. Challenge: Common situation
            3. Solution: Expert advice
            4. Implementation: How-to steps
            5. Variations: Age adaptations
            6. Safety notes: Important considerations
            7. Success indicators: What to expect
            8. Supportive call to action
            
            Key Areas to Address:
            - Child development stages
            - Safety considerations
            - Expert recommendations
            - Common concerns
            - Practical tips
            - Parent self-care
            - Support resources
            """,

            'quick_meals': """
            Create an ENERGETIC and PRACTICAL script about {topic} for short-form video.
            Focus on delicious, time-saving meal solutions.

            Style Guide:
            - Enthusiastic and encouraging tone
            - Clear, concise instructions
            - Time-saving tips
            - Practical ingredient choices
            - Visual appeal focus
            - Engaging food descriptions
            
            Content Elements:
            - Quick recipe overview
            - Time-saving techniques
            - Ingredient substitutions
            - Kitchen hacks
            - Plating tips
            - Nutrition highlights
            
            Structure:
            1. Hook: Mouth-watering intro
            2. Recipe overview: Time/difficulty
            3. Ingredients: Simple list
            4. Quick steps: Clear process
            5. Pro tips: Time savers
            6. Variations: Easy swaps
            7. Final presentation
            8. Tasty call to action
            
            Key Features:
            - Prep time under 30 mins
            - Common ingredients
            - Tool substitutions
            - Storage tips
            - Batch cooking options
            - Nutritional benefits
            - Flavor enhancers
            """,

            'fitness_motivation': """
            Create an INSPIRING and ENERGIZING script about {topic} for short-form video.
            Focus on achievable fitness goals and motivation.

            Style Guide:
            - High-energy, motivational tone
            - Clear exercise instruction
            - Form-focused guidance
            - Inclusive language
            - Progress-oriented
            - Safety-conscious
            
            Content Elements:
            - Motivational hook
            - Exercise demonstration
            - Form guidance
            - Modification options
            - Progress tracking
            - Mental benefits
            
            Structure:
            1. Hook: Motivational opener
            2. Challenge: Fitness goal
            3. Technique: Proper form
            4. Progression: Level options
            5. Tips: Success secrets
            6. Benefits: Results preview
            7. Motivation: Inspiration
            8. Energetic call to action
            
            Key Components:
            - Form cues
            - Safety checks
            - Modification levels
            - Progress metrics
            - Recovery tips
            - Mental strategies
            - Success habits
            """
        }
        
        return prompts.get(channel_type, "").format(topic=topic)

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

    async def generate_with_gpt(self, prompt):
        """Generate content using GPT"""
        try:
            # Use the existing GPT integration
            from gpt import generate_gpt_response
            
            response = await generate_gpt_response(prompt)
            return response
            
        except Exception as e:
            print(colored(f"Error generating GPT response: {str(e)}", "red"))
            return None 