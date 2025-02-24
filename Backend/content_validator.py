from termcolor import colored
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

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
            # Check if we already have a valid script saved
            cache_file = f"cache/scripts/{channel_type}_latest.json"
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    if cached.get('is_valid'):
                        print(colored("Using cached valid script", "green"))
                        return True, cached['script']

            # If no valid cache, proceed with validation
            word_count = len(script.split())
            if word_count < self.standards['min_words']:
                return False, "Script too short"
            if word_count > self.standards['max_words']:
                return False, "Script too long"

            # Validate script structure using GPT-4
            prompt = f"""
            Analyze this script for a {channel_type} video and check for:
            1. Clear hook in first 5 seconds
            2. Logical flow and structure
            3. Complete thoughts and explanations
            4. Strong conclusion
            5. Clear call-to-action
            6. Estimated video duration
            7. Overall quality and engagement potential

            Script:
            {script}

            Provide analysis in JSON format with these keys:
            - has_hook (boolean)
            - has_structure (boolean)
            - is_complete (boolean)
            - has_conclusion (boolean)
            - has_cta (boolean)
            - estimated_duration (number in seconds)
            - quality_score (1-10)
            - issues (array of strings)
            - suggestions (array of strings)
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a video script analyzer."},
                    {"role": "user", "content": prompt}
                ]
            )

            analysis = json.loads(response.choices[0].message.content)
            
            # Check if meets all requirements
            meets_requirements = (
                analysis['has_hook'] and
                analysis['has_structure'] and
                analysis['is_complete'] and
                analysis['has_conclusion'] and
                analysis['has_cta'] and
                analysis['quality_score'] >= 8
            )

            if not meets_requirements:
                return False, {
                    'analysis': analysis,
                    'message': "Script doesn't meet quality standards"
                }

            # Check duration
            duration_range = self.standards['ideal_duration']['shorts']
            if not (duration_range[0] <= analysis['estimated_duration'] <= duration_range[1]):
                return False, {
                    'analysis': analysis,
                    'message': f"Duration ({analysis['estimated_duration']}s) outside ideal range {duration_range}"
                }

            # If valid, cache the script
            if meets_requirements:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'is_valid': True,
                        'script': script,
                        'timestamp': time.time()
                    }, f)

            return meets_requirements, analysis

        except Exception as e:
            print(colored(f"Error validating script: {str(e)}", "red"))
            return False, str(e)

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
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.validator = ContentValidator()
        
        # Define model options with costs per 1K tokens
        self.models = {
            'o1_premium': {
                'name': 'gpt-4-0125-preview',  # O1 model
                'cost': 0.08,
                'client': 'openai',
                'max_tokens': 4096
            },
            'gpt4_turbo': {
                'name': 'gpt-4-1106-preview',
                'cost': 0.03,
                'client': 'openai',
                'max_tokens': 4096
            }
        }
        
        # Set O1 as primary model and GPT-4 Turbo as fallback
        self.primary_model = self.models['o1_premium']
        self.fallback_model = self.models['gpt4_turbo']

    async def generate_script(self, topic, channel_type, use_premium=True, retry_count=3):
        """Generate script using O1 with GPT-4 Turbo fallback"""
        prompt = self.get_enhanced_prompt(topic, channel_type)
        
        for attempt in range(retry_count):
            try:
                model = self.primary_model if use_premium else self.fallback_model
                print(colored(f"Using {model['name']} for content generation", "blue"))
                
                response = self.openai_client.chat.completions.create(
                    model=model['name'],
                    messages=[
                        {"role": "system", "content": self.get_system_prompt(channel_type)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=model['max_tokens'],
                    top_p=0.9
                )
                script = response.choices[0].message.content

                # Validate script
                is_valid, validation_result = self.validator.validate_script(script, channel_type)
                
                if is_valid:
                    # Track usage for cost monitoring
                    self.track_usage(script, use_premium)
                    return True, script
                else:
                    print(colored(f"Attempt {attempt + 1}: {validation_result['message']}", "yellow"))
                    
                    # If primary model fails, try improvement
                    if attempt == retry_count - 2:
                        print(colored("Trying to improve script...", "yellow"))
                        improved_script = await self.improve_script(script, channel_type)
                        if improved_script:
                            return True, improved_script

            except Exception as e:
                print(colored(f"Error generating script: {str(e)}", "red"))
                
        return False, "Failed to generate valid script after multiple attempts"

    async def improve_script(self, original_script, channel_type):
        """Use O1's reasoning to improve the script"""
        try:
            improvement_prompt = f"""
            Analyze and improve this script for maximum engagement and quality:

            Original Script:
            {original_script}

            Please:
            1. Enhance the hook to be more captivating
            2. Improve the flow and pacing
            3. Make the language more natural and conversational
            4. Strengthen the emotional connection
            5. Make the call-to-action more compelling

            Maintain the same basic structure but make it more engaging and authentic.
            """

            response = self.openai_client.chat.completions.create(
                model=self.primary_model['name'],
                messages=[
                    {"role": "system", "content": "You are an expert script editor focused on engagement and authenticity."},
                    {"role": "user", "content": improvement_prompt}
                ],
                temperature=0.7,
                max_tokens=self.primary_model['max_tokens']
            )

            improved_script = response.choices[0].message.content
            is_valid, validation_result = self.validator.validate_script(improved_script, channel_type)
            
            if is_valid:
                return improved_script
            return None

        except Exception as e:
            print(colored(f"Error improving script: {str(e)}", "red"))
            return None

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
        prompts = {
            'tech_humor': """
                Create an engaging and humorous script about {topic}.
                Requirements:
                - Hook: Funny tech situation or relatable coding moment
                - At least 3 jokes/punchlines
                - Mix technical accuracy with humor
                - Use analogies for complex concepts
                - Keep technical terms under 5
                - 45-60 seconds length
                - End: Funny conclusion + subscribe CTA
                
                Format:
                [Hook - 5s]
                [Setup - 15s]
                [Technical Explanation with Jokes - 25s]
                [Punchline - 10s]
                [Call to action - 5s]
            """,
            'ai_money': """
                Create an informative script about {topic}.
                Requirements:
                - Hook: Specific earning example
                - 3 clear, actionable steps
                - Include real numbers and timeframes
                - Address common obstacles
                - Show proof/results
                - 45-60 seconds length
                - End: Value proposition + CTA
                
                Format:
                [Hook with Proof - 10s]
                [Problem/Market Gap - 10s]
                [3 Steps with Details - 30s]
                [Results/Timeline - 5s]
                [Call to action - 5s]
            """,
            'baby_tips': """
                Create a helpful parenting script about {topic}.
                Requirements:
                - Hook: Common parenting challenge
                - Safety-first approach
                - Expert-backed tips
                - Include medical disclaimer
                - Address common concerns
                - 45-60 seconds length
                - End: Supportive message + CTA
                
                Format:
                [Hook/Problem - 10s]
                [Safety Note - 5s]
                [Solution Steps - 35s]
                [Tips & Warnings - 5s]
                [Call to action - 5s]
            """,
            'quick_meals': """
                Create a recipe script about {topic}.
                Requirements:
                - Hook: Final dish appeal
                - Max 10 ingredients
                - 15-minute prep time
                - Include nutrition basics
                - Time-saving tips
                - 45-60 seconds length
                - End: Serving suggestion + CTA
                
                Format:
                [Hook/Final Result - 5s]
                [Ingredients List - 10s]
                [Step-by-Step - 35s]
                [Tips & Nutrition - 5s]
                [Call to action - 5s]
            """,
            'fitness_motivation': """
                Create a workout script about {topic}.
                Requirements:
                - Hook: Benefit/Transformation
                - Safety instructions
                - Form guidance
                - Modification options
                - Motivation elements
                - 45-60 seconds length
                - End: Motivation + CTA
                
                Format:
                [Hook/Benefits - 10s]
                [Safety & Form - 10s]
                [Workout Steps - 30s]
                [Motivation - 5s]
                [Call to action - 5s]
            """
        }
        return prompts.get(channel_type, "").format(topic=topic) 