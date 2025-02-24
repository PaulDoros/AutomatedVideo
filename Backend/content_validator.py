from termcolor import colored
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

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
                'shorts': (30, 60),    # 30-60 seconds
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
            # Check basic requirements
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

            return True, {
                'analysis': analysis,
                'message': "Script meets all quality standards"
            }

        except Exception as e:
            return False, f"Validation error: {str(e)}"

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
        self.validator = ContentValidator()

    def generate_script(self, topic, channel_type, retry_count=3):
        """Generate and validate script, retry if needed"""
        for attempt in range(retry_count):
            try:
                prompt = self.get_channel_prompt(channel_type, topic)
                
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an expert YouTube script writer."},
                        {"role": "user", "content": prompt}
                    ]
                )

                script = response.choices[0].message.content
                
                # Validate script
                is_valid, validation_result = self.validator.validate_script(script, channel_type)
                
                if is_valid:
                    return True, script
                else:
                    print(colored(f"Attempt {attempt + 1}: Script validation failed", "yellow"))
                    print(colored(f"Reason: {validation_result['message']}", "yellow"))
                    continue
                    
            except Exception as e:
                print(colored(f"Error generating script: {str(e)}", "red"))
                
        return False, "Failed to generate valid script after multiple attempts"

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