from termcolor import colored
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from typing import Tuple
from datetime import datetime

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

            # Channel-specific length limits
            channel_limits = {
                'tech_humor': {'max_words': 100, 'min_words': 10, 'max_duration': 60, 'max_lines': 10},
                'ai_money': {'max_words': 200, 'min_words': 30, 'max_duration': 90, 'max_lines': 15},
                'baby_tips': {'max_words': 200, 'min_words': 30, 'max_duration': 90, 'max_lines': 15},
                'quick_meals': {'max_words': 200, 'min_words': 30, 'max_duration': 90, 'max_lines': 15},
                'fitness_motivation': {'max_words': 200, 'min_words': 30, 'max_duration': 90, 'max_lines': 15}
            }
            
            # Get limits for this channel type, or use default limits
            limits = channel_limits.get(channel_type, {'max_words': 150, 'min_words': 20, 'max_duration': 60, 'max_lines': 12})
            
            # Strict length validation - more flexible for tech_humor
            if total_words > limits['max_words']:
                return False, {"message": f"Script too long ({total_words} words). Maximum is {limits['max_words']} words."}
            
            if len(lines) > limits['max_lines']:
                return False, {"message": f"Too many lines ({len(lines)}). Maximum is {limits['max_lines']} lines."}

            # Print analysis
            print(colored("\nScript Analysis:", "blue"))
            print(colored(f"Lines: {len(lines)}/{limits['max_lines']} max", "cyan"))
            print(colored(f"Words: {total_words}/{limits['max_words']} max", "cyan"))
            print(colored(f"Estimated Duration: {estimated_duration:.1f}s", "cyan"))
            
            # Print the processed script
            print(colored("\nProcessed Script:", "blue"))
            for i, line in enumerate(lines, 1):
                print(colored(f"{i}. {line}", "cyan"))

            # Check for minimum content - more flexible for tech_humor
            if total_words < limits['min_words']:
                return False, {"message": f"Script too short, need at least {limits['min_words']} words"}

            # Save successful scripts for learning
            ideal_min = limits['max_duration'] * 0.4  # 40% of max duration
            ideal_max = limits['max_duration']
            
            # For tech_humor, we're more flexible with the ideal range
            if channel_type == 'tech_humor':
                ideal_min = 10  # Even very short jokes are fine
            
            if ideal_min <= estimated_duration <= ideal_max:
                self._save_successful_script(channel_type, script, {
                    "duration": estimated_duration,
                    "words": total_words,
                    "lines": len(lines)
                })

            return True, {
                "lines": len(lines),
                "words": total_words,
                "estimated_duration": estimated_duration,
                "is_ideal_length": ideal_min <= estimated_duration <= ideal_max
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
            
            # Add request for thumbnail title in a separate field
            content_prompt += """
            
            AFTER you've completed the script, please also include a separate catchy thumbnail title.
            Place it at the very end after a blank line and prefix it with "THUMBNAIL_TITLE:" 
            The thumbnail title should:
            - Be attention-grabbing (5-7 words max)
            - Include 1-2 relevant emoji
            - NOT be part of the script itself
            """
            
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
            
            full_response = response.choices[0].message.content.strip()
            
            # Extract the script and thumbnail title separately
            script = full_response
            thumbnail_title = ""
            
            # Check for thumbnail title marker - should be at the end
            if "THUMBNAIL_TITLE:" in full_response:
                parts = full_response.split("THUMBNAIL_TITLE:")
                script = parts[0].strip()
                thumbnail_title = parts[1].strip()
                print(colored(f"\nExtracted Thumbnail Title: {thumbnail_title}", "green"))
            
            # Validate script
            validator = ContentValidator()
            is_valid, analysis = validator.validate_script(script, channel_type)
            
            if is_valid:
                print(colored("\nGenerated Script:", "green"))
                print(colored(script, "cyan"))
                print(colored(f"\nEstimated Duration: {analysis.get('estimated_duration', 0):.1f}s", "blue"))
                
                # Save script and thumbnail title separately
                self._save_script(script, thumbnail_title, channel_type, is_valid, analysis)
                
                # Return ONLY the script for processing
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

    def _save_script(self, script: str, thumbnail_title: str, channel_type: str, is_valid: bool, analysis: dict):
        """Save script to cache with thumbnail title"""
        try:
            # Create directory if it doesn't exist
            os.makedirs("cache/scripts", exist_ok=True)
            
            # Define section headers to remove
            section_headers = [
                "**Hook:**", "**Problem/Setup:**", "**Solution/Development:**", 
                "**Result/Punchline:**", "**Call to action:**", 
                "Hook:", "Problem/Setup:", "Solution/Development:", 
                "Result/Punchline:", "Call to action:",
                "**Script:**", "Script:"
            ]
            
            # Define patterns to filter out
            special_patterns = ["---", "***", "**", "##"]
            
            # Process line by line
            cleaned_lines = []
            for line in script.split('\n'):
                line_to_add = line
                skip_line = False
                
                # Skip lines that only contain special characters
                if line.strip() in special_patterns or line.strip("-*#") == "":
                    skip_line = True
                    continue
                
                # Remove numeric prefixes (like "1.")
                if re.match(r'^\d+\.', line.strip()):
                    line_to_add = re.sub(r'^\d+\.\s*', '', line.strip())
                
                for header in section_headers:
                    if line.strip() == header or line.strip().startswith(header):
                        # If the line is just a header, skip it entirely
                        if line.strip() == header or line[line.find(header) + len(header):].strip() == "":
                            skip_line = True
                            break
                        # Otherwise, remove the header and keep the content
                        line_to_add = line[line.find(header) + len(header):].strip()
                        break
                
                if not skip_line and line_to_add.strip():
                    cleaned_lines.append(line_to_add)
            
            # Reassemble the script
            cleaned_script = '\n'.join(cleaned_lines)
            
            # Save a plain text version for TTS
            with open(f"cache/scripts/{channel_type}_latest.txt", "w", encoding="utf-8") as f:
                f.write(cleaned_script)
            
            # Generate a preview of the script (first 100 characters)
            preview = cleaned_script[:100] + "..." if len(cleaned_script) > 100 else cleaned_script
            
            # Save the script data to a JSON file
            script_data = {
                "script": script,  # Original script for reference
                "cleaned_script": cleaned_script,  # Cleaned script for TTS and subtitles
                "thumbnail_title": thumbnail_title,
                "preview": preview,
                "is_valid": is_valid,
                "metrics": analysis.get("metrics", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(f"cache/scripts/{channel_type}_latest.json", "w", encoding="utf-8") as f:
                json.dump(script_data, f, indent=2)
            
            print(colored(f"✓ Script saved to cache/scripts/{channel_type}_latest.json", "green"))
            return True

        except Exception as e:
            print(colored(f"Error saving script: {str(e)}", "red"))
            return False

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

    def get_channel_prompt(self, channel_type, topic=None):
        """Get the appropriate prompt for the channel type"""
        base_prompt = """
You are a professional script writer for YouTube Shorts. 
Your task is to create engaging, concise scripts that capture attention quickly.

IMPORTANT GUIDELINES:
"""

        # Channel-specific length guidelines
        length_guidelines = {
            'tech_humor': """
- Keep it SHORT and ENGAGING: 15-60 seconds when read aloud
- Word count: 10-100 words
- Structure: 3-10 lines maximum
- Focus on ONE clear punchline or joke
""",
            'quick_meals': """
- Keep it CONCISE but INFORMATIVE: 30-90 seconds when read aloud
- Word count: 80-200 words
- Structure: 5-15 lines maximum
- Focus on ONE clear recipe or cooking tip
""",
            'ai_money': """
- Keep it ACTIONABLE and FOCUSED: 30-90 seconds when read aloud
- Word count: 80-200 words
- Structure: 5-15 lines maximum
- Focus on ONE specific AI tool or money-making strategy
""",
            'baby_tips': """
- Keep it HELPFUL and PRACTICAL: 30-90 seconds when read aloud
- Word count: 80-200 words
- Structure: 5-15 lines maximum
- Focus on ONE specific parenting tip or solution
""",
            'fitness_motivation': """
- Keep it ENERGETIC and MOTIVATIONAL: 30-90 seconds when read aloud
- Word count: 80-200 words
- Structure: 5-15 lines maximum
- Focus on ONE specific workout tip or fitness motivation
"""
        }
        
        # Get length guidelines for this channel type, or use default
        guidelines = length_guidelines.get(channel_type, length_guidelines['tech_humor'])
        
        base_prompt += guidelines

        # Add channel-specific content guidelines
        if channel_type == "tech_humor":
            prompt = base_prompt + f"""
- TONE: Humorous, witty, slightly sarcastic
- CONTENT: Focus on tech jokes, programmer humor, or funny tech situations related to {topic if topic else "technology"}
- STYLE: Use puns, unexpected twists, or relatable tech frustrations
- FORMAT: Can be a short joke, a funny observation, or a humorous tech tip
- IMPORTANT: Make it genuinely funny! Avoid clichés and predictable punchlines.
"""
        elif channel_type == "ai_money":
            prompt = base_prompt + f"""
- TONE: Informative, exciting, motivational
- CONTENT: Focus on AI tools for making money, side hustles, or passive income related to {topic if topic else "AI opportunities"}
- STYLE: Use concrete examples, specific tools, and actionable advice
- FORMAT: Quick tip, tool introduction, or money-making strategy
- IMPORTANT: Provide SPECIFIC, ACTIONABLE advice that viewers can implement immediately.
"""
        elif channel_type == "baby_tips":
            prompt = base_prompt + f"""
- TONE: Helpful, warm, supportive
- CONTENT: Focus on practical baby care tips, parenting hacks, or child development insights related to {topic if topic else "baby care"}
- STYLE: Use evidence-based information, relatable situations, and gentle advice
- FORMAT: Quick tip, solution to common problem, or developmental milestone explanation
- IMPORTANT: Provide PRACTICAL advice that tired parents can easily implement.
"""
        elif channel_type == "quick_meals":
            prompt = base_prompt + f"""
- TONE: Enthusiastic, practical, encouraging
- CONTENT: Focus on easy recipes, cooking hacks, or meal prep tips related to {topic if topic else "quick cooking"}
- STYLE: Use simple ingredients, minimal steps, and time-saving techniques
- FORMAT: Recipe walkthrough, cooking tip, or food hack
- IMPORTANT: Keep recipes SIMPLE with FEW ingredients and CLEAR instructions.
"""
        elif channel_type == "fitness_motivation":
            prompt = base_prompt + f"""
- TONE: Energetic, motivational, supportive
- CONTENT: Focus on workout tips, fitness motivation, or healthy habits related to {topic if topic else "fitness"}
- STYLE: Use encouraging language, achievable goals, and scientific facts
- FORMAT: Quick exercise demo, motivational message, or fitness hack
- IMPORTANT: Make workouts ACCESSIBLE to beginners while still CHALLENGING for regulars.
"""
        else:
            prompt = base_prompt + f"""
- TONE: Engaging, informative, conversational
- CONTENT: Focus on valuable information, interesting facts, or useful tips related to {topic if topic else "this topic"}
- STYLE: Use clear language, specific examples, and actionable advice
- FORMAT: Quick tip, interesting fact, or helpful explanation
"""

        # Add specific topic instruction
        if topic:
            prompt += f"""
SPECIFIC TOPIC: Create a script about {topic}. Focus on ONE clear message or takeaway.
"""
        else:
            prompt += """
Choose a trending, engaging topic for this channel. Focus on ONE clear message or takeaway.
"""

        # Add example structure
        prompt += """
STRUCTURE EXAMPLE (DO NOT include these labels in your actual script):
- Start with an attention-grabbing opening (10-15 words)
- Identify the situation or problem (10-20 words)
- Present key information or solution (20-50 words)
- Deliver the main point or punchline (10-20 words)
- End with a brief call to action (5-10 words)

IMPORTANT: DO NOT include section labels like "Hook:", "Problem/Setup:", etc. in your script. 
Write the script as a flowing, natural piece of content without any section headers or markers.
"""

        return prompt

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