import re
import os
import g4f
import json
import openai
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from typing import Tuple, List

# Load environment variables
load_dotenv("../.env")

# Set environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def generate_gpt_response(prompt: str) -> str:
    """Generate response using GPT model"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using latest GPT-4 model
            messages=[
                {"role": "system", "content": "You are a creative content writer specializing in short-form video scripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150  # Limit response length for short-form content
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in GPT response generation: {str(e)}")
        return None

def generate_response(prompt: str, ai_model: str) -> str:
    """
    Generate a script for a video, depending on the subject of the video.

    Args:
        video_subject (str): The subject of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        str: The response from the AI model.
    """

    # Normalize the model name to lowercase
    ai_model = ai_model.lower()

    if ai_model == 'g4f':
        # Newest G4F Architecture
        try:
            client = Client()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                # Try different providers in order until one works
                provider=g4f.Provider.DeepAi,  # Changed from You to DeepAi
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content
            return response
        except Exception as e:
            print(colored(f"[-] First G4F provider failed, trying backup...", "yellow"))
            try:
                # Backup provider
                response = g4f.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    provider=g4f.Provider.You,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response
            except Exception as e2:
                print(colored(f"[-] All G4F providers failed, falling back to OpenAI", "red"))
                return generate_response(prompt, "gpt-3.5-turbo-1106")

    elif ai_model in [
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-1106", 
        "gpt-4", 
        "gpt-4-1106-preview",
        "gpt-4-turbo-preview",
        "gpt3.5-turbo",  # Support legacy names
        "gpt4"
    ]:
        model_name = ai_model
        if ai_model == "gpt3.5-turbo":
            model_name = "gpt-3.5-turbo-1106"
        elif ai_model == "gpt4":
            model_name = "gpt-4-1106-preview"

        try:
            response = openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content
            return response
        except Exception as e:
            print(colored(f"[-] OpenAI API error: {str(e)}", "red"))
            # Try one last time with G4F as backup
            return generate_response(prompt, "g4f")

    elif ai_model == 'gemmini':
        model = genai.GenerativeModel('gemini-pro')
        response_model = model.generate_content(prompt)
        return response_model.text

    else:
        print(colored(f"[-] Unknown model: {ai_model}, falling back to gpt-3.5-turbo-1106", "yellow"))
        return generate_response(prompt, "gpt-3.5-turbo-1106")

def generate_script(video_subject: str, paragraph_number: int, ai_model: str, voice: str, customPrompt: str) -> str:
    """
    Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.

    Args:
        video_subject (str): The subject of the video.
        paragraph_number (int): The number of paragraphs to generate.
        ai_model (str): The AI model to use for generation.
        voice (str): The voice to use for TTS.
        customPrompt (str): Custom prompt to use instead of default.

    Returns:
        str: The script for the video.
    """
    # Check if using DeepSeek
    if ai_model.lower() == "deepseek":
        try:
            from deepseek_integration import DeepSeekAPI
            import asyncio
            
            deepseek = DeepSeekAPI()
            success, script = asyncio.run(deepseek.generate_script(video_subject, paragraph_number, customPrompt))
            
            if success and script:
                print(colored(script, "cyan"))
                return script
            else:
                print(colored("[-] DeepSeek script generation failed, falling back to OpenAI.", "yellow"))
                # Fall through to OpenAI
        except Exception as e:
            print(colored(f"[-] Error with DeepSeek: {str(e)}, falling back to OpenAI.", "yellow"))
            # Fall through to OpenAI

    # Build prompt for OpenAI
    if customPrompt:
        prompt = customPrompt
    else:
        prompt = """
            Generate a script for a video, depending on the subject of the video.

            The script is to be returned as a string with the specified number of paragraphs.

            Here is an example of a string:
            "This is an example string."

            Do not under any circumstance reference this prompt in your response.

            Get straight to the point, don't start with unnecessary things like, "welcome to this video".

            Obviously, the script should be related to the subject of the video.

            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
            YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
            ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.
        """

    prompt += f"""
    
    Subject: {video_subject}
    Number of paragraphs: {paragraph_number}
    Language: {voice}

    """

    # Generate script
    response = generate_response(prompt, ai_model)

    print(colored(response, "cyan"))

    # Return the generated script
    if response:
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        final_script = "\n\n".join(selected_paragraphs)

        # Print to console the number of paragraphs used
        print(colored(f"Number of paragraphs used: {len(selected_paragraphs)}", "green"))

        return final_script
    else:
        print(colored("[-] GPT returned an empty response.", "red"))
        return None


def get_search_terms(video_subject: str, amount: int, script: str, ai_model: str) -> List[str]:
    """
    Generate a JSON-Array of search terms for stock videos,
    depending on the subject of a video.

    Args:
        video_subject (str): The subject of the video.
        amount (int): The amount of search terms to generate.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        List[str]: The search terms for the video subject.
    """

    # Build prompt
    prompt = f"""
    Generate {amount} search terms for stock videos,
    depending on the subject of a video.
    Subject: {video_subject}

    The search terms are to be returned as
    a JSON-Array of strings.

    Each search term should consist of 1-3 words,
    always add the main subject of the video.
    
    YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    YOU MUST NOT RETURN ANYTHING ELSE. 
    YOU MUST NOT RETURN THE SCRIPT.
    
    The search terms must be related to the subject of the video.
    Here is an example of a JSON-Array of strings:
    ["search term 1", "search term 2", "search term 3"]

    For context, here is the full text:
    {script}
    """

    # Generate search terms
    response = generate_response(prompt, ai_model)
    print(response)

    # Parse response into a list of search terms
    search_terms = []
    
    try:
        # First try to parse as pure JSON
        search_terms = json.loads(response.strip())
        if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
            raise ValueError("Response is not a list of strings.")

    except (json.JSONDecodeError, ValueError) as e:
        print(colored("[*] GPT returned an unformatted response. Attempting to clean...", "yellow"))
        try:
            # Find anything that looks like a JSON array
            match = re.search(r'\[[\s\S]*\]', response)
            if match:
                cleaned_json = match.group()
                search_terms = json.loads(cleaned_json)
            else:
                # If no JSON array found, split by commas and clean up
                terms = response.replace('[', '').replace(']', '').split(',')
                search_terms = [term.strip().strip('"\'') for term in terms if term.strip()]
                
            # Validate the terms
            search_terms = [term for term in search_terms if term and isinstance(term, str)]
            
            if not search_terms:
                # Fallback to basic terms if everything else fails
                search_terms = [
                    f"{video_subject} footage",
                    f"{video_subject} video",
                    f"{video_subject} clip",
                    "business footage",
                    "office work"
                ]
                print(colored("[-] Using fallback search terms", "yellow"))
        except Exception as e2:
            print(colored(f"[-] Error parsing search terms: {str(e2)}", "red"))
            # Use fallback terms
            search_terms = [
                f"{video_subject} footage",
                f"{video_subject} video",
                f"{video_subject} clip",
                "business footage",
                "office work"
            ]

    # Ensure we have enough terms
    while len(search_terms) < amount:
        search_terms.append(f"{video_subject} clip {len(search_terms) + 1}")

    # Limit to requested amount
    search_terms = search_terms[:amount]

    # Let user know
    print(colored(f"\nGenerated {len(search_terms)} search terms: {', '.join(search_terms)}", "cyan"))

    # Return search terms
    return search_terms


def generate_metadata(video_subject: str, script: str, ai_model: str) -> Tuple[str, str, List[str]]:  
    """  
    Generate metadata for a YouTube video, including the title, description, and keywords.  
  
    Args:  
        video_subject (str): The subject of the video.  
        script (str): The script of the video.  
        ai_model (str): The AI model to use for generation.  
  
    Returns:  
        Tuple[str, str, List[str]]: The title, description, and keywords for the video.  
    """  
  
    # Build prompt for title  
    title_prompt = f"""  
    Generate a catchy and SEO-friendly title for a YouTube shorts video about {video_subject}.  
    """  
  
    # Generate title  
    title = generate_response(title_prompt, ai_model).strip()  
    
    # Build prompt for description  
    description_prompt = f"""  
    Write a brief and engaging description for a YouTube shorts video about {video_subject}.  
    The video is based on the following script:  
    {script}  
    """  
  
    # Generate description  
    description = generate_response(description_prompt, ai_model).strip()  
  
    # Generate keywords  
    keywords = get_search_terms(video_subject, 6, script, ai_model)  

    return title, description, keywords  
