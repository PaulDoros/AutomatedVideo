import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def list_models():
    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(model.id)
    except Exception as e:
        print(f"Error fetching models: {e}")

list_models()
