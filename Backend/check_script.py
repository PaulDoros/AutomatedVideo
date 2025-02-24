import json
from termcolor import colored

def check_script_content():
    script_file = "cache/scripts/tech_humor_latest.json"
    
    try:
        with open(script_file, 'r') as f:
            data = json.load(f)
            script = data.get('script', '')
            
            print(colored("\n=== Script Content ===\n", "blue"))
            print(script)
            print("\n")
            
            # Check for sections
            sections = [line for line in script.split('\n') if line.strip().startswith('[') and line.strip().endswith(']')]
            print(colored("Found Sections:", "green"))
            for section in sections:
                print(colored(f"- {section}", "cyan"))
            
    except Exception as e:
        print(colored(f"Error reading script: {str(e)}", "red"))

if __name__ == "__main__":
    check_script_content() 