import asyncio
from deepseek_integration import DeepSeekAPI
from termcolor import colored

async def test_deepseek():
    """Test DeepSeek's script generation and TTS capabilities"""
    try:
        print(colored("\n=== Testing DeepSeek Integration ===", "blue"))
        
        # Initialize DeepSeek API
        deepseek = DeepSeekAPI()
        
        # Test script generation
        print(colored("\nTesting script generation...", "blue"))
        topic = "Why programmers love coffee"
        success, script = await deepseek.generate_script(topic, 3)
        
        if success and script:
            print(colored("✓ Script generated successfully:", "green"))
            print(colored(script, "cyan"))
            
            # Test voice generation
            print(colored("\nTesting voice generation...", "blue"))
            audio_path = await deepseek.generate_voice(script)
            
            if audio_path:
                print(colored(f"✓ Voice generated successfully at: {audio_path}", "green"))
            else:
                print(colored("✗ Voice generation failed", "red"))
        else:
            print(colored("✗ Script generation failed", "red"))
            
        # Test available voices
        print(colored("\nGetting available voices...", "blue"))
        voices = deepseek.get_available_voices()
        if voices:
            print(colored("✓ Available voices:", "green"))
            for voice in voices:
                print(colored(f"- {voice}", "cyan"))
        else:
            print(colored("✗ Failed to get available voices", "red"))
            
    except Exception as e:
        print(colored(f"[-] Error during DeepSeek test: {str(e)}", "red"))

if __name__ == "__main__":
    asyncio.run(test_deepseek()) 