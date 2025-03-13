from TTS.api import TTS
import torch

def list_all_models():
    print("=== Available TTS Models ===\n")
    
    # Get all available models
    tts = TTS()
    all_models = tts.list_models()
    
    # Categorize models
    language_models = {}
    for model in all_models:
        if model.startswith("tts_models/"):
            parts = model.split("/")
            if len(parts) >= 3:
                lang = parts[1]
                if lang not in language_models:
                    language_models[lang] = []
                language_models[lang].append(model)
    
    # Print models by language
    for lang, models in sorted(language_models.items()):
        print(f"\n=== Language: {lang.upper()} ===")
        for model in sorted(models):
            print(f"\nTesting model: {model}")
            try:
                # Try to load the model to get more info
                tts = TTS(model_name=model, progress_bar=False)
                if hasattr(tts, "is_multi_speaker"):
                    print(f"Multi-speaker: {tts.is_multi_speaker}")
                if hasattr(tts, "speakers"):
                    if tts.speakers is not None:
                        print(f"Available speakers: {len(tts.speakers)}")
                        if len(tts.speakers) > 0:
                            print(f"Sample speakers: {tts.speakers[:5]}")
                print(f"Model type: {model.split('/')[-1]}")
            except Exception as e:
                print(f"Could not load model: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    print("Retrieving all TTS models and their capabilities...")
    list_all_models() 