import nltk

def download_nltk_resources():
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    print("NLTK resources downloaded successfully.")

if __name__ == "__main__":
    download_nltk_resources() 