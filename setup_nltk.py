import nltk

def download_nltk_data():
    print("Downloading required NLTK data for TextBlob...")
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('averaged_perceptron_tagger')
    # Use newer punkt_tab if required by some nltk versions
    try:
        nltk.download('punkt_tab')
    except:
        pass
    print("Download complete!")

if __name__ == "__main__":
    download_nltk_data()
