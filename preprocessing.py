import re

def clean_text(text):
    """
    Preprocesses the input text:
    - Lowercase
    - Remove URLs
    - Remove special characters but keeps essential punctuation
    - Trim spaces
    """
    if not text:
        return ""
        
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove excessive newlines
    text = re.sub(r'\n+', ' ', text)
    # Trim spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
