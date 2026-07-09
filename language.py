from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import logging

# Ensure consistent language detection
DetectorFactory.seed = 0

def detect_and_translate(text):
    """
    Detects the language of the provided text.
    If the text is in Hindi (or another non-English language), translates it to English.
    Returns: (processed_text, original_language)
    """
    if not text or not text.strip():
        return "", "en"
        
    # Heuristic: If Devanagari characters are highly present, force Hindi translation
    # This bypasses the statistical limits of langdetect on short strings
    import re
    if re.search(r'[\u0900-\u097F]', text):
        lang = 'hi'
    else:
        try:
            lang = detect(text)
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            lang = "en" # Fallback to English

    # Translate if not English
    if lang != 'en':
        try:
            translator = GoogleTranslator(source='auto', target='en')
            # deep_translator has a 5000 character limit, but frequently times out on anything >2000 chars. Use 1500.
            chunk_size = 1500
            if len(text) > chunk_size:
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                translated_chunks = [translator.translate(chunk) for chunk in chunks]
                translated_text = " ".join(translated_chunks)
            else:
                translated_text = translator.translate(text)
            
            return translated_text, lang
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return text, lang # Return original if translation fails
            
    return text, 'en'
