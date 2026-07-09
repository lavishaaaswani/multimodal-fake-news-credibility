import sys
from input_module import extract_text_from_string
from language import detect_and_translate
from preprocessing import clean_text
from features import extract_all_features
from fuzzy_model import evaluate_credibility

def test_pipeline():
    sample_text = "OMGGG you won't believe this shocking secret! It is totally insane and mind-blowing!"
    print("1. Extracted Text:", sample_text)
    
    translated, lang = detect_and_translate(sample_text)
    print("2. Translated:", translated, "Lang:", lang)
    
    cleaned = clean_text(translated)
    print("3. Cleaned:", cleaned)
    
    features = extract_all_features(cleaned, domain=None)
    print("4. Features extracted:", features)
    
    result = evaluate_credibility(features)
    print("5. Credibility Result:", result)
    
    if result['score'] < 50:
        print("PIPELINE TEST PASSED! Clickbait detected properly.")
        sys.exit(0)
    else:
        print("PIPELINE TEST FAILED! Clickbait not penalized correctly.")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
