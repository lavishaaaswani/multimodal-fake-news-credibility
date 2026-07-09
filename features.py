import os
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from ml_models import predict_clickbait, predict_formality

# Load assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Keep basic lexicons for Streamlit UI Highlights ONLY (XAI Explainability)
try:
    with open(os.path.join(BASE_DIR, 'data', 'clickbait_words.txt'), 'r') as f:
        CLICKBAIT_WORDS = [line.strip().lower() for line in f if line.strip()]
except:
    CLICKBAIT_WORDS = []

# Load Domain Scores Database
try:
    with open(os.path.join(BASE_DIR, 'data', 'domain_scores.json'), 'r') as f:
        DOMAIN_SCORES = json.load(f)
except:
    DOMAIN_SCORES = {}

analyzer = SentimentIntensityAnalyzer()

def emotional_score(text):
    """Returns emotional intensity. TextBlob polarity is averaged over words, making it ideal for long texts."""
    blob = TextBlob(text)
    return min(abs(blob.sentiment.polarity) * 2.0, 1.0)

def objectivity_score(text):
    """Uses VADER and TextBlob to estimate objectivity robustly."""
    blob = TextBlob(text)
    blob_obj = min((1.0 - blob.sentiment.subjectivity) * 1.2, 1.0)
    scores = analyzer.polarity_scores(text)
    vader_obj = scores['neu']
    return (blob_obj + vader_obj) / 2.0

def source_score(domain, text):
    """
    Hybrid scoring: Uses known domain scores if available, but blends them with the ML Formality prediction.
    This ensures that even articles from the exact same URL get slightly different Source Scores
    based on how well-written that specific text is.
    If the domain is unknown, it falls back entirely to the ML assessment.
    """
    ml_formality = predict_formality(text)
    
    if domain:
        domain_lower = domain.lower()
        for known_domain, score in DOMAIN_SCORES.items():
            if known_domain in domain_lower:
                # Blend 80% domain reputation with 20% specific article text formality
                return (float(score) * 0.8) + (ml_formality * 0.2)
                
    # Fallback to pure ML Formality assessment if domain is unknown
    return ml_formality

def clickbait_score(text):
    """Returns a score based on scikit-learn ML + Conspiracy Lexicon Heuristics."""
    ml_score = predict_clickbait(text)
    
    # Catch well-written conspiracies that mimic formal journalism
    text_lower = text.lower()
    hits = sum(1 for word in CLICKBAIT_WORDS if word in text_lower)
    
    # Boost Fake News probability by 8% per heuristic hit, capped at +30% total boost
    # This prevents the heuristic from accidentally overriding standard political journalism
    return min(ml_score + min(hits * 0.08, 0.30), 1.0)

def length_score(text):
    """Returns score based on word count. Longer text up to 150 words is considered better."""
    words = text.split()
    return min(len(words) / 150.0, 1.0)

def repetition_score(text):
    """Returns repetition score. Adjusted to rigorously ignore stop words and punctuation masks."""
    # Extract pure words over 2 chars, stripping out all punctuation formatting
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    # Strip out extremely common English structural words to target *true* content repetition
    stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'have', 'has', 'had'}
    content_words = [w for w in words if w not in stop_words]
    
    if len(content_words) < 10:
        return 0.0
        
    unique_words = set(content_words)
    ratio = len(unique_words) / len(content_words)
    
    # Normal journalism has high diversity of content words (ratio > 0.6)
    # Spammers repeat the exact same substantive keywords over and over (ratio < 0.3)
    rep = 1.0 - ratio
    
    # Penalty starts aggressively if content diversity drops below 50%
    final_score = max((rep - 0.40) * 3.0, 0.0)
    return min(final_score, 1.0)

def extract_all_features(text, domain=None):
    """Runs all feature extractors and returns a dictionary."""
    features = {
        "emotion": emotional_score(text),
        "objectivity": objectivity_score(text),
        "source": source_score(domain, text),
        "clickbait": clickbait_score(text),
        "length": length_score(text),
        "repetition": repetition_score(text)
    }
    return features
