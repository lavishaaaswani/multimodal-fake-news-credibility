import os
import pickle
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'clickbait_model.pkl')

def load_models():
    """Loads cache pickle models trained on the ISOT DataFrame."""
    if not os.path.exists(MODEL_PATH):
        logging.error("Model not found! Please run 'python train_real_model.py' first.")
        return None
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# Global models instance initialized upon app start
_MODELS = load_models()

def predict_clickbait(text):
    """Predict probabilistic likelihood of text containing sensational clickbait phrasing."""
    if _MODELS and 'clickbait_model' in _MODELS:
        # Probability of class 1 (Clickbait)
        probs = _MODELS['clickbait_model'].predict_proba([text])[0]
        return probs[1]
    return 0.5

def predict_formality(text):
    """Predict probabilistic likelihood of text being legitimate, reliable formal content."""
    if _MODELS and 'formal_model' in _MODELS:
        # Probability of class 1 (Formal/News/Objective)
        probs = _MODELS['formal_model'].predict_proba([text])[0]
        return probs[1]
    return 0.5
