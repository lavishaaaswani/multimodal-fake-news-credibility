import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def build_fuzzy_system():
    # Antecedents (Inputs)
    # Ranging from 0 to 1
    emotion = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'emotion')
    objectivity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'objectivity')
    source = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'source')
    clickbait = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'clickbait')
    length = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'length')
    repetition = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'repetition')

    # Consequent (Output)
    # Ranging from 0 to 100
    credibility = ctrl.Consequent(np.arange(0, 101, 1), 'credibility')
    
    # Custom membership functions using triangular reqs
    for var in [emotion, objectivity, source, clickbait, length, repetition]:
        var['low'] = fuzz.trimf(var.universe, [0, 0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0, 0.5, 1])
        var['high'] = fuzz.trimf(var.universe, [0.5, 1, 1])

    # Credibility membership functions (Equally wide bases prevent Centroid Gravity skew!)
    credibility['low'] = fuzz.trimf(credibility.universe, [0, 0, 50])
    credibility['medium'] = fuzz.trimf(credibility.universe, [25, 50, 75])
    credibility['high'] = fuzz.trimf(credibility.universe, [50, 100, 100])

    # Mamdani Rule Base
    rule1 = ctrl.Rule(emotion['high'] | clickbait['high'], credibility['low'])
    rule2 = ctrl.Rule(objectivity['high'] & source['high'] & ~clickbait['high'], credibility['high'])
    rule3 = ctrl.Rule(repetition['high'], credibility['low'])
    rule4 = ctrl.Rule(length['low'] & source['low'], credibility['low'])
    rule5 = ctrl.Rule(emotion['low'] & objectivity['high'] & source['high'] & ~clickbait['high'], credibility['high'])
    rule6 = ctrl.Rule(clickbait['low'] & objectivity['medium'], credibility['medium'])
    rule7 = ctrl.Rule(emotion['high'] & source['low'], credibility['low'])
    rule8 = ctrl.Rule(source['high'] & clickbait['low'] & length['high'], credibility['high'])
    rule9 = ctrl.Rule(repetition['medium'] & emotion['medium'], credibility['medium'])
    rule10 = ctrl.Rule(objectivity['low'] & source['medium'], credibility['low'])
    rule11 = ctrl.Rule(emotion['low'] & repetition['low'] & clickbait['low'], credibility['high'])
    
    # Additional robust rules to catch credible text when source is unknown (i.e. source='medium' at 0.5)
    rule12 = ctrl.Rule(objectivity['high'] & length['high'] & clickbait['low'], credibility['high'])
    # VETO RULE: Do not boost if clickbait/conspiracy is high
    rule13 = ctrl.Rule(emotion['low'] & objectivity['high'] & ~clickbait['high'], credibility['high'])
    rule14 = ctrl.Rule(emotion['low'] & clickbait['low'] & repetition['low'] & source['medium'], credibility['high'])
    
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14]
    
    # Control System
    credibility_ctrl = ctrl.ControlSystem(rules)
    credibility_sim = ctrl.ControlSystemSimulation(credibility_ctrl)
    
    return credibility_sim, credibility

def evaluate_credibility(features):
    """
    Takes a features dictionary and evaluates credibility using Fuzzy Logic.
    Returns the score (0-100), label, and activated rule explanation.
    """
    sim, credibility_var = build_fuzzy_system()
    
    # Compute Inputs safely
    sim.input['emotion'] = max(0, min(1, features.get('emotion', 0.5)))
    sim.input['objectivity'] = max(0, min(1, features.get('objectivity', 0.5)))
    sim.input['source'] = max(0, min(1, features.get('source', 0.5)))
    sim.input['clickbait'] = max(0, min(1, features.get('clickbait', 0.5)))
    sim.input['length'] = max(0, min(1, features.get('length', 0.5)))
    sim.input['repetition'] = max(0, min(1, features.get('repetition', 0.5)))
    
    try:
        sim.compute()
        score = sim.output['credibility']
    except Exception as e:
        # If rules do not fire, return a default
        score = 50.0

    # Determine label
    if score < 35:
        label = "Low"
    elif score < 60:
        label = "Medium"
    else:
        label = "High"

    # Generate basic explanation based on feature thresholds
    explanations = []
    if features.get('emotion', 0) > 0.6: explanations.append("High emotional tone detected.")
    if features.get('clickbait', 0) > 0.5: explanations.append("Contains clickbait phrases.")
    if features.get('repetition', 0) > 0.6: explanations.append("High repetition of words (spam-like).")
    if features.get('objectivity', 0) > 0.7: explanations.append("Highly objective text.")
    if features.get('source', 0.5) > 0.8: explanations.append("Source is highly reliable.")
    if features.get('source', 0.5) < 0.3: explanations.append("Source has low reliability or is known for fake news.")
    if features.get('length', 0) < 0.2: explanations.append("Content is extremely short, lacking depth.")

    if not explanations:
        explanations.append("Mixed signals, resulting in a moderate credibility assessment.")
        
    return {
        "score": round(float(score), 2),
        "label": label,
        "explanation": explanations
    }
