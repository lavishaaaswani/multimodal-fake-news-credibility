from input_module import extract_text_from_url
from language import detect_and_translate
from preprocessing import clean_text
from features import extract_all_features
from fuzzy_model import evaluate_credibility

url = "https://www.hindustantimes.com/india-news/nitin-nabin-s-resignation-from-assembly-held-up-as-bjp-brainstorms-next-bihar-cm-101774801449649.html"
text, meta = extract_text_from_url(url)
t_t, l = detect_and_translate(text)
c_t = clean_text(t_t)

features = extract_all_features(c_t, meta.get('domain'))
print("FEATURES EXTRACTED:")
for k, v in features.items():
    print(f"  {k}: {round(v, 3)}")

result = evaluate_credibility(features)
print(f"FINAL SCORE: {result['score']}")
print(f"FINAL LABEL: {result['label']}")
print(f"EXPLANATION:")
for idx, r in enumerate(result['explanation']):
    print(f"  {idx+1}. {r}")
