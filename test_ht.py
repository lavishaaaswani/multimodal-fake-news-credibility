from input_module import extract_text_from_url
from language import detect_and_translate
from preprocessing import clean_text

url = "https://www.hindustantimes.com/world-news/only-you-can-stop-the-war-egypt-president-abdel-fattah-al-sisi-message-to-donald-trump-as-us-iran-war-rages-101774883997884.html"
extracted_text, meta = extract_text_from_url(url)
print('EXTRACTED_LEN:', len(extracted_text))

original_lang, translated_text = detect_and_translate(extracted_text)
print('LANG:', original_lang)
print('TRANSLATED_LEN:', len(translated_text) if translated_text else 0)

cleaned_text = clean_text(translated_text)
print('CLEANED_LEN:', len(cleaned_text) if cleaned_text else 0)
