import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import pdfplumber
from newspaper import Article, Config
import io
import re

def extract_text_from_string(text):
    """Simply returns the text."""
    return text.strip()

def extract_text_from_url(url):
    """Extracts main body text from a news article URL."""
    try:
        # Anti-scraping bypass (Al Jazeera, etc. block basic python bots)
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        text = article.text.strip()
        title = article.title.strip() if article.title else ""
        authors = article.authors if article.authors else []
        publish_date = article.publish_date
        
        # Clean URL to get base domain
        domain = url.split("//")[-1].split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
            
        metadata = {
            "title": title,
            "authors": authors,
            "publish_date": str(publish_date) if publish_date else None,
            "domain": domain
        }
        
        if not text:
            # Fallback if article.text is empty
            return "", {"error": "Could not extract text from this URL.", **metadata}
            
        full_text = f"{title}\n\n{text}"
        return full_text, metadata

    except Exception as e:
        domain = url.split("//")[-1].split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return "", {"error": str(e), "domain": domain}

def extract_text_from_image(image_file):
    """Extracts text from an uploaded image file using Tesseract OCR."""
    try:
        image = Image.open(image_file)
        # Convert to Grayscale to vastly improve Tesseract's ability to read colored/noisy infographics
        gray_image = ImageOps.grayscale(image.convert('RGB'))
        
        # Boost contrast to force white/yellow text to pop against blue/dark backgrounds
        enhancer = ImageEnhance.Contrast(gray_image)
        contrast_img = enhancer.enhance(2.0)
        
        # Use Page Segmentation Mode 11 (Sparse Text) to find scattered text like posters/news-cards
        try:
            # Bilingual extraction for Indian News images
            text = pytesseract.image_to_string(contrast_img, lang='hin+eng', config='--psm 11')
        except Exception:
            # Graceful fallback if tesseract-hin language pack is not installed on the OS
            text = pytesseract.image_to_string(contrast_img, config='--psm 11')
            
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except Exception as e:
        return f"PDF Error: {str(e)}"
