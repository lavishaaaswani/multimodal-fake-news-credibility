import streamlit as st
import plotly.graph_objects as go
from annotated_text import annotated_text
import os

from input_module import extract_text_from_string, extract_text_from_url, extract_text_from_image, extract_text_from_pdf
from language import detect_and_translate
from preprocessing import clean_text
from features import extract_all_features, CLICKBAIT_WORDS
from fuzzy_model import evaluate_credibility

# Config
st.set_page_config(page_title="DeepTruth | AI Credibility", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

# Extreme Premium Theme CSS Override
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;900&display=swap');

    :root {
        --bg:        #0A0A0F;
        --surface:   rgba(255,255,255,0.025);
        --border:    rgba(255,255,255,0.07);
        --violet:    #7B4FFF;
        --pink:      #FF3C78;
        --text:      #E8E6FF;
        --muted:     #6B5FBB;
        --dim:       #4A4870;
        --green:     #27C88C;
        --amber:     #F59E0B;
        --red:       #FF3C78;
        --font-mono: 'Space Mono', monospace;
        --font-body: 'Outfit', sans-serif;
    }

    /* ─── Targeted Font Override (Prevents Icon Overlap) ─── */
    html, body, p, div, label, span, li, input, textarea, summary { 
        font-family: var(--font-body); 
    }
    .material-symbols-rounded, .material-icons {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--bg) !important;
        background-image:
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.012) 2px, rgba(255,255,255,0.012) 4px),
            radial-gradient(ellipse 70% 50% at 15% 20%, rgba(120,80,255,0.1) 0%, transparent 55%),
            radial-gradient(ellipse 60% 40% at 85% 75%, rgba(255,60,120,0.07) 0%, transparent 55%);
        color: var(--text) !important;
    }

    [data-testid="stHeader"] { background-color: transparent !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090912 0%, #0C0B18 100%) !important;
        border-right: 1px solid var(--border) !important;
        border-left: 3px solid var(--violet) !important;
    }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    @keyframes titleIn {
        0%   { opacity: 0; letter-spacing: 12px; filter: blur(6px); }
        100% { opacity: 1; letter-spacing: -1px;  filter: blur(0px); }
    }

    .premium-title {
        animation: titleIn 1s cubic-bezier(0.16,1,0.3,1) both;
        text-align: center;
        font-family: var(--font-mono) !important;
        font-size: 3.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -1px;
        line-height: 1;
        background: linear-gradient(135deg, #fff 0%, #C8BBFF 40%, var(--pink) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 40px rgba(123,79,255,0.2));
    }

    .premium-subtitle {
        animation: titleIn 1.1s 0.15s cubic-bezier(0.16,1,0.3,1) both;
        text-align: center;
        font-family: var(--font-mono) !important;
        font-size: 0.6rem !important;
        letter-spacing: 5px !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        margin-bottom: 48px !important;
    }

    div[data-testid="stVerticalBlock"] > div > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        position: relative;
        /* WARNING: Never use overflow: hidden here; it horizontally truncates Expander text! */
    }
    div[data-testid="stVerticalBlock"] > div > div::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(123,79,255,0.5), transparent);
        pointer-events: none;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--violet) 0%, var(--pink) 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 12px 26px !important;
        box-shadow: 0 4px 24px rgba(123,79,255,0.35) !important;
        transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 10px 36px rgba(123,79,255,0.5), 0 0 0 1px rgba(255,60,120,0.3) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98) !important;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: rgba(0,0,0,0.35) !important;
        color: var(--text) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        font-family: var(--font-body) !important;
        caret-color: var(--violet);
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--violet) !important;
        background-color: rgba(123,79,255,0.05) !important;
        box-shadow: 0 0 0 3px rgba(123,79,255,0.15), 0 0 20px rgba(123,79,255,0.08) !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--dim) !important;
        font-style: italic;
    }

    h1, h2, h3 {
        font-family: var(--font-mono) !important;
        color: var(--text) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    p, span, label, li { color: #94A3B8 !important; }

    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(123,79,255,0.3), rgba(255,60,120,0.2), transparent) !important;
        margin: 36px 0 !important;
    }

    [data-testid="stExpander"] {
        background: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: var(--font-mono) !important;
        font-size: 0.7rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: var(--violet) !important;
    }

    [data-testid="stRadio"] label,
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        font-family: var(--font-body) !important;
        color: var(--text) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(0,0,0,0.25) !important;
        border: 2px dashed rgba(123,79,255,0.2) !important;
        border-radius: 14px !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(123,79,255,0.45) !important;
    }

    [data-testid="stSpinner"] > div {
        border-top-color: var(--violet) !important;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--violet), var(--pink));
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='premium-title'>NEWS ANALYSIS</h1>", unsafe_allow_html=True)
st.markdown("<p class='premium-subtitle'>Mamdani Neuro-Fuzzy Analytics</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 style='color: white; text-align: center;'>DATALINK</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
input_option = st.sidebar.radio("Select Input Vector:", ["URL Scrape", "Raw Text", "Image OCR", "PDF Scan"])

def render_gauge_chart(score, label, color):
    """Deep Dark Theme Plotly Speedometer"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{label.upper()}", 'font': {'size': 28, 'color': color, 'family': 'Inter'}},
        number = {'font': {'color': 'white', 'size': 50}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': "#64748B"}},
            'bar': {'color': color, 'thickness': 0.15},
            'bgcolor': "rgba(255,255,255,0.02)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': "rgba(255, 60, 120, 0.1)"},     # Pink Zone (Bad)
                {'range': [35, 60], 'color': "rgba(255, 255, 255, 0.05)"},  # Neutral
                {'range': [60, 100], 'color': "rgba(123, 79, 255, 0.1)"}],  # Violet Zone (Good)
        }
    ))
    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def render_feature_chart(features):
    """Deep Dark Theme Feature Bar Chart"""
    categories = list(features.keys())
    categories_ui = [c.upper() for c in categories]
    values = [round(v, 2) for v in features.values()]
    
    colors = []
    for c in categories:
        if c in ['emotion', 'clickbait', 'repetition']:
            colors.append('rgba(255, 60, 120, 0.85)')   # pink for bad signals
        else:
            colors.append('rgba(123, 79, 255, 0.85)')   # violet for good signals

    fig = go.Figure(go.Bar(
        x=values,
        y=categories_ui,
        orientation='h',
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont=dict(size=14, color='white', family='Inter')
    ))
    
    fig.update_layout(
        title=dict(text="<b>LINGUISTIC VECTORS</b>", font=dict(color="white", size=18, family="Inter")),
        xaxis=dict(range=[0, 1.1], visible=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color='#94A3B8', family='Inter')),
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def highlight_text(text, clickbait_lexicon):
    """Robust XAI multi-axial highlighting (Clickbait, Emotion, Repetition)."""
    import re
    from collections import Counter
    from textblob import TextBlob
    
    # 1. Math for Repetition Penalty
    words_only = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'have', 'has', 'had'}
    content_words = [w for w in words_only if w not in stop_words]
    counts = Counter(content_words)
    # Flag words repeated >3 times mathematically occupying >5% of the total article
    repetitive_words = {w for w, c in counts.items() if c > 3 and c / max(len(content_words), 1) > 0.05}
    
    # 2. Math for Clickbait Multi-word Phrases
    lexicon_set = set(w.lower() for w in clickbait_lexicon if w.strip())
    sorted_lex = sorted(list(lexicon_set), key=len, reverse=True)
    if sorted_lex:
        escaped_lex = [re.escape(term) for term in sorted_lex]
        pattern = r'\b(' + '|'.join(escaped_lex) + r')\b'
        parts = re.split(pattern, text, flags=re.IGNORECASE)
    else:
        parts = [text]
        
    annotated = []
    for part in parts:
        if not part: continue
        
        # Exact Phrase Match (Pink)
        if part.lower() in lexicon_set:
            annotated.append((part, "Fake Signal", "#FF3C78"))
            continue
            
        # Standard grammatical evaluation
        subparts = re.split(r'([^\w]+)', part)
        for sub in subparts:
            if not sub: continue
            clean_w = sub.lower()
            
            # Spam Repetition Engine (Violet)
            if clean_w in repetitive_words:
                annotated.append((sub, "Repetitive Spam", "#6B5FBB"))
                continue
                
            # Dictionary Polarity Engine (Amber)
            if len(clean_w) >= 4 and clean_w.isalpha():
                if abs(TextBlob(sub).sentiment.polarity) >= 0.5:
                    annotated.append((sub, "Extreme Emotion", "#F59E0B"))
                    continue
                    
            # Safe standard text
            annotated.append(sub)
            
    return annotated

# Main Execution Flow
domain = None
raw_text = ""
metadata = {}

with st.container():
    if input_option == "Raw Text":
        raw_input = st.text_area("Paste Article Buffer:", height=180, placeholder="Awaiting raw text input...")
        if st.button("INITIATE ANALYSIS", use_container_width=True):
            if raw_input:
                raw_text = extract_text_from_string(raw_input)
            else:
                st.error("Text buffer empty.")

    elif input_option == "URL Scrape":
        url_input = st.text_input("Enter Target URL:", placeholder="https://...")
        if st.button("BYPASS & SCRAPE", use_container_width=True):
            if url_input:
                with st.spinner("Executing Anti-Bot Bypass..."):
                    raw_text, metadata = extract_text_from_url(url_input)
                    domain = metadata.get('domain', None)
                    if 'error' in metadata and not raw_text:
                        st.error(f"Target blocked scrape attempt: {metadata['error']}")
            else:
                st.error("No URL provided.")

    elif input_option == "Image OCR":
        uploaded_file = st.file_uploader("Upload Image File", type=['png', 'jpg', 'jpeg'])
        if st.button("EXTRACT PIXELS", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Processing Matrix via Tesseract..."):
                    raw_text = extract_text_from_image(uploaded_file)
                    if raw_text.startswith("OCR Error"):
                        st.error(raw_text)
                        raw_text = ""
            else:
                st.error("No image buffer loaded.")
                
    elif input_option == "PDF Scan":
        uploaded_file = st.file_uploader("Upload Document Artifact", type=['pdf'])
        if st.button("SCAN DOCUMENT", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner("Executing PDF Plumber..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    if raw_text.startswith("PDF Error"):
                        st.error(raw_text)
                        raw_text = ""
            else:
                st.error("No document buffer loaded.")


if raw_text:
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.spinner("Translating & Stripping Noise..."):
        translated_text, original_lang = detect_and_translate(raw_text)
        cleaned_text = clean_text(translated_text)
        
    if not cleaned_text or len(cleaned_text) < 10:
        st.warning("Payload extremely small. Validation aborted.")
    else:
        with st.spinner("Engaging Mamdani Inference Engine..."):
            features = extract_all_features(cleaned_text, domain)
            result = evaluate_credibility(features)
            
        st.markdown("<h2 style='text-align: center; color: white;'>NEURO-FUZZY DIAGNOSTIC</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="large")
        
        score = result['score']
        label = result['label']
        # Cyberpunk colors for dark theme
        color = "#7B4FFF" if label == "High" else "#C8BBFF" if label == "Medium" else "#FF3C78"
        
        with col1:
            st.plotly_chart(render_gauge_chart(score, label, color), use_container_width=True)
            
            st.markdown("### 🔍 INFERENCE LOG")
            for exp in result['explanation']:
                if any(bad in exp.lower() for bad in ['emotional', 'clickbait', 'spam', 'fake', 'low']):
                   st.markdown(f"<div style='border-left: 4px solid #FF3C78; padding-left: 12px; margin-bottom: 10px; background: rgba(255,60,120,0.06); border-radius: 0 8px 8px 0; padding: 8px 12px; color: #E8E6FF;'><span style='color:#FF7BAA; font-family: Space Mono, monospace; font-size: 0.65rem; letter-spacing: 1px;'>ALERT</span><br>{exp}</div>", unsafe_allow_html=True)
                elif any(good in exp.lower() for good in ['objective', 'reliable', 'high']):
                   st.markdown(f"<div style='border-left: 4px solid #27C88C; padding-left: 12px; margin-bottom: 10px; background: rgba(39,200,140,0.06); border-radius: 0 8px 8px 0; padding: 8px 12px; color: #E8E6FF;'><span style='color:#6DECB8; font-family: Space Mono, monospace; font-size: 0.65rem; letter-spacing: 1px;'>VALID</span><br>{exp}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='border-left: 4px solid #7B4FFF; padding-left: 12px; margin-bottom: 10px; background: rgba(123,79,255,0.06); border-radius: 0 8px 8px 0; padding: 8px 12px; color: #E8E6FF;'><span style='color:#A99BFF; font-family: Space Mono, monospace; font-size: 0.65rem; letter-spacing: 1px;'>INFO</span><br>{exp}</div>", unsafe_allow_html=True)

        with col2:
            st.plotly_chart(render_feature_chart(features), use_container_width=True)
            
            if metadata and len(metadata) > 1:
                with st.expander("VIEW HEADER METADATA"):
                    st.json({k: v for k, v in metadata.items() if k != 'error' and v})
            
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>XAI EXPLAINABILITY BLOCK</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Detected Lang: <code>{original_lang}</code></p>", unsafe_allow_html=True)
        
        with st.expander("INSPECT RAW PARSED PAYLOAD", expanded=False):
            # Safe wrapper for the Annotated Text module inside a dark expander
            st.markdown("""
            <style>.st-emotion-cache-16txtl3 {background-color: transparent !important;}</style>
            """, unsafe_allow_html=True)
            annotated = highlight_text(cleaned_text, CLICKBAIT_WORDS)
            annotated_text(*annotated)
