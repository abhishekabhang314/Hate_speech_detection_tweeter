import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np
import nltk
import os
from nltk.corpus import stopwords

# Configure page
st.set_page_config(
    page_title="Hate Speech Detector",
    page_icon="🚫",
    layout="centered"
)

# Custom CSS with improved dark mode palette
st.markdown("""
    <style>
        :root {
            /* Light Mode Colors */
            --text-color: #333333;
            --bg-color: #ffffff;
            --hate-bg: #ffebee;
            --hate-border: #d32f2f;
            --hate-tag-bg: #ffcdd2;
            --hate-tag-text: #c62828;
            --offensive-bg: #fff8e1;
            --offensive-border: #ffa000;
            --offensive-tag-bg: #ffe0b2;
            --offensive-tag-text: #e65100;
            --neutral-bg: #e8f5e9;
            --neutral-border: #388e3c;
            --neutral-tag-bg: #c8e6c9;
            --neutral-tag-text: #2e7d32;
            --uncertain-bg: #e3f2fd;
            --uncertain-border: #1976d2;
            --uncertain-tag-bg: #bbdefb;
            --uncertain-tag-text: #1565c0;
            --info-bg: #f5f5f5;
            --model-card-bg: #ffffff;
            --confidence-track: #e0e0e0;
            --primary-color: #d63031;
        }

        [data-theme="dark"] {
            /* Dark Mode Colors - More Subtle and Professional */
            --text-color: #e0e0e0;
            --bg-color: #121212;
            --hate-bg: #1f1a1a;
            --hate-border: #cf6679;
            --hate-tag-bg: #2a1e1e;
            --hate-tag-text: #ff8a80;
            --offensive-bg: #211e1a;
            --offensive-border: #ffb74d;
            --offensive-tag-bg: #2a261e;
            --offensive-tag-text: #ffcc80;
            --neutral-bg: #1a211b;
            --neutral-border: #81c784;
            --neutral-tag-bg: #1e2a1f;
            --neutral-tag-text: #a5d6a7;
            --uncertain-bg: #1a1d21;
            --uncertain-border: #64b5f6;
            --uncertain-tag-bg: #1e222a;
            --uncertain-tag-text: #90caf9;
            --info-bg: #1e1e1e;
            --model-card-bg: #1e1e1e;
            --confidence-track: #424242;
            --primary-color: #ff6b6b;
        }

        body {
            color: var(--text-color);
            background-color: var(--bg-color);
        }

        .main {
            max-width: 800px;
            padding: 2rem;
        }

        .stTextArea textarea {
            min-height: 120px;
            border-radius: 8px;
            background-color: var(--bg-color);
            color: var(--text-color);
            border: 1px solid var(--confidence-track);
        }

        .header {
            color: var(--primary-color);
            text-align: center;
        }

        .result-box {
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            color: var(--text-color);
        }

        .hate-speech {
            background-color: var(--hate-bg);
            border-left: 4px solid var(--hate-border);
        }

        .offensive {
            background-color: var(--offensive-bg);
            border-left: 4px solid var(--offensive-border);
        }

        .neutral {
            background-color: var(--neutral-bg);
            border-left: 4px solid var(--neutral-border);
        }

        .uncertain {
            background-color: var(--uncertain-bg);
            border-left: 4px solid var(--uncertain-border);
        }

        .model-card {
            background-color: var(--model-card-bg);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            color: var(--text-color);
            border: 1px solid var(--confidence-track);
        }

        .confidence-bar {
            height: 8px;
            background-color: var(--confidence-track);
            border-radius: 4px;
            margin: 0.5rem 0;
        }

        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
        }

        .info-box {
            background-color: var(--info-bg);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: var(--text-color);
            border: 1px solid var(--confidence-track);
        }

        .tag {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .tag-hate {
            background-color: var(--hate-tag-bg);
            color: var(--hate-tag-text);
            border: 1px solid var(--hate-border);
        }

        .tag-offensive {
            background-color: var(--offensive-tag-bg);
            color: var(--offensive-tag-text);
            border: 1px solid var(--offensive-border);
        }

        .tag-neutral {
            background-color: var(--neutral-tag-bg);
            color: var(--neutral-tag-text);
            border: 1px solid var(--neutral-border);
        }

        .tag-uncertain {
            background-color: var(--uncertain-tag-bg);
            color: var(--uncertain-tag-text);
            border: 1px solid var(--uncertain-border);
        }

        .footer {
            font-size: 0.8rem;
            color: var(--text-color);
            opacity: 0.7;
            text-align: center;
            margin-top: 2rem;
        }

        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: var(--text-color);
        }

        /* Streamlit components */
        .stMarkdown {
            color: var(--text-color) !important;
        }

        .stAlert {
            background-color: var(--info-bg) !important;
        }

        /* Button styling */
        .stButton>button {
            border: 1px solid var(--primary-color);
        }
    </style>
""", unsafe_allow_html=True)


# Load models and resources
@st.cache_resource
def load_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')

    nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
    nltk.data.path.append(nltk_data_path)

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", download_dir=nltk_data_path)
        stop_words = set(stopwords.words("english"))

    logistic_model = joblib.load('logistic_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    return logistic_model, svm_model, rf_model, vectorizer, stop_words


logistic_model, svm_model, rf_model, vectorizer, stop_words = load_resources()
lemmatizer = WordNetLemmatizer()

label_map = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neutral",
    3: "Uncertain"
}

label_colors = {
    0: "hate-speech",
    1: "offensive",
    2: "neutral",
    3: "uncertain"
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]*', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# App header
st.markdown("<h1 class='header'>🚫 Hate Speech Detection</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; margin-bottom: 2rem;'>
    Analyze text for hate speech, offensive language, or neutral content
    </p>
""", unsafe_allow_html=True)

# Input area
tweet = st.text_area(
    "Enter text to analyze:",
    placeholder="Paste or type the text you want to analyze here...",
    help="The text will be processed to remove URLs, mentions, and special characters before analysis."
)

# Info box with tags
st.markdown("""
    <div class='info-box'>
        <h4 style='margin-top: 0;'>Classification Categories:</h4>
        <span class='tag tag-hate'>Hate Speech</span>
        <span class='tag tag-offensive'>Offensive Language</span>
        <span class='tag tag-neutral'>Neutral</span>
        <span class='tag tag-uncertain'>Uncertain</span>
        <p style='margin-bottom: 0; font-size: 0.9rem;'>
        Uses three ML models with majority voting for final classification.
        </p>
    </div>
""", unsafe_allow_html=True)

# Analyze button
if st.button("Analyze Text", type="primary", use_container_width=True):
    if not tweet.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned = clean_text(tweet)
            vectorized = vectorizer.transform([cleaned])

            # Get predictions
            pred_log = logistic_model.predict(vectorized)[0]
            pred_log_prob = np.max(logistic_model.predict_proba(vectorized)) * 100

            pred_rf = rf_model.predict(vectorized)[0]
            pred_rf_prob = np.max(rf_model.predict_proba(vectorized)) * 100

            pred_svm = svm_model.predict(vectorized)[0]

            preds = [pred_log, pred_rf, pred_svm]
            majority_vote = Counter(preds).most_common(1)[0][0]

            # Adjust for low confidence
            if pred_log_prob + pred_rf_prob < 120:
                majority_vote = 3  # "Uncertain"

            # Display results
            st.markdown("---")

            # Final result box
            result_class = label_colors.get(majority_vote, "uncertain")
            st.markdown(f"""
                <div class='result-box {result_class}'>
                    <h3 style='margin-top: 0; margin-bottom: 0.5rem;'>Final Classification</h3>
                    <h2 style='margin-top: 0; margin-bottom: 1rem;'>{label_map[majority_vote]}</h2>
                    <p style='margin-bottom: 0;'><strong>Analyzed text:</strong> <i>"{tweet[:100]}{'...' if len(tweet) > 100 else ''}"</i></p>
                </div>
            """, unsafe_allow_html=True)

            # Model predictions section
            st.subheader("Model Predictions")

            # Logistic Regression
            with st.expander("🔍 Logistic Regression", expanded=False):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label="Prediction", value=label_map[pred_log])
                with col2:
                    st.write(f"Confidence: {pred_log_prob:.1f}%")
                    st.markdown(f"""
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {pred_log_prob}%'></div>
                        </div>
                    """, unsafe_allow_html=True)

            # Random Forest
            with st.expander("🌲 Random Forest", expanded=False):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label="Prediction", value=label_map[pred_rf])
                with col2:
                    st.write(f"Confidence: {pred_rf_prob:.1f}%")
                    st.markdown(f"""
                        <div class='confidence-bar'>
                            <div class='confidence-fill' style='width: {pred_rf_prob}%'></div>
                        </div>
                    """, unsafe_allow_html=True)

            # SVM
            with st.expander("⚡ Support Vector Machine", expanded=False):
                st.metric(label="Prediction", value=label_map[pred_svm])
                st.info("This model doesn't provide probability estimates by default")

            # Voting explanation
            st.markdown("""
                <div class='info-box'>
                    <h4 style='margin-top: 0;'>How It Works</h4>
                    <p>The system uses majority voting among three models. When confidence scores from Logistic Regression and Random Forest are both low (<60% each), the system defaults to "Uncertain" to avoid unreliable classifications.</p>
                </div>
            """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p>Hate Speech Detection System | For research and educational purposes</p>
        <p>Note: This is an automated system and may not always be accurate.</p>
    </div>
""", unsafe_allow_html=True)