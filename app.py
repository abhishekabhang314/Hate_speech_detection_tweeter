import streamlit as st
import joblib
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np

import nltk
import os
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

logistic_model = joblib.load('logistic_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)
    stop_words = set(stopwords.words("english"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@[\w]*', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral", 3: "Can't tell"}

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="smile",
)

st.title("Hate Speech Detection in Tweets")
st.write("Enter a tweet to check if it's hate speech, offensive or neutral.")
st.write("Project made by:\n 1. Abhishek Abhang\n 2. Soham Kulkarni\n 3. Shantanu Sangle")

tweet = st.text_area("Tweet")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])

        pred_log = logistic_model.predict(vectorized)[0]
        pred_log_prob = np.max(logistic_model.predict_proba(vectorized)) * 100

        pred_rf = rf_model.predict(vectorized)[0]
        pred_rf_prob = np.max(rf_model.predict_proba(vectorized)) * 100

        pred_svm = svm_model.predict(vectorized)[0]

        preds = [pred_log, pred_rf, pred_svm]
        majority_vote = Counter(preds).most_common(1)[0][0]
        st.markdown("---")
        st.subheader("✅ Final Output (Majority Vote):")

        if pred_rf_prob+pred_rf_prob < 120:
            majority_vote = 2
            st.success(f"**Can't tell, but most probably {label_map[majority_vote]}**")

        else:
            st.success(f"**{label_map[majority_vote]}**")

        st.subheader("🔍 Individual Model Predictions:")
        st.write(f"**Logistic Regression:** {label_map[pred_log]} ({pred_log_prob:.2f}% confidence)")
        st.write(f"**Random Forest:** {label_map[pred_rf]} ({pred_rf_prob:.2f}% confidence)")
        st.write(f"**SVM (Linear):** {label_map[pred_svm]} (confidence N/A)")
