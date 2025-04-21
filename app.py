import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import numpy as np

# Download NLTK assets
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
# Load all models
logistic_model = joblib.load('logistic_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
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

# Class labels
label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral", 3: "Can't tell"}

# Streamlit UI
st.title("🛡️ Hate Speech Detection in Tweets")
st.write("Enter a tweet to check if it's hate speech, offensive or neutral.")

tweet = st.text_area("Tweet")

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])

        # Individual model predictions
        pred_log = logistic_model.predict(vectorized)[0]
        pred_log_prob = np.max(logistic_model.predict_proba(vectorized)) * 100

        pred_rf = rf_model.predict(vectorized)[0]
        pred_rf_prob = np.max(rf_model.predict_proba(vectorized)) * 100

        pred_svm = svm_model.predict(vectorized)[0]
        # No predict_proba for LinearSVC

        if pred_rf_prob+pred_rf_prob < 120:
            majority_vote = 3

        else:
            # Majority vote
            preds = [pred_log, pred_rf, pred_svm]
            majority_vote = Counter(preds).most_common(1)[0][0]

        st.markdown("---")
        st.subheader("✅ Final Output (Majority Vote):")
        st.success(f"**{label_map[majority_vote]}**")

        # Display predictions with confidence
        st.subheader("🔍 Individual Model Predictions:")
        st.write(f"**Logistic Regression:** {label_map[pred_log]} ({pred_log_prob:.2f}% confidence)")
        st.write(f"**Random Forest:** {label_map[pred_rf]} ({pred_rf_prob:.2f}% confidence)")
        st.write(f"**SVM (Linear):** {label_map[pred_svm]} (confidence N/A)")
