# Hate Speech Detection in Tweets

A Machine Learning-powered web app to classify tweets as **Hate Speech**, **Offensive Language**, or **Neither**, using three models: Logistic Regression, SVM, and Random Forest.

---

## 🚀 Features

- Classifies tweet text into 3 categories
- Shows predictions from:
  - ✅ Logistic Regression
  - ✅ Support Vector Machine (SVM)
  - ✅ Random Forest
- Displays confidence score (where applicable)
- Shows final output based on **majority voting**
- Accepts **direct tweet links** and extracts content automatically
**Try Yourself:** [Hate Speech Detector](https://abhishekabhang314-hate-speech-detection-tweeter-app-kherzz.streamlit.app/)

---

## 📊 Dataset Used

**Source:** [Hate Speech Twitter Dataset](https://www.kaggle.com/code/kirollosashraf/hate-speech-and-offensive-language-detection/input)

**Columns:**
- `tweet`: Text of the tweet
- `class`: 0 - Hate Speech, 1 - Offensive Language, 2 - Neither
- Other columns include annotation counts

---

## 🧠 Models Used

- Logistic Regression
- Linear SVM (`LinearSVC`)
- Random Forest Classifier

---

## 🧰 Libraries Used

- `scikit-learn`
- `streamlit`
- `pandas`
- `nltk`
- `joblib`
- `snscrape` (for tweet extraction)

---

## 🖥️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/hate-speech-detector.git
   cd hate-speech-detector
