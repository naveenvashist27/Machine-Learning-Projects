import streamlit as st
from pathlib import Path
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📩",
    layout="centered"
)

# Cached so it downloads cleanly on startup

@st.cache_resource
def load_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

load_nltk_resources()

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

with open(BASE_DIR / "spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open(BASE_DIR / "vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# -----------------------------
# NLP Tools
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Preprocessing
# Same preprocessing used during training
# -----------------------------
def preprocess(text):

    text = text.lower()

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word.isalpha():

            if word not in stop_words:

                word = stemmer.stem(word)

                cleaned_words.append(word)

    return " ".join(cleaned_words)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📖 About")

st.sidebar.info(
"""
This project detects whether an SMS is:

✅ Ham (Normal Message)

🚨 Spam Message

Model Used:
- Linear SVM

Feature Extraction:
- TF-IDF

Text Processing:
- NLTK
"""
)

st.title("📩 Spam SMS Detector")

st.markdown(
"""
Enter an SMS below and click **Predict**.

The application will classify the message as **Spam** or **Ham** using a trained Machine Learning model.
"""
)

message = st.text_area(
    "Enter your SMS",
    height=180,
    placeholder="Example: Congratulations! You won ₹50,000..."
)

# Predict Button

if st.button("🔍 Predict"):

    if message.strip() == "":

        st.warning("⚠ Please enter an SMS message.")

    else:

        processed_message = preprocess(message)

        vector = vectorizer.transform([processed_message])

        prediction = model.predict(vector)

        st.divider()

        st.subheader("Prediction")

        if prediction[0] == 1:

            st.error("🚨 SPAM MESSAGE")

        else:

            st.success("✅ HAM MESSAGE")

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Built using Python • Scikit-Learn • NLTK • Streamlit & Buillt By Naveen Vashist"
)
