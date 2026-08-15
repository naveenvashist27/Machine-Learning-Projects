# ---------------------------------------------------------
# Spam Detection - Prediction File
# This file loads the saved model and predicts
# whether a new SMS is Spam or Ham.
# ---------------------------------------------------------

import pickle
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources (runs only if needed)
# ---------------------------------------------------------
nltk.download("punkt") #without punkt word_tokenize() does not work 
nltk.download("stopwords")

# Load the saved Machine Learning model
# ---------------------------------------------------------
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the saved TF-IDF Vectorizer
# ---------------------------------------------------------
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Initialize NLP tools
# ---------------------------------------------------------
stemmer = PorterStemmer()
# Load stopwords only once
stop_words = set(stopwords.words("english"))

# Text Preprocessing Function
# This MUST be the same as used while training.
# ---------------------------------------------------------
def preprocess(text):
    # Convert into lowercase
    text = text.lower()
    # Split sentence into words
    words = word_tokenize(text)

    cleaned_words = []
    for word in words:
        # Keep only alphabets
        if word.isalpha():
            # Remove common English words
            if word not in stop_words:
                # Convert playing -> play
                word = stemmer.stem(word)
                cleaned_words.append(word)
    return " ".join(cleaned_words)


# Take user input
# ---------------------------------------------------------
message = input("Enter your SMS : ")

# Preprocess the message
# ---------------------------------------------------------
processed_message = preprocess(message)

# Convert text into TF-IDF features
# ---------------------------------------------------------
vector = vectorizer.transform([processed_message])

# Predict
# ---------------------------------------------------------
prediction = model.predict(vector)

# Display Result
# ---------------------------------------------------------
if prediction[0] == 1:
    print("\n🚨 This is a SPAM message.")
else:
    print("\n✅ This is a HAM message.")
