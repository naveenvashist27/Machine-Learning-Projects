import pandas as pd
# NLP Libraries
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer

import ssl
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# Download required NLP resources (runs once)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df = pd.read_csv(
    r"C:\Coder_vashist\Python\project\Spam_Detection\SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

print(df.head())
print(df.shape)
print(df.info())
print(df["label"].value_counts())
# convert labels into numbers so Machine understands \
# # Machine learning models cannot understand text labels.
# Convert:
# ham  -> 0
# spam -> 1
df["label"] = df["label"].map({
    "ham" : 0,
    "spam" : 1
})
print(df.head())
print(df["label"].value_counts())
print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)
# you will oobserver the shape is changed from 5572 to 5169 so this is duplicates dropped 
print(df.duplicated().sum())
print(df["label"].value_counts(normalize=True)*100)
# 0 is 87 & 1 is 12 which means data is imbalanced need to handle 
# now we have installed Natural language toolkit 
# # NLTK provides tools for Natural Language Processing (NLP)
# such as tokenization, stopword removal and stemming.


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load English stopwords once instead of loading them for every word
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()  #initialize an instance of the Porter Stemming algorithm
def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    cleaned_words = []
    for word in words:
         # Keep only alphabetic words 
        if word.isalpha():
             # Remove stopwords
            if word not in stop_words:
                 # Apply stemming
                #  # Reduce different forms of the same word to a common root
# Example: playing, played, plays -> play
# This reduces the vocabulary size and helps the model treat
# similar words as the same feature.
                word = stemmer.stem(word) #like playing ->play
                cleaned_words.append(word) #Stores the processed word.
    return " ".join(cleaned_words)      
          
df["processed_message"] = df["message"].apply(preprocess)
# .apply() is a pandas method that iterates over every row in that column and passes the row's text directly into your function named preprocess
print(df[["message", "processed_message"]].head())   

# Machine Learning algorithms cannot work directly with text.
# TF-IDF converts each message into a numerical feature vector.
# # we need to do this  from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer.
# It learns the vocabulary from the dataset
# and converts every SMS into numerical features. 
tfidf = TfidfVectorizer()
# Think of it as creating a machine that knows how to convert text into numbers. 
X = tfidf.fit_transform(df["processed_message"])
y = df["label"]
print(X.shape)

print(y.shape)
print(len(tfidf.vocabulary_))
print(tfidf.vocabulary_)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets.
# X contains TF-IDF features.
# y contains target labels (0 = ham, 1 = spam).
# 80% data is used for learning.
# 20% data is kept unseen to evaluate the model.
# random_state=42 ensures the same split every time.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Verify the size of each dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.naive_bayes import MultinomialNB

# Create the Multinomial Naive Bayes classifier.
# This algorithm is well suited for text classification tasks.
model = MultinomialNB()

# Train the model using the training dataset.
# X_train = TF-IDF features
# y_train = Correct labels (0 = ham, 1 = spam)
model.fit(X_train, y_train)

# Predict labels for the unseen testing dataset.
# The model has never seen these messages before.
y_pred = model.predict(X_test)

print(y_pred[:10])

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Detailed performance report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
lr = LogisticRegression(max_iter=1000)
# max itr = 1000 bcoz lr learns gradually 
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression")
print(accuracy_score(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))

svm = LinearSVC()
svm.fit(X_train,y_train)
y_pred_svm = svm.predict(X_test)
print("Linear SVM")
print(accuracy_score(y_test,y_pred_svm))
print(classification_report(y_test,y_pred_svm))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest")
print(accuracy_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
results = {
    "Naive Bayes":accuracy_score(y_test,y_pred),
    "Logistic Regression":accuracy_score(y_test,y_pred_lr),
    "Linear SVM":accuracy_score(y_test,y_pred_svm),
    "Random Forest":accuracy_score(y_test,y_pred_rf)
}

print(results)

# ---------------------------------------------------------
# Select and save the best-performing model automatically
# ---------------------------------------------------------

models = {
    "Naive Bayes": model,
    "Logistic Regression": lr,
    "Linear SVM": svm,
    "Random Forest": rf
}

# Find the model with the highest accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model :", best_model_name)
print("Best Accuracy :", results[best_model_name])

# Save the best model
import pickle

with open("spam_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as file:
    pickle.dump(tfidf, file)

print("\nModel saved successfully as 'spam_model.pkl'")
print("Vectorizer saved successfully as 'vectorizer.pkl'")   

print("\nConfusion Matrix")
print(cm)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Spam Detection - Confusion Matrix")
plt.show()