import pandas as pd
import numpy as np
import re
import nltk
import os
import joblib
import mlflow
import mlflow.sklearn

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords if not already
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text preprocessing setup
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')

# Function to preprocess raw text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in set(all_stopwords)]
    return ' '.join(words)

# Load dataset and apply preprocessing
def load_and_preprocess_data():
    df = pd.read_csv('train.csv', header=None)
    df.columns = ['text', 'label']
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

# Train model and log everything with MLflow autologging
def train_model():
    print("[INFO] Loading and preprocessing data...")
    df = load_and_preprocess_data()

    tfidf = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()

    # ✅ Enable autologging
    mlflow.sklearn.autolog()

    print("[INFO] Starting MLflow run and training model...")
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # ✅ Save model and vectorizer manually
        joblib.dump(clf, 'spam_model.pkl')
        joblib.dump(tfidf, 'vectorizer.pkl')

        print(f"[SUCCESS] Model training complete. Accuracy: {acc:.4f}")
        return acc

# Use saved model to predict new input text
def predict_spam(text):
    if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
        return "Model not trained. Please run train_model() first."

    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed]).toarray()
    pred = model.predict(vectorized)

    return "Spam" if pred[0] == 1 else "Not Spam (Ham)"

# Entry point for execution
if __name__ == "__main__":
    accuracy = train_model()
    print(f"[RESULT] Final Model Accuracy: {accuracy:.4f}")
