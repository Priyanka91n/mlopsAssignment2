import pandas as pd
import numpy as np
import re
import nltk
import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup debug logs
logging.basicConfig(level=logging.INFO)

# Set custom tracking folder to avoid OneDrive conflicts
mlflow.set_tracking_uri("file:///C:/mlflow_logs")

# Download NLTK stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
if 'not' in all_stopwords:
    all_stopwords.remove('not')

def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in set(all_stopwords)]
    return ' '.join(words)

def load_and_preprocess_data():
    filepath = 'train.csv'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File not found: {os.path.abspath(filepath)}")
    df = pd.read_csv(filepath, header=None)
    df.columns = ['text', 'label']
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

def plot_and_log_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")

    img_path = f"conf_matrix_{title.replace(' ', '_')}.png"
    plt.savefig(img_path)
    plt.close()

    if os.path.exists(img_path):
        mlflow.log_artifact(img_path)
        os.remove(img_path)
    else:
        logging.warning(f"Image file missing: {img_path}")

def train_and_log_model(name, model, X_train, y_train, X_test, y_test, tfidf):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name=name):
        logging.info(f"[INFO] Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)

        # Save and log model + vectorizer
        model_file = f"{name}_model.pkl"
        vect_file = f"{name}_vectorizer.pkl"

        joblib.dump(model, model_file)
        joblib.dump(tfidf, vect_file)

        for file in [model_file, vect_file]:
            if os.path.exists(file):
                mlflow.log_artifact(file)
                os.remove(file)
            else:
                logging.warning(f"[WARNING] Missing file for artifact log: {file}")

        # Confusion matrix artifact
        plot_and_log_confusion_matrix(y_test, y_pred, name)

        print(f"[RESULT] {name} Accuracy: {acc:.4f}\n")

def run_experiments():
    print("[INFO] Loading and preprocessing data...")
    df = load_and_preprocess_data()

    tfidf = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Spam Classification Comparison")

    train_and_log_model("Naive Bayes", MultinomialNB(), X_train, y_train, X_test, y_test, tfidf)
    train_and_log_model("Logistic Regression", LogisticRegression(max_iter=200), X_train, y_train, X_test, y_test, tfidf)
    train_and_log_model("Random Forest", RandomForestClassifier(n_estimators=100), X_train, y_train, X_test, y_test, tfidf)

    print("[INFO] All experiments completed.")

def predict_spam(text, model_file="Naive Bayes_model.pkl", vect_file="Naive Bayes_vectorizer.pkl"):
    if not os.path.exists(model_file) or not os.path.exists(vect_file):
        return "Model not trained. Please run the script first."

    model = joblib.load(model_file)
    vectorizer = joblib.load(vect_file)

    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed]).toarray()
    pred = model.predict(vectorized)

    return "Spam" if pred[0] == 1 else "Not Spam (Ham)"

if __name__ == "__main__":
    run_experiments()
