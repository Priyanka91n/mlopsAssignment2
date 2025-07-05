import pandas as pd
import numpy as np
import re
import nltk
import mlflow
import mlflow.sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("Downloaded NLTK stopwords.")

# Load data
print("--- 1. Loading and Analyzing Data ---")
try:
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv not found.")
    exit()

df.columns = ['text', 'label']

# Preprocess
print("\n--- 2. Preprocessing Text ---")
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

df['processed_text'] = df['text'].apply(preprocess_text)

# Define hyperparameter grid
tfidf_params = [
    {'max_features': 2500, 'min_df': 5, 'max_df': 0.7},
    {'max_features': 3000, 'min_df': 3, 'max_df': 0.8},
    {'max_features': 2000, 'min_df': 10, 'max_df': 0.6}
]

nb_params = [
    {'alpha': 1.0},
    {'alpha': 0.5},
    {'alpha': 0.1}
]

# Set MLflow experiment
mlflow.set_experiment("Spam_Classifier_Hyperparameter_Tuning")

best_accuracy = 0
best_model = None
best_vectorizer = None
best_run_params = {}

# Loop over combinations
for tfidf_p in tfidf_params:
    vectorizer = TfidfVectorizer(max_features=tfidf_p['max_features'],
                                 min_df=tfidf_p['min_df'],
                                 max_df=tfidf_p['max_df'])
    X = vectorizer.fit_transform(df['processed_text']).toarray()
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for nb_p in nb_params:
        with mlflow.start_run():
            mlflow.log_param("tfidf_max_features", tfidf_p['max_features'])
            mlflow.log_param("tfidf_min_df", tfidf_p['min_df'])
            mlflow.log_param("tfidf_max_df", tfidf_p['max_df'])
            mlflow.log_param("nb_alpha", nb_p['alpha'])

            model = MultinomialNB(alpha=nb_p['alpha'])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(sk_model=model, artifact_path="spam_classifier_model")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_vectorizer = vectorizer
                best_run_params = {**tfidf_p, **nb_p}

# Print best model info
print("\n--- Best Model Found ---")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Params: {best_run_params}")

# Simple prediction function
def predict_spam(text_input, vectorizer, model):
    processed_input = preprocess_text(text_input)
    vectorized_input = vectorizer.transform([processed_input]).toarray()
    prediction = model.predict(vectorized_input)
    return "Spam" if prediction[0] == 1 else "Not Spam (Ham)"

# Test examples
example_spam = "Congratulations! You've won a FREE iPhone! Click this link to claim."
example_ham = "Hey, just checking in. How are you doing today?"

print(f"\nExample: '{example_spam}' -> {predict_spam(example_spam, best_vectorizer, best_model)}")
print(f"Example: '{example_ham}' -> {predict_spam(example_ham, best_vectorizer, best_model)}")

print("\nScript finished. ")
