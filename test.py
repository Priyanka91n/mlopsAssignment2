import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os
 
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
    df = pd.read_csv('train.csv')
    df.columns = ['text', 'label']
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df
 
def train_model():
    df = load_and_preprocess_data()
    tfidf = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'spam_model.pkl')
    joblib.dump(tfidf, 'vectorizer.pkl')
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)
 
def get_best_params():
    df = load_and_preprocess_data()
    tfidf = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7)
    X = tfidf.fit_transform(df['processed_text']).toarray()
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(clf, param_grid, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_params_
 
def predict_spam(text):
    if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
        return "Model not trained."
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed]).toarray()
    pred = model.predict(vectorized)
    return "Spam" if pred[0] == 1 else "Not Spam (Ham)"
 