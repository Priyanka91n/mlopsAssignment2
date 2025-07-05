import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    print("Downloaded NLTK stopwords.")

# --- 1. Load and Analyze the Data ---
print("--- 1. Loading and Analyzing Data ---")
try:
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv not found. Please make sure the file is in the same directory as the script.")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Check the distribution of the 'label' column
print("\nDistribution of 'label' (0: Ham, 1: Spam):")
print(df['label'].value_counts())

# Get basic information about the dataframe
print("\nDataFrame Info:")
df.info()

# Rename columns for clarity if they are 'sms' and 'label'
# Based on the content fetch, the columns are 'sms' and 'label'
df.columns = ['text', 'label']
print("\nColumns renamed to 'text' and 'label'.")

# --- 2. Text Preprocessing ---
print("\n--- 2. Preprocessing Text Data ---")
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
# Remove common words that might not be useful for spam detection but are in stopwords
all_stopwords.remove('not') # 'not' can be important for sentiment/spam detection

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters and replace with space
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    # Stemming and remove stopwords
    words = text.split()
    words = [ps.stem(word) for word in words if word not in set(all_stopwords)]
    text = ' '.join(words)
    return text

# Apply preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)
print("\nText preprocessing complete. Example of processed text:")
print(df['processed_text'].head())

# --- 3. Feature Extraction (TF-IDF) ---
print("\n--- 3. Extracting Features using TF-IDF ---")
# Initialize TF-IDF Vectorizer
# max_features: consider only top N words by frequency
# min_df: ignore terms that appear in less than this many documents
# max_df: ignore terms that appear in more than this proportion of documents
tfidfconverter = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.7)

# Fit and transform the processed text
X = tfidfconverter.fit_transform(df['processed_text']).toarray()
y = df['label']

print(f"Shape of feature matrix (X): {X.shape}")
print(f"Shape of target vector (y): {y.shape}")

# --- 4. Model Building ---
print("\n--- 4. Building the Model ---")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# Initialize and train the Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
print("Multinomial Naive Bayes classifier trained successfully.")

# --- 5. Model Evaluation ---
print("\n--- 5. Evaluating the Model ---")
y_pred = classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Spam Classifier')
plt.show()

# --- 6. Simple Prediction Function (for demonstration) ---
print("\n--- 6. Demonstration: Making a prediction ---")

def predict_spam(text_input, vectorizer, model):
    processed_input = preprocess_text(text_input)
    # Transform the input text using the *fitted* vectorizer
    vectorized_input = vectorizer.transform([processed_input]).toarray()
    prediction = model.predict(vectorized_input)
    return "Spam" if prediction[0] == 1 else "Not Spam (Ham)"

# Test with some example messages
example_spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
example_ham = "Ok lar... Joking wif u oni..."
example_new_spam = "Congratulations! You've won a FREE iPhone! Click this link now to claim your prize!"
example_new_ham = "Hey, just checking in. How are you doing today?"

print(f"\nExample 1: '{example_spam}' -> {predict_spam(example_spam, tfidfconverter, classifier)}")
print(f"Example 2: '{example_ham}' -> {predict_spam(example_ham, tfidfconverter, classifier)}")
print(f"Example 3: '{example_new_spam}' -> {predict_spam(example_new_spam, tfidfconverter, classifier)}")
print(f"Example 4: '{example_new_ham}' -> {predict_spam(example_new_ham, tfidfconverter, classifier)}")

print("\nScript finished.")