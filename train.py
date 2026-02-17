import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preprocess_text

def load_and_prepare_data():
    # --- 1. Load Enron Dataset ---
    print("Loading Enron dataset...")
    try:
        df_enron = pd.read_csv('enron emails.csv', encoding='latin-1', on_bad_lines='skip')
        # Rename columns
        df_enron.rename(columns={'Message': 'text', 'Category': 'label_text'}, inplace=True)
        # Drop Unnamed: 2
        df_enron.drop(columns=['Unnamed: 2'], inplace=True, errors='ignore')
        # Drop missing values
        df_enron.dropna(subset=['text', 'label_text'], inplace=True)
        # Map labels
        df_enron['label'] = df_enron['label_text'].map({'ham': 0, 'spam': 1})
        # Ensure label is int and text/label columns exist
        df_enron = df_enron[['text', 'label']]
        df_enron.dropna(subset=['label'], inplace=True)
        df_enron['label'] = df_enron['label'].astype(int)
        print(f"Enron dataset loaded: {len(df_enron)} rows")
    except FileNotFoundError:
        print("Error: 'enron emails.csv' not found.")
        df_enron = pd.DataFrame(columns=['text', 'label'])

    # --- 2. Load Spam Ham Dataset ---
    print("Loading Spam Ham dataset...")
    try:
        df_new = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1', on_bad_lines='skip', sep='\t', header=None, names=['label_text', 'text'])
        # Map labels
        df_new['label'] = df_new['label_text'].map({'ham': 0, 'spam': 1})
        df_new = df_new[['text', 'label']]
        df_new.dropna(inplace=True)
        df_new['label'] = df_new['label'].astype(int)
        print(f"Spam Ham dataset loaded: {len(df_new)} rows")
    except FileNotFoundError:
        print("Error: 'spam_ham_dataset.csv' not found.")
        df_new = pd.DataFrame(columns=['text', 'label'])

    # --- 3. Combine Datasets ---
    print("Combining datasets...")
    if not df_enron.empty and not df_new.empty:
        df_combined = pd.concat([df_enron, df_new], ignore_index=True)
    elif not df_enron.empty:
        df_combined = df_enron.copy()
    elif not df_new.empty:
        df_combined = df_new.copy()
    else:
        raise ValueError("No data loaded!")

    # Drop duplicates on text
    df_combined.drop_duplicates(subset=['text'], inplace=True)
    print(f"Total unique emails: {len(df_combined)}")

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_combined

def train_model():
    # 1. Load Data
    df = load_and_prepare_data()

    # 2. Preprocessing
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # 3. Train-Test Split
    X = df['cleaned_text']
    y = df['label']
    
    print("Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 4. Vectorization
    print("Vectorizing data...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Save vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("vectorizer.pkl saved.")

    # 5. Model Training (SVM)
    print("Training SVM model...")
    classifier = SVC(kernel='linear', probability=True, random_state=42)
    classifier.fit(X_train_tfidf, y_train)

    # Save classifier
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    print("classifier.pkl saved.")

    # 6. Evaluation
    print("Evaluating model...")
    y_pred = classifier.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("confusion_matrix.png saved.")

if __name__ == "__main__":
    train_model()
