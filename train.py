import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def find_dataset():
    candidates = [
        "spam_ham_dataset.csv/spam_ham_dataset.csv",
        "spam_ham_dataset.csv",
        "data/spam_ham_dataset.csv",
        "data/spam.csv"
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isfile(path):
            return path
    raise FileNotFoundError("Could not locate the dataset file. Please check dataset location.")

def train():
    os.makedirs("models", exist_ok=True)
    dataset_path = find_dataset()
    print(f"Loading dataset from: {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Standardize columns
    if 'text' not in df.columns:
        if 'v2' in df.columns:
            df = df.rename(columns={'v2': 'text', 'v1': 'label'})
        elif 'Message' in df.columns:
            df = df.rename(columns={'Message': 'text', 'Category': 'label'})

    # Target column
    if 'label_num' not in df.columns:
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # Drop missing values
    df = df.dropna(subset=['text', 'label_num'])
    print(f"Dataset loaded: {len(df)} total emails")
    print(f"Ham emails: {sum(df['label_num'] == 0)}, Spam emails: {sum(df['label_num'] == 1)}")

    # Split dataset
    X = df['text']
    y = df['label_num'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Build Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(max_iter=1000, C=1.0))
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions & Evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Extract Top Predictive Features (Words)
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]

    top_spam_indices = np.argsort(coefs)[-15:][::-1]
    top_ham_indices = np.argsort(coefs)[:15]

    top_spam_words = [feature_names[i] for i in top_spam_indices]
    top_ham_words = [feature_names[i] for i in top_ham_indices]

    # Save Model & Metrics
    model_file = "models/email_classifier.pkl"
    joblib.dump(pipeline, model_file)
    print(f"Saved model to: {model_file}")

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_ham": int(cm[0][0]),
            "false_spam": int(cm[0][1]),
            "false_ham": int(cm[1][0]),
            "true_spam": int(cm[1][1])
        },
        "total_emails": int(len(df)),
        "train_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "top_spam_words": top_spam_words,
        "top_ham_words": top_ham_words
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved metrics to: models/metrics.json")
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    train()
