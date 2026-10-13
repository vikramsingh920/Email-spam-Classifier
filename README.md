---
title: Email Classifier
emoji: 📧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 📧 MailGuard AI: End-to-End Email Spam Classifier

An intelligent, full-stack Machine Learning web application that classifies emails into **Legitimate (Ham)** or **Spam / Phishing** in real-time using Natural Language Processing (NLP).

Built with **Python**, **Scikit-Learn (TF-IDF + Logistic Regression)**, **FastAPI**, and a **Modern Dark-Mode Web Dashboard**.

---

## 🚀 Key Features

* **Real-time Email Analysis:** Instantly analyze email subjects and body text.
* **Dual-Class Probabilities:** View exact percentage breakdown for Ham vs. Spam.
* **Suspicious Trigger Word Detection:** Automatically detects spam keywords and phishing markers.
* **Interactive Sample Presets:** 1-click test buttons for Business Meetings, Lottery Scams, and Account Phishing.
* **High Production Accuracy:** Trained and evaluated with rigorous stratification.

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **98.65%** |
| **Precision** | **97.67%** |
| **Recall** | **97.67%** |
| **F1-Score** | **97.67%** |

### Confusion Matrix (Test Set: 1,035 emails)
* **True Ham:** 728 / 735 (99.0% caught correctly)
* **True Spam:** 293 / 300 (97.7% caught correctly)
* **False Alarms:** Only 7 emails misclassified

---

## 🏗️ Project Architecture

```
Email-Classifier/
├── models/
│   ├── email_classifier.pkl  # Trained ML pipeline (TF-IDF + Classifier)
│   └── metrics.json          # Cached evaluation metrics & top signals
├── static/
│   ├── index.html            # Responsive UI with glassmorphism
│   ├── style.css             # Tailored dark theme design
│   └── app.js                # Frontend reactive state & API integration
├── train.py                  # End-to-end model training script
├── app.py                    # FastAPI server exposing REST API & web UI
├── requirements.txt          # Python dependencies
└── README.md
```

---

## ⚡ Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
This cleans the dataset, trains the pipeline, logs the evaluation metrics, and exports the serialized model to `models/email_classifier.pkl`.

### 3. Launch the Web App
```bash
python app.py
```
Open your browser at:
👉 **`http://127.0.0.1:8050`**

---

## 🔌 API Documentation

### `POST /api/predict`
**Request Body:**
```json
{
  "subject": "Urgent verification needed",
  "body": "Please wire the funds immediately to claim your lottery prize."
}
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 99.1,
  "is_spam": true,
  "probabilities": {
    "ham": 0.9,
    "spam": 99.1
  },
  "detected_keywords": ["urgent", "wire transfer", "lottery", "prize"]
}
```

### `GET /api/metrics`
Returns model benchmark metrics, confusion matrix, and top predictive keywords.
