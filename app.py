import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict

app = FastAPI(title="Email Spam Classifier API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Smart Path Detection (Supports both subfolders and flat root files)
if os.path.exists(os.path.join(BASE_DIR, "models", "email_classifier.pkl")):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "email_classifier.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
else:
    MODEL_PATH = os.path.join(BASE_DIR, "email_classifier.pkl")
    METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

if os.path.exists(os.path.join(BASE_DIR, "static", "index.html")):
    STATIC_DIR = os.path.join(BASE_DIR, "static")
else:
    STATIC_DIR = BASE_DIR

pipeline = None
metrics_data = {}

def get_pipeline():
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = joblib.load(MODEL_PATH)
        else:
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    return pipeline

def load_metrics():
    global metrics_data
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    return metrics_data

@app.on_event("startup")
def startup_event():
    try:
        get_pipeline()
        load_metrics()
        print("Model and metrics loaded successfully.")
    except Exception as e:
        print(f"Warning during startup: {e}")

class EmailRequest(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    text: Optional[str] = ""

class EmailPrediction(BaseModel):
    prediction: str
    label_num: int
    confidence: float
    is_spam: bool
    probabilities: Dict[str, float]
    word_count: int
    char_count: int
    detected_keywords: List[str]

SPAM_KEYWORDS = [
    "lottery", "winner", "cash", "prize", "free", "urgent", "act now", "claim",
    "viagra", "pills", "guaranteed", "investment", "cryptocurrency", "bitcoin",
    "password", "verify account", "bank transfer", "millions", "bonus", "cheap",
    "risk free", "exclusive deal", "wire transfer", "unsecured", "inheritance"
]

@app.post("/api/predict", response_model=EmailPrediction)
def predict_email(req: EmailRequest):
    model = get_pipeline()
    
    # Combine subject and body if provided separately
    full_text = req.text.strip()
    if not full_text:
        parts = []
        if req.subject.strip():
            parts.append(f"Subject: {req.subject.strip()}")
        if req.body.strip():
            parts.append(req.body.strip())
        full_text = "\n".join(parts)

    if not full_text:
        raise HTTPException(status_code=400, detail="Please provide email subject or body text to analyze.")

    # Model inference
    pred = int(model.predict([full_text])[0])
    proba = model.predict_proba([full_text])[0]
    
    ham_prob = float(proba[0])
    spam_prob = float(proba[1])
    confidence = spam_prob if pred == 1 else ham_prob

    # Find matching suspicious keywords for UI highlight
    lower_text = full_text.lower()
    detected = [kw for kw in SPAM_KEYWORDS if kw in lower_text]

    return EmailPrediction(
        prediction="spam" if pred == 1 else "ham",
        label_num=pred,
        confidence=round(confidence * 100, 2),
        is_spam=(pred == 1),
        probabilities={
            "ham": round(ham_prob * 100, 2),
            "spam": round(spam_prob * 100, 2)
        },
        word_count=len(full_text.split()),
        char_count=len(full_text),
        detected_keywords=detected
    )

@app.get("/api/metrics")
def get_metrics():
    data = load_metrics()
    if not data:
        return {"status": "Metrics not found. Run train.py to generate."}
    return data

@app.get("/api/samples")
def get_samples():
    return [
        {
            "id": 1,
            "title": "Business Project Sync",
            "type": "ham",
            "subject": "Sprint planning & pipeline review for Q4",
            "body": "Hi team,\n\nLet's connect this Thursday at 10:00 AM to review the Q4 deliverables and pipeline architectures. Please update your tickets beforehand.\n\nBest,\nAlex"
        },
        {
            "id": 2,
            "title": "Lottery Cash Prize Scam",
            "type": "spam",
            "subject": "CONGRATULATIONS! You have won $2,500,000 cash prize",
            "body": "Dear beneficiary,\n\nYour email address was selected in our international lottery promotion. Claim your free cash prize of $2,500,000 immediately by wiring bank transfer fees to our agent today! Act now, offer expires in 24 hours."
        },
        {
            "id": 3,
            "title": "Account Verification Phishing",
            "type": "spam",
            "subject": "URGENT: Your account access has been suspended",
            "body": "Dear customer,\n\nWe detected suspicious activity on your account. To prevent permanent suspension, please verify your password and identity immediately by clicking the secure link below. Risk free verification."
        },
        {
            "id": 4,
            "title": "Coffee Meetup Catch-up",
            "type": "ham",
            "subject": "Quick coffee catch up next week?",
            "body": "Hey Sarah,\n\nIt's been a while since we caught up. Are you free next Tuesday afternoon for a quick coffee near the downtown office? Let me know what time works best for you.\n\nCheers,\nDavid"
        }
    ]

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(STATIC_DIR, "style.css"))

@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(STATIC_DIR, "app.js"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=True)
