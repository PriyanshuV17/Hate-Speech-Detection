from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import joblib
import os

# Initialize FastAPI with Swagger metadata
app = FastAPI(
    title="CodeBros Hate Speech Detection API",
    description="Multilingual hate speech classifier with 10 categories. Supports single comment prediction.",
    version="1.0.0"
)

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("tfidf_logreg_model.pkl")

# Label mapping
LABEL_MAP = {
    0: "bullying", 1: "non_bullying", 2: "offensive", 3: "racist", 4: "sexist",
    5: "abusive", 6: "clean", 7: "no_hate", 8: "harassing", 9: "personal_attacks"
}

# API key from environment variable
API_KEY = os.getenv("API_KEY")

@app.get("/")
def home():
    return {"message": "CodeBros Hate Speech Detection API is live!"}

@app.post("/predict")
async def predict(request: Request):
    # Check API key
    key = request.headers.get("x-api-key")
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # Parse input
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return JSONResponse(content={"error": "Empty input"}, status_code=400)

    # Predict
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = LABEL_MAP.get(int(pred), str(pred))
    proba = model.predict_proba(X)[0]
    confidences = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {"label": label, "confidence": confidences}
