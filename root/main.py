from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

# Load model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("tfidf_logreg_model.pkl")

LABEL_MAP = {
    0: "bullying", 1: "non_bullying", 2: "offensive", 3: "racist", 4: "sexist",
    5: "abusive", 6: "clean", 7: "no_hate", 8: "harassing", 9: "personal_attacks"
}

@app.get("/")
def home():
    return {"message": "CodeBros Hate Speech Detection API is live!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text.strip():
        return JSONResponse(content={"error": "Empty input"}, status_code=400)

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = LABEL_MAP.get(int(pred), str(pred))
    proba = model.predict_proba(X)[0]
    confidences = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {"label": label, "confidence": confidences}