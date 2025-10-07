import os
import joblib
import gradio as gr

# Adjust paths if needed
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "tfidf_logreg_model.pkl"

# If you have your own cleaner, import it:
# from src.utils.text_cleaning import clean_text
# For demo, define a simple cleaner:
def clean_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

# Load artifacts
vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

# Label mapping (update to match your project)
IDX_TO_LABEL = {
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
}

def predict_comment(comment: str):
    if not comment or not comment.strip():
        return "Please enter a non-empty comment.", None
    cleaned = clean_text(comment)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    label = IDX_TO_LABEL.get(int(pred), str(pred))
    # If available, add confidence
    try:
        proba = model.predict_proba(X)[0]
        confidences = {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(proba)}
    except Exception:
        confidences = None
    return label, confidences

with gr.Blocks() as demo:
    gr.Markdown("# Hate Speech Comment Classification")
    gr.Markdown("Enter a comment to classify. Outputs the predicted label and confidence.")

    inp = gr.Textbox(label="Comment", lines=3, placeholder="Type a comment...")
    btn = gr.Button("Classify")
    out_label = gr.Textbox(label="Predicted label")
    out_conf = gr.JSON(label="Confidence (per class)")

    btn.click(predict_comment, inputs=inp, outputs=[out_label, out_conf])

if __name__ == "__main__":
    demo.launch()