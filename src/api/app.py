# ==========================
# Hate Speech Detection: Gradio Web App
# ==========================
import os
import joblib
import gradio as gr
import sys

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config_loader import ConfigLoader
from src.utils.helper import map_label

# --------------------------
# 1. Load Config and Artifacts
# --------------------------
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()
label_map = config.get_labels()

vectorizer = joblib.load(processed["tfidf_vectorizer"])
model = joblib.load(models["baseline_model"])

# --------------------------
# 2. Prediction Function
# --------------------------
def predict_comment(comment: str):
    if not comment or not comment.strip():
        return "Please enter a non-empty comment.", None

    # Use raw text (no aggressive cleaning for multilingual support)
    X = vectorizer.transform([comment])
    pred = model.predict(X)[0]
    label = map_label(int(pred))

    # Confidence scores (if available)
    try:
        proba = model.predict_proba(X)[0]
        confidences = {map_label(i): float(p) for i, p in enumerate(proba)}
    except Exception:
        confidences = None

    return label, confidences

# --------------------------
# 3. Gradio Interface
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Hate Speech Comment Classification")
    gr.Markdown("Enter a comment to classify. Outputs the predicted label and confidence scores.")

    inp = gr.Textbox(label="Comment", lines=3, placeholder="Type a comment...")
    btn = gr.Button("Classify")
    out_label = gr.Textbox(label="Predicted Label")
    out_conf = gr.JSON(label="Confidence (per class)")

    btn.click(predict_comment, inputs=inp, outputs=[out_label, out_conf])

# --------------------------
# 4. Launch App
# --------------------------
if __name__ == "__main__":
    demo.launch()