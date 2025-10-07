# ==========================
# Hate Speech Detection: Local Gradio App
# Team: CodeBros
# ==========================
import os
import sys
import joblib
import gradio as gr
import pandas as pd
import xml.etree.ElementTree as ET

# ✅ Patch sys.path to allow absolute imports from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import ConfigLoader
from src.utils.helper import map_label

# --------------------------
# 1. Load Config and Artifacts
# --------------------------
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()

vectorizer = joblib.load(processed["tfidf_vectorizer"])
model = joblib.load(models["baseline_model"])

# --------------------------
# 2. Prediction Logic
# --------------------------
def classify_text(text: str):
    if not text or not text.strip():
        return "Please enter a non-empty comment.", None

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = map_label(int(pred))

    try:
        proba = model.predict_proba(X)[0]
        confidences = {map_label(i): round(float(p), 4) for i, p in enumerate(proba)}
    except Exception:
        confidences = None

    return label, confidences

def classify_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
        elif ext == ".txt":
            df = pd.read_csv(file, delimiter="\n", header=None, names=["text"])
        elif ext == ".xml":
            tree = ET.parse(file)
            root = tree.getroot()
            texts = [elem.text for elem in root.iter() if elem.text]
            df = pd.DataFrame(texts, columns=["text"])
        else:
            return "Unsupported file format", None

        df["prediction"] = df["text"].apply(lambda x: map_label(int(model.predict(vectorizer.transform([str(x)]))[0])))
        return df
    except Exception as e:
        return f"Error processing file: {e}", None

# --------------------------
# 3. Gradio Interface
# --------------------------
demo = gr.Blocks()

with demo:
    gr.Markdown("## 🌐 Hate Speech Detection (Multilingual)")
    gr.Markdown("**Team: CodeBros**")
    gr.Markdown("""
This tool classifies comments into 10 hate speech categories using a multilingual model trained on 11 languages.

**Supported Languages:** English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Urdu, Nepali  
**Classes:** bullying, non_bullying, offensive, racist, sexist, abusive, clean, no_hate, harassing, personal_attacks

You can classify:
- A single comment
- A batch of comments via CSV, Excel, TXT, or XML file (must contain a column or line of text)
""")

    with gr.Tab("Single Comment"):
        inp = gr.Textbox(label="Comment", lines=3, placeholder="Type a comment...")
        btn = gr.Button("Classify")
        out_label = gr.Textbox(label="Predicted Label")
        out_conf = gr.JSON(label="Confidence (per class)")
        btn.click(classify_text, inputs=inp, outputs=[out_label, out_conf])

    with gr.Tab("Batch File Upload"):
        file_input = gr.File(label="Upload File (.csv, .xlsx, .txt, .xml)")
        file_output = gr.Dataframe(label="Predictions", wrap=True)
        file_input.change(classify_file, inputs=file_input, outputs=file_output)

# --------------------------
# 4. Launch App
# --------------------------
if __name__ == "__main__":
    demo.launch()