# ==========================
# Hate Speech Detection: Predict Comments (Interactive or CSV)
# ==========================
import sys
import os
import joblib
import argparse
import pandas as pd

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import ConfigLoader
from src.utils.text_cleaning import clean_text
from src.utils.helper import map_label

# Load config and model
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()

vectorizer = joblib.load(processed["tfidf_vectorizer"])
model = joblib.load(models["baseline_model"])

# CLI arguments
parser = argparse.ArgumentParser(description="Predict hate speech label for comment(s)")
parser.add_argument("--csv", type=str, help="Path to CSV file with a 'comment' column")
args = parser.parse_args()

# Predict from CSV
if args.csv:
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if "comment" not in df.columns:
        print("CSV must contain a 'comment' column.")
        sys.exit(1)

    df["cleaned"] = df["comment"].apply(clean_text)
    X = vectorizer.transform(df["cleaned"])
    df["predicted_label"] = model.predict(X)
    df["label_name"] = df["predicted_label"].map(map_label)

    print("\nPredictions:")
    print(df[["comment", "label_name"]].head())

    output_path = os.path.join(models["dir"], "predicted_comments.csv")
    df.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to: {output_path}")

# Interactive keyboard input
else:
    print("Enter a comment to classify (press Ctrl+C to exit):")
    try:
        while True:
            raw_text = input("\nYour comment: ")
            cleaned = clean_text(raw_text)
            vectorized = vectorizer.transform([cleaned])
            predicted_label = model.predict(vectorized)[0]
            label_name = map_label(predicted_label)

            print(f"Predicted Label: {label_name}")
    except KeyboardInterrupt:
        print("\nExiting prediction tool.")