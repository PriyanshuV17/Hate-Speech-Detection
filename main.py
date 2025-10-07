# ==========================
# Hate Speech Detection: Main Entry Point
# ==========================

import argparse
import subprocess
import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def run_stage(script_path, description, extra_args=None):
    """Helper to run a pipeline stage script."""
    abs_path = os.path.join(BASE_DIR, script_path)
    cmd = [sys.executable, abs_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n🚀 {description}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {description}: {e}")
        sys.exit(1)

def run_preprocessing():
    run_stage("src/preprocessing/preprocess_data.py", "Running preprocessing")

def run_training():
    run_stage("src/models/train_tfidf_model.py", "Running model training")

def run_evaluation():
    run_stage("src/models/evaluate_model.py", "Evaluating model")

def run_eda():
    print("\n📊 Opening EDA notebook... (run manually if required)")
    print("👉 Open: notebooks/01_data_exploration.ipynb in Jupyter or VSCode")

def run_prediction(csv_path=None):
    extra_args = ["--csv", csv_path] if csv_path else None
    run_stage("src/models/predict_comments.py", "Launching prediction interface", extra_args)

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["preprocess", "train", "eval", "eda", "predict", "all"],
        required=True,
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Optional CSV file path for prediction stage"
    )
    args = parser.parse_args()

    if args.stage == "preprocess":
        run_preprocessing()
    elif args.stage == "train":
        run_training()
    elif args.stage == "eval":
        run_evaluation()
    elif args.stage == "eda":
        run_eda()
    elif args.stage == "predict":
        run_prediction(args.csv)
    elif args.stage == "all":
        run_preprocessing()
        run_training()
        run_evaluation()

if __name__ == "__main__":
    main()