# ==========================
# Hate Speech Detection: Main Entry Point
# ==========================

import argparse
import subprocess
import os

def run_preprocessing():
    print("Running preprocessing...")
    subprocess.run(["python", "src/preprocessing/preprocess_data.py"])

def run_training():
    print("Running model training...")
    subprocess.run(["python", "src/models/train_tfidf_model.py"])

def run_evaluation():
    print("Evaluating model...")
    subprocess.run(["python", "src/models/evaluate_model.py"])

def run_eda():
    print("Opening EDA notebook... (run manually if required)")
    print("Open: notebooks/01_data_exploration.ipynb in Jupyter or VSCode")

def run_prediction():
    print("Launching prediction interface...")
    subprocess.run(["python", "src/models/predict_comments.py"])

def main():
    parser = argparse.ArgumentParser(description="Hate Speech Detection Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["preprocess", "train", "eval", "eda", "predict"],
        help="Pipeline stage to run"
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
        run_prediction()
    else:
        print("Please specify a valid --stage: preprocess, train, eval, eda, predict")

if __name__ == "__main__":
    main()