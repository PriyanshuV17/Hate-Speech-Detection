# src/models/evaluate_model.py

import sys
import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import ConfigLoader

# --------------------------
# 1. Load Config
# --------------------------
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()
label_map = config.get_labels()
label_names = list(label_map.values())

# --------------------------
# 2. Load Data and Model
# --------------------------
print("📥 Loading test data and trained model...")
X_test = np.load(processed["test_features"])
y_test = np.load(processed["test_labels"])
model = joblib.load(models["baseline_model"])
print("✅ Data and model loaded successfully!")

# --------------------------
# 3. Predict and Evaluate
# --------------------------
print("\n🔮 Generating predictions...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy on test data: {accuracy:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_names,
    digits=4
)
print("\n📊 Classification Report:\n", report)

# --------------------------
# 4. Save Report
# --------------------------
report_path = os.path.join(models["dir"], "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"\n📝 Report saved to {report_path}")