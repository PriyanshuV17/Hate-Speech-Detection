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

# --------------------------
# 2. Load Data and Model
# --------------------------
X_test = np.load(processed["test_features"])
y_test = np.load(processed["test_labels"])
model = joblib.load(models["baseline_model"])

# --------------------------
# 3. Predict and Evaluate
# --------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test data: {accuracy:.4f}")

report = classification_report(
    y_test, y_pred,
    target_names=['Hate Speech', 'Offensive Language', 'Neither']
)
print("\n📊 Classification Report:\n", report)

report_path = os.path.join(models["dir"], "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)
print(f"\n📝 Report saved to {report_path}")