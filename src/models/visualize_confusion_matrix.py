# src/models/visualize_confusion_matrix.py

import sys
import os
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import ConfigLoader
from src.utils.plotting import plot_confusion_matrix

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
cm = confusion_matrix(y_test, y_pred)

# --------------------------
# 4. Plot and Save
# --------------------------
print("\n📊 Plotting confusion matrix...")
plot_confusion_matrix(
    cm,
    labels=label_names,
    title="Confusion Matrix - Hate Speech Detection",
    save_path=models["confusion_matrix"]
)

print("\n✅ Confusion matrix visualization complete!")