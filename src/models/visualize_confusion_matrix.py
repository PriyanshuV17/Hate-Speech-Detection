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
cm = confusion_matrix(y_test, y_pred)

# --------------------------
# 4. Plot and Save
# --------------------------
labels = ['Hate Speech', 'Offensive Language', 'Neither']
plot_confusion_matrix(
    cm,
    labels=labels,
    title='Confusion Matrix - Hate Speech Detection',
    save_path=models["confusion_matrix"]
)