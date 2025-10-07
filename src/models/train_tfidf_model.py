# ==========================
# Hate Speech Detection: Train TF-IDF + Logistic Regression
# ==========================
import sys
import os
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 🔧 Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import ConfigLoader

# --------------------------
# 1. Load Config
# --------------------------
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()
training = config.get_training_params()

# --------------------------
# 2. Load Preprocessed Data
# --------------------------
print("📥 Loading preprocessed data...")

X_train = np.load(processed["train_features"])
y_train = np.load(processed["train_labels"])
X_test = np.load(processed["test_features"])
y_test = np.load(processed["test_labels"])

print("✅ Data loaded successfully!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --------------------------
# 3. Load TF-IDF Vectorizer
# --------------------------
tfidf = joblib.load(processed["tfidf_vectorizer"])
print("✅ TF-IDF vectorizer loaded successfully!")

# --------------------------
# 4. Train Model
# --------------------------
print("\n🚀 Training Logistic Regression model...")

logreg_config = training["logistic_regression"]
clf = LogisticRegression(
    max_iter=logreg_config["max_iter"],
    C=logreg_config["C"],
    solver=logreg_config["solver"],
    class_weight=logreg_config["class_weight"]
)
clf.fit(X_train, y_train)

print("✅ Model training completed successfully!")

# --------------------------
# 5. Evaluate Model
# --------------------------
print("\n📊 Evaluating model performance...")

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)
print(f"\n🎯 Accuracy: {acc:.4f}")
print("\n📄 Classification Report:")
print(report)

# --------------------------
# 6. Save Outputs
# --------------------------
models_dir = os.path.dirname(models["baseline_model"])
os.makedirs(models_dir, exist_ok=True)

# Save model
joblib.dump(clf, models["baseline_model"])

# Save metrics report
metrics = {
    "accuracy": acc,
    "classification_report": classification_report(y_test, y_pred, output_dict=True)
}
with open(models["metrics_report"], "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig(models["confusion_matrix"])
plt.close()

print("\n💾 Model saved to:", models["baseline_model"])
print("📊 Metrics saved to:", models["metrics_report"])
print("📉 Confusion matrix saved to:", models["confusion_matrix"])
print("\n✅ Training complete!")