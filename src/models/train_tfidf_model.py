# src/models/train_tfidf_model.py

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.config_loader import ConfigLoader
from src.utils.plotting import plot_confusion_matrix

# === Load config ===
config = ConfigLoader()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()
training = config.get_training_params()
label_map = config.get_labels()
label_names = list(label_map.values())

# Ensure model save directory exists
models_dir = os.path.dirname(models["baseline_model"])
os.makedirs(models_dir, exist_ok=True)

# === Load data ===
print("📥 Loading preprocessed data...")
X_train = np.load(processed["train_features"])
y_train = np.load(processed["train_labels"])
X_test = np.load(processed["test_features"])
y_test = np.load(processed["test_labels"])

print("✅ Data loaded successfully!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# === Load vectorizer ===
vectorizer = joblib.load(processed["tfidf_vectorizer"])
print("✅ TF-IDF vectorizer loaded successfully!")

# === Train model ===
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=training["logistic_regression"]["max_iter"],
    class_weight=training["logistic_regression"]["class_weight"],
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model training completed successfully!")

# === Evaluate model ===
print("\n📊 Evaluating model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}\n")

print("📄 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_names
))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm,
    labels=label_names,
    title="Hate Speech Detection - Confusion Matrix",
    save_path=models["confusion_matrix"]
)

# === Save trained model ===
joblib.dump(model, models["baseline_model"])
print(f"💾 Model saved to: {models['baseline_model']}")

print("\n✅ Training completed successfully!")