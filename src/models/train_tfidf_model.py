# src/models/train_tfidf_model.py

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Define paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data/processed")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")

# Ensure model save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === Load data ===
print("📥 Loading preprocessed data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train_bal.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train_bal.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

print(f"✅ Data loaded successfully!")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# === Load vectorizer ===
VECTORIZER_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
vectorizer = joblib.load(VECTORIZER_PATH)
print("✅ TF-IDF vectorizer loaded successfully!")

# === Train model ===
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=500, n_jobs=-1)
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
    target_names=["Hate Speech", "Offensive Language", "Neither"]
))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Hate", "Offensive", "Neither"],
            yticklabels=["Hate", "Offensive", "Neither"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Hate Speech Detection - Confusion Matrix")
plt.tight_layout()

CM_PATH = os.path.join(MODEL_SAVE_DIR, "tfidf_confusion_matrix.png")
plt.savefig(CM_PATH)
plt.close()
print(f"🖼️ Confusion matrix saved to: {CM_PATH}")

# === Save trained model ===
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "tfidf_logreg_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")

print("\n✅ Training completed successfully!")
