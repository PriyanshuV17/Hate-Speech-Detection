# ==========================
# Hate Speech Detection: Data Preprocessing (Multilingual, No Cleaning)
# ==========================
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Ensure root path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.helper import map_label
from src.utils.config_loader import ConfigLoader

# --------------------------
# 1. Load Config
# --------------------------
config = ConfigLoader()
paths = config.get_paths()
processed = config.get_processed_data_paths()
models = config.get_saved_model_paths()
training = config.get_training_params()

# --------------------------
# 2. Load Dataset
# --------------------------
df = pd.read_csv(paths["raw_data"])
print("📥 Dataset loaded successfully!")
print("Shape:", df.shape)

# --------------------------
# 3. Prepare Data (no cleaning)
# --------------------------
# Keep only relevant columns
df = df[['label_id', 'label_name', 'comment', 'language']].copy()
df.rename(columns={'comment': 'text', 'label_id': 'label'}, inplace=True)

# Ensure label_name is consistent with mapping
df["label_name"] = df["label"].map(map_label)

print("\n🔎 Sample text:")
print(df["text"].head(5))

print("\n📊 Label distribution before balancing:")
print(df["label_name"].value_counts())

# --------------------------
# 4. Train-Test Split
# --------------------------
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=training["test_size"],
    random_state=training["random_seed"],
    stratify=y
)

# --------------------------
# 5. TF-IDF + SMOTE
# --------------------------
tfidf = TfidfVectorizer(
    max_features=training["tfidf_max_features"],
    stop_words=None   # multilingual → don’t use English-only stopwords
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

sm = SMOTE(random_state=training["smote_random_seed"])
X_train_bal, y_train_bal = sm.fit_resample(X_train_tfidf, y_train)

print("\n📊 Class distribution after SMOTE balancing:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

# --------------------------
# 6. Save Outputs
# --------------------------
# Derive directories from file paths
processed_dir = os.path.dirname(processed["train_features"])
models_dir = os.path.dirname(models["baseline_model"])

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Save vectorizer and arrays
joblib.dump(tfidf, processed["tfidf_vectorizer"])
np.save(processed["train_features"], X_train_bal.toarray())
np.save(processed["train_labels"], y_train_bal)
np.save(processed["test_features"], X_test_tfidf.toarray())
np.save(processed["test_labels"], y_test)

# Save cleaned dataset for inspection
df_to_save = df[['label', 'label_name', 'text', 'language']]
df_to_save.to_excel(processed["cleaned_excel"], index=False)

# --------------------------
# 7. Final Check
# --------------------------
print("\n💾 Preprocessed data saved to:", processed_dir)
print("💾 TF-IDF vectorizer saved to:", processed["tfidf_vectorizer"])
print("💾 Cleaned Excel saved to:", processed["cleaned_excel"])
print(f"\n✅ Preprocessing Complete! Training set: {X_train_bal.shape}, Test set: {X_test_tfidf.shape}")