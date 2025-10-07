# ==========================
# Hate Speech Detection: Data Preprocessing
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

from src.utils.text_cleaning import clean_text
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
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)

# --------------------------
# 3. Clean and Prepare
# --------------------------
df = df[['class', 'tweet']].copy()
df.rename(columns={'class': 'label', 'tweet': 'text'}, inplace=True)
df["clean_text"] = df["text"].apply(clean_text)
df["label_name"] = df["label"].map(map_label)

print("\n✅ Sample cleaned text:")
print(df["clean_text"].head(5))

print("\n✅ Label distribution before balancing:")
print(df["label_name"].value_counts())

# --------------------------
# 4. Train-Test Split
# --------------------------
X = df["clean_text"]
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
    stop_words=training["tfidf_stop_words"]
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

sm = SMOTE(random_state=training["smote_random_seed"])
X_train_bal, y_train_bal = sm.fit_resample(X_train_tfidf, y_train)

print("\n✅ Class distribution after SMOTE balancing:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

# --------------------------
# 6. Save Outputs
# --------------------------
os.makedirs(processed["dir"], exist_ok=True)
os.makedirs(models["dir"], exist_ok=True)

joblib.dump(tfidf, processed["tfidf_vectorizer"])
np.save(processed["train_features"], X_train_bal.toarray())
np.save(processed["train_labels"], y_train_bal)
np.save(processed["test_features"], X_test_tfidf.toarray())
np.save(processed["test_labels"], y_test)

df_to_save = df[['label', 'label_name', 'text', 'clean_text']]
df_to_save.to_excel(processed["cleaned_excel"], index=False)

# --------------------------
# 7. Final Check
# --------------------------
print("\n💾 Preprocessed data saved to:", processed["dir"])
print("📁 TF-IDF vectorizer saved to:", processed["tfidf_vectorizer"])
print("📝 Cleaned Excel saved to:", processed["cleaned_excel"])
print(f"\n✅ Preprocessing Complete! Training set: {X_train_bal.shape}, Test set: {X_test_tfidf.shape}")