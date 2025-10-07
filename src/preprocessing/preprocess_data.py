# ==========================
# Hate Speech Detection: Data Preprocessing
# ==========================

import pandas as pd
import numpy as np
import re
import string
import emoji
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --------------------------
# 1. Load Dataset
# --------------------------
df = pd.read_csv("labeled_data.csv")

# Display initial info
print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# --------------------------
# 2. Keep only relevant columns
# --------------------------
df = df[['class', 'tweet']].copy()

# Rename columns for clarity
df.rename(columns={'class': 'label', 'tweet': 'text'}, inplace=True)

# --------------------------
# 3. Text Cleaning Function
# --------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)                      # remove mentions
    text = re.sub(r"#\w+", "", text)                      # remove hashtags
    text = emoji.replace_emoji(text, replace='')           # remove emojis
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = re.sub(r"\s+", " ", text).strip()              # remove extra spaces
    return text

# Apply cleaning
df["clean_text"] = df["text"].apply(clean_text)

print("\n✅ Sample cleaned text:")
print(df["clean_text"].head(5))

# --------------------------
# 4. Encode Class Labels
# --------------------------
# Mapping already provided: 0=hate_speech, 1=offensive_language, 2=neither
label_map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
df["label_name"] = df["label"].map(label_map)

print("\n✅ Label distribution before balancing:")
print(df["label_name"].value_counts())

# --------------------------
# 5. Handle Imbalanced Data
# --------------------------
X = df["clean_text"]
y = df["label"]

# Split before balancing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Apply SMOTE to balance training data
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_tfidf, y_train)

print("\n✅ Class distribution after SMOTE balancing:")
unique, counts = np.unique(y_train_bal, return_counts=True)
print(dict(zip(unique, counts)))

# --------------------------
# 6. Save Processed Data and TF-IDF Model
# --------------------------
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
np.save("X_train_bal.npy", X_train_bal.toarray())
np.save("y_train_bal.npy", y_train_bal)
np.save("X_test.npy", X_test_tfidf.toarray())
np.save("y_test.npy", y_test)

print("\n💾 Preprocessed data and vectorizer saved successfully!")

# --------------------------
# 7. Quick sanity check
# --------------------------
print("\n✅ Preprocessing Complete!")
print(f"Training set: {X_train_bal.shape}, Test set: {X_test_tfidf.shape}")

# --------------------------
# 8. Save Cleaned Data to Excel
# --------------------------
df_to_save = df[['label', 'label_name', 'text', 'clean_text']]
df_to_save.to_excel("labeled_data_1.xlsx", index=False)

print("\n📁 Cleaned dataset saved to labeled_data_1.xlsx!")
