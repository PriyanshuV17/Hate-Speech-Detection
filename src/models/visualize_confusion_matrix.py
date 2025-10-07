import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths setup (adjust if running from elsewhere)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_path = os.path.join(project_root, 'data', 'processed')
model_path = os.path.join(project_root, 'saved_models', 'tfidf_logreg_model.pkl')

# Load data and model
X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))
model = joblib.load(model_path)

# Predict
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot and save confusion matrix
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hate Speech', 'Offensive Language', 'Neither'],
            yticklabels=['Hate Speech', 'Offensive Language', 'Neither'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Hate Speech Detection')
plt.tight_layout()

# Save to disk
output_path = os.path.join(project_root, 'saved_models', 'confusion_matrix.png')
plt.savefig(output_path)

print(f"Confusion matrix saved to {output_path}")
plt.show()
