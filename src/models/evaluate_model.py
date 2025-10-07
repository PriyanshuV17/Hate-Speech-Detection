import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Get project root directory (two levels up from this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

data_path = os.path.join(project_root, 'data', 'processed')

X_test = np.load(os.path.join(data_path, 'X_test.npy'))
y_test = np.load(os.path.join(data_path, 'y_test.npy'))

model_path = os.path.join(project_root, 'saved_models', 'tfidf_logreg_model.pkl')
model = joblib.load(model_path)

y_pred = model.predict(X_test)

print(f"Accuracy on test data: {accuracy_score(y_test, y_pred):.4f}")

report = classification_report(
    y_test, y_pred,
    target_names=['Hate Speech', 'Offensive Language', 'Neither']
)
print("\nClassification Report:\n", report)
