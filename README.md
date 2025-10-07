folder structure


hate_speech_detection/
│
├── data/
│   ├── labeled_data.csv                 # Original dataset
│   ├── processed/                       # Processed data after cleaning
│   │   ├── X_train_bal.npy
│   │   ├── y_train_bal.npy
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA and visualization
│   ├── 02_preprocessing.ipynb           # Preprocessing & cleaning
│   ├── 03_model_training.ipynb          # Training traditional ML models
│   ├── 04_transformer_finetuning.ipynb  # Fine-tuning BERT/RoBERTa (future)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocess_data.py           # Preprocessing script (you already have)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_tfidf_model.py         # Logistic Regression / SVM
│   │   ├── train_bert_model.py          # Transformer fine-tuning (future)
│   │   └── evaluate_model.py            # Evaluation metrics
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                       # Flask/FastAPI deployment
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_cleaning.py             # All text cleaning functions
│   │   └── helper.py                    # Reusable functions
│
├── saved_models/
│   ├── tfidf_logreg_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_model/                      # Folder for BERT fine-tuned model
│
├── requirements.txt                     # All dependencies
├── config.yaml                          # Configuration for paths, params
├── main.py                              # Entry point (for training or testing)
├── README.md                            # Project documentation
└── .gitignore
