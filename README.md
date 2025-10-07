==**HATE SPEECH DETECTION PROJECT**==


Project Overview
----------------
This project focuses on detecting hate speech and offensive content in text data using Natural Language Processing (NLP) and Machine Learning models.  
It classifies input text into categories such as hate, offensive, bullying, clean, etc.  

The system is designed for multilingual support and can be extended for detailed sub-categorization of hate types in future work.

-------------------------------------------------------------
Live Deployments
-------------------------------------------------------------
🔹 Web App (Deployed on Hugging Face Spaces):  
   https://huggingface.co/spaces/PriyanshuV17/Hate_Speech_Classifier  

🔹 API Endpoint (Deployed on Render):  
   https://hate-speech-detection-1uqd.onrender.com  

-------------------------------------------------------------
Features
-------------------------------------------------------------
- Detects and classifies hate speech or offensive language in comments, tweets, or text data.
- Supports multiple languages (future versions planned for multilingual expansion).
- Clean modular architecture for easy training, fine-tuning, and deployment.
- REST API support for integration with other systems.
- Deployed live as a web-based interface (Gradio on Hugging Face).

-------------------------------------------------------------
Repository Structure
-------------------------------------------------------------
data/
  |—— labeled_data.csv -> Cleaned & labeled_dataset
  |—— processed/ -> Preprocessed training/test data

notebooks/
  |—— 01_data_exploration.ipynb
  |—— 02_preprocessing.ipynb
  |—— 03_model_training.ipynb
  |—— 04_evaluation.ipynb

src/
  |—— preprocessing/ -> Text cleaning and preprocessing scripts
  |—— models/ -> Training and evaluation scripts
  |—— api/ -> Backend API scripts (Flask / FastAPI)
  |—— utils/ -> Helper and utility functions
  |—— init.py

saved_models/
  |—— tfidf_logreg_model.pki
  |—— tfidf_vectorizer.pki
  |—— bert_model/ (optional for future transformer models)

config.yaml -> Configuration file
requirements.txt -> Required dependencies
main.py -> Main entry script
.gitignore -> ignore unnecessary files

-------------------------------------------------------------
Using the Deployed API
-------------------------------------------------------------
Endpoint: https://hate-speech-detection-1uqd.onrender.com/predict  
Method: POST  
Content-Type: application/json  

Example:
{
    "text": "I hate this!"
}

Response Example:
{
    "prediction": "Hate Speech",
    "confidence": 0.94
}

-------------------------------------------------------------
Evaluation Metrics
-------------------------------------------------------------
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC/AUC (optional)

-------------------------------------------------------------
Future Enhancements
-------------------------------------------------------------
- Multilingual hate speech detection (Italian, Spanish, Punjabi, Tamil, Telugu)
- Transformer-based fine-tuning (BERT / RoBERTa)
- Live data analysis from social media feeds
- Advanced dashboard visualization
- Integration with moderation tools and APIs

-------------------------------------------------------------
Tech Stack
-------------------------------------------------------------
- Python
- Scikit-learn
- Pandas, NumPy
- Flask / FastAPI
- Gradio
- Hugging Face Spaces
- Render (API Hosting)

-------------------------------------------------------------
Author
-------------------------------------------------------------
Developed by: ** Codebros **  
GitHub: https://github.com/PriyanshuV17/Hate-Speech-Detection  
Hugging Face App: https://huggingface.co/spaces/PriyanshuV17/Hate_Speech_Classifier  
API: https://hate-speech-detection-1uqd.onrender.com  

For contributions, issues, or suggestions, please raise an issue or pull request on GitHub.

-------------------------------------------------------------
License
-------------------------------------------------------------
This project is open-source under the MIT License.  
You are free to use, modify, and distribute it with attribution.

=============================================================
