# Resume Screening Assistant

A smart and efficient resume classification tool built using Python and Machine Learning. It automatically analyzes PDF resumes and predicts the most suitable job category such as **Data Scientist**, **Software Engineer**, **HR**, and more.

---

##  Features

Parses and reads PDF resumes
- Cleans and vectorizes resume content using **TF-IDF**
- Classifies resumes using a trained **Random Forest** model
- Integrates **Transformer-based language models** for deeper semantic understanding
- Generates **confidence scores** for predictions
- Web interface built with **Streamlit** for interactive resume upload and filtering
- Exports predictions as CSV
- Supports filtering resumes with high confidence (>0.7)


Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

## How to Use
1. Clone the repository:
```bash
git clone https://github.com/Mevow/resume-screening.git
cd resume-screening


Install dependencies: pip install -r requirements.txt

Train the model (if needed): python train_model_from_folders.py
Run resume screening: python resume_screening.py

Folder Structure
data/ – Contains categorized PDF resumes
models/ – Stores the trained model and vectorizer
src/ – Python files for feature extraction and prediction
resume_screening.py – Main script for screening resumes
resume_screening_web.py- Run the Streamlit app streamlit run app.py

Output
resume_screening_results.csv – All predictions
high_confidence_predictions.csv – Only predictions with high confidence
