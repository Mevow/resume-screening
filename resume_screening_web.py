import os
import fitz
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title(" Resume Screening with GenAI (Zero-Shot Classification)")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

JOB_ROLES = [
    "Data Scientist", "Software Engineer", "HR", "Accountant", "Sales",
    "Chef", "Advocate", "Designer", "Teacher", "Finance", "Construction",
    "Digital Media", "Engineer", "Healthcare", "Public Relations"
]

def extract_text_from_pdf(file):
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc]).strip().lower()
    except Exception as e:
        st.error(f" Error reading PDF: {e}")
        return ""

def classify_resume(text):
    try:
        result = classifier(text[:3000], candidate_labels=JOB_ROLES)
        return result["labels"][0], round(result["scores"][0], 2)
    except Exception as e:
        return "Unknown", 0.0

uploaded_files = st.file_uploader(" Upload PDF Resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        st.write(f" Processing: {uploaded_file.name}")
        text = extract_text_from_pdf(uploaded_file)
        if not text or len(text) < 50:
            st.warning(f" Skipped empty or short resume: {uploaded_file.name}")
            continue
        role, confidence = classify_resume(text)
        results.append({
            "filename": uploaded_file.name,
            "predicted_role": role,
            "confidence": confidence
        })
        progress.progress((idx + 1) / len(uploaded_files))

    df = pd.DataFrame(results)
    st.subheader(" Prediction Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Results as CSV", csv, "resume_screening_results.csv", "text/csv")
