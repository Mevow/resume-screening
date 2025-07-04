import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

resumes = [
    "Python, machine learning, data analysis, statistics",
    "Java, C++, software development, system design",
    "Recruitment, employee engagement, onboarding, payroll"
]
labels = [
    "Data Scientist",
    "Software Engineer",
    "HR"
]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes)

# Model training
model = RandomForestClassifier()
model.fit(X, labels)

# Create models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the vectorizer and model
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model and vectorizer saved in 'models/' folder.")
