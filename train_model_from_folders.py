import os
import pickle
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.lower()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

def load_data_from_folders(data_root="data"):
    texts = []
    labels = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith(".pdf"):
                label = os.path.basename(root)
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                if text.strip():
                    texts.append(text)
                    labels.append(label)
                else:
                    print(f"⚠️ Skipped empty: {file}")
    return texts, labels

def main():
    print("🔍 Loading data from folders...")
    texts, labels = load_data_from_folders("data")
    print(f"📦 Loaded {len(texts)} resumes across {len(set(labels))} categories")

    if not texts:
        print("❌ No usable resume data found.")
        return

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)

    # Evaluate (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model and vectorizer saved in 'models/'")

if __name__ == "__main__":
    main()
