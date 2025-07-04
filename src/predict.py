import pickle

def load_model_and_vectorizer():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

def predict_role(text, vectorizer, model):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()
    return prediction, round(confidence, 2)
