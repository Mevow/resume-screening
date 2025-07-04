import os
import pandas as pd
from src.extract import extract_features_from_pdf, count_keywords
from src.predict import load_model_and_vectorizer, predict_role
from src.utils import keywords

def main():
    root_folder = "data"  # folder containing subfolders of resumes by category
    vectorizer, model = load_model_and_vectorizer()
    results = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                resume_text, pages, word_count = extract_features_from_pdf(pdf_path)

                if resume_text.strip() == "":
                    print(f" Skipped empty: {file}")
                    continue

                # Count keywords for stats
                ds_keywords = count_keywords(resume_text, keywords["data_scientist"])
                se_keywords = count_keywords(resume_text, keywords["software_engineer"])
                hr_keywords = count_keywords(resume_text, keywords["hr"])

                # Predict role
                role, confidence = predict_role(resume_text, vectorizer, model)

                results.append({
                    "filename": file,
                    "folder": os.path.relpath(root, root_folder),
                    "pages": pages,
                    "words": word_count,
                    "ds_keywords": ds_keywords,
                    "se_keywords": se_keywords,
                    "hr_keywords": hr_keywords,
                    "predicted_role": role,
                    "confidence": confidence
                })

    if not results:
        print("⚠ No resumes processed or all resumes were empty.")
        return

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values(by="confidence", ascending=False)
    df.to_csv("resume_screening_results.csv", index=False)

    # Save high-confidence results
    filtered_df = df[df['confidence'] > 0.9]
    filtered_df.to_csv("high_confidence_predictions.csv", index=False)

    # Print top results
    print("\n High-confidence predictions (> 0.9):\n")
    print(filtered_df)

if __name__ == "__main__":
    main()
