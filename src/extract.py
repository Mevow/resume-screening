import fitz 

def extract_features_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        num_pages = len(doc)
        num_words = len(text.split())
        doc.close() 
        return text.lower(), num_pages, num_words
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return "", 0, 0

def count_keywords(text, keyword_list):
    return sum(keyword in text for keyword in keyword_list)
