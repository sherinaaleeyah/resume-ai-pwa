import fitz  # PyMuPDF

def extract_resume_text(file_path):
    text = ""
    doc = fitz.open(file_path)

    for page in doc:
        text += page.get_text()
