"""
pdf_utils.py
Handles extracting raw text from uploaded PDF files (job descriptions or resumes).
"""

from pypdf import PdfReader
import io


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Takes a Streamlit UploadedFile object (PDF) and returns extracted text.
    Works directly from memory — no need to save the file to disk first.
    """
    pdf_bytes = uploaded_file.read()
    pdf_stream = io.BytesIO(pdf_bytes)

    reader = PdfReader(pdf_stream)

    extracted_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text.strip()