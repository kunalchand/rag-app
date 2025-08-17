from typing import List, BinaryIO
import PyPDF2


def extract_text_from_pdfs(files: List[BinaryIO]) -> str:
    text = []
    for f in files:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text.append(extracted)
    return "\n".join(text)
