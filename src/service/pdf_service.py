from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from configs import settings


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        reader = PyPDF2.PdfReader(pdf_doc)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )
    return splitter.split_text(text)
