from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2

from src.config import settings


class PDFService:
    def __init__(self):
        """
        PDFService Initialization
        """
        pass

    def get_pdf_text(self, pdf_docs):
        text = ""
        for pdf_doc in pdf_docs:
            reader = PyPDF2.PdfReader(pdf_doc)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def get_text_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        return splitter.split_text(text)
