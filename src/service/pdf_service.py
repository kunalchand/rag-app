# FILE: src/service/pdf_service.py
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
from src.config import settings


class PDFService:
    """
    PDFService handles text extraction and chunking from PDF documents.
    """

    def get_pdf_text(self, pdf_docs: List) -> str:
        """
        Extracts text from a list of uploaded PDF files.

        Args:
            pdf_docs (List): List of PDF file-like objects.

        Returns:
            str: Combined text from all PDFs.
        """
        text = ""
        for pdf_doc in pdf_docs:
            reader = PyPDF2.PdfReader(pdf_doc)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    def get_text_chunks(self, text: str) -> List[str]:
        """
        Splits text into chunks for embeddings.

        Args:
            text (str): Full text to split.

        Returns:
            List[str]: List of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        return splitter.split_text(text)
