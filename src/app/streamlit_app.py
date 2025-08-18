# FILE: src/app/streamlit_app.py
import streamlit as st
from typing import Optional

from src.config import settings
from src.service.pdf_service import PDFService
from src.service.embeddings_service import EmbeddingsService
from src.service.pinecone_service import PineconeService
from src.service.chat_service import ChatService
from src.app import ui


def main() -> None:
    """
    Entry point for the Streamlit app.
    Sets page configuration, initializes services, and renders the UI.
    """
    # Streamlit page configuration
    st.set_page_config(
        page_title=settings.APP_TITLE, page_icon=settings.APP_ICON, layout="wide"
    )

    # Initialize services
    pdf_service = PDFService()
    embeddings_service = EmbeddingsService()
    pinecone_service = PineconeService()
    chat_service = ChatService()

    # Render sidebar UI for PDF upload
    uploaded_pdfs: Optional[list] = ui.sidebar_ui(
        pdf_service=pdf_service,
        embeddings_service=embeddings_service,
        pinecone_service=pinecone_service,
    )

    # Render chat interface
    ui.chat_ui(
        chat_service=chat_service,
        embeddings_service=embeddings_service,
        pinecone_service=pinecone_service,
    )


if __name__ == "__main__":
    main()
