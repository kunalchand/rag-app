# FILE: src/app/ui.py
import streamlit as st
from streamlit_chat import message
from typing import Optional

from src.service.pdf_service import PDFService
from src.service.embeddings_service import EmbeddingsService
from src.service.pinecone_service import PineconeService
from src.service.chat_service import ChatService


def sidebar_ui(
    pdf_service: PDFService,
    embeddings_service: EmbeddingsService,
    pinecone_service: PineconeService,
) -> Optional[list]:
    """
    Renders the sidebar UI for PDF upload and processing.

    Args:
        pdf_service (PDFService): PDF processing service.
        embeddings_service (EmbeddingsService): Embedding service.
        pinecone_service (PineconeService): Pinecone vector service.

    Returns:
        Optional[list]: List of uploaded PDFs.
    """
    with st.sidebar:
        st.title("Submit Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF Documents", type="pdf", accept_multiple_files=True
        )
        if st.button("Send to Pinecone"):
            with st.spinner("Processing..."):
                raw_text = pdf_service.get_pdf_text(pdf_docs)
                chunks = pdf_service.get_text_chunks(raw_text)
                embeddings = embeddings_service.embed_texts(chunks)
                pinecone_service.delete_all_vectors()
                pinecone_service.upsert_vectors(chunks, embeddings)
                st.success("Documents processed and uploaded!")
    return pdf_docs


def chat_ui(
    chat_service: ChatService,
    embeddings_service: EmbeddingsService,
    pinecone_service: PineconeService,
) -> None:
    """
    Renders the chat UI and manages chat interactions.

    Args:
        chat_service (ChatService): Handles conversation and LLM queries.
        embeddings_service (EmbeddingsService): Embedding service.
        pinecone_service (PineconeService): Pinecone vector service.
    """
    st.markdown("# PDF Langchain Chatbot")
    response_container = st.container()
    text_container = st.container()

    with text_container:
        query = st.text_input("Your question:", key="input")
        if query:
            with st.spinner("Generating answer..."):
                conversation_str = chat_service.get_conversation_string()
                refined_query = chat_service.query_refiner(conversation_str, query)
                vector = embeddings_service.embed_query(refined_query)
                result = pinecone_service.query_vectors(vector)

                if len(result["matches"]) < 2:
                    context = "Sorry, not enough info in uploaded PDFs."
                else:
                    context = (
                        result["matches"][0]["metadata"]["text"]
                        + "\n"
                        + result["matches"][1]["metadata"]["text"]
                    )

                response = chat_service.conversation.predict(
                    input=f"Context:\n{context}\n\nQuery:\n{query}"
                )

                chat_service.add_interaction(query, response)

    with response_container:
        for i in range(len(chat_service.responses)):
            message(chat_service.responses[i], key=str(i))
            if i < len(chat_service.requests):
                message(chat_service.requests[i], is_user=True, key=f"{i}_user")
