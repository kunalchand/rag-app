import streamlit as st
import logging

from src.config.settings import settings
from src.core.logging_config import configure_logging
from src.ingestion.pdf_loader import extract_text_from_pdfs
from src.ingestion.chunking import chunk_text
from src.vectorstore.pinecone_store import PineconeStore
from src.rag.retriever import Retriever
from src.rag.pipeline import RAGPipeline
from src.llm.groq_client import build_groq_llm
from src.llm.prompts import conversation_prompt


def get_namespace():
    # Strategy: per session or static; you can extend to per-user
    if settings.namespace_strategy == "session":
        if "namespace" not in st.session_state:
            st.session_state["namespace"] = (
                f"ns_{st.session_state.get('RunId', '') or st.session_state.get('_session_id', '') or 'default'}"
            )
        return st.session_state["namespace"]
    return "default"


def main():
    configure_logging(level=settings.log_level)
    log = logging.getLogger(__name__)

    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖", layout="wide")
    st.title("PDF LangChain Chatbot")

    # Instantiate dependencies
    if not settings.groq_api_key or not settings.pinecone_api_key:
        st.error("Missing GROQ_API_KEY or PINECONE_API_KEY in environment.")
        return

    store = PineconeStore(
        api_key=settings.pinecone_api_key,
        environment=settings.pinecone_env,
        index_name=settings.pinecone_index,
        embedding_model=settings.embedding_model,
    )
    retriever = Retriever(store=store, top_k=settings.top_k)
    rag = RAGPipeline(store=store, retriever=retriever)

    llm = build_groq_llm(api_key=settings.groq_api_key, model=settings.llm_model)
    prompt = conversation_prompt()
    memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    conversation = ConversationChain(
        memory=memory, prompt=prompt, llm=llm, verbose=False
    )

    # Sidebar: Upload and index
    with st.sidebar:
        st.header("Submit Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF(s)", type="pdf", accept_multiple_files=True
        )
        ns = get_namespace()
        if st.button("Index Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing & indexing..."):
                    try:
                        raw_text = extract_text_from_pdfs(pdf_docs)
                        if not raw_text.strip():
                            st.warning("No extractable text found in PDFs.")
                        else:
                            chunks = chunk_text(
                                raw_text, settings.chunk_size, settings.chunk_overlap
                            )
                            count = rag.index_texts(chunks, namespace=ns)
                            st.success(f"Indexed {count} chunks to namespace '{ns}'.")
                            log.info(
                                "Indexed_chunks",
                                extra={"count": count, "namespace": ns},
                            )
                    except Exception as e:
                        log.exception("Indexing_failed")
                        st.error("Failed to index documents. See logs.")

        if st.button("Clear Namespace"):
            with st.spinner("Clearing..."):
                try:
                    store.clear_namespace(namespace=ns)
                    st.success(f"Cleared namespace '{ns}'.")
                except Exception:
                    log.exception("Namespace_clear_failed")
                    st.error("Failed to clear namespace.")

    # Chat UI
    user_q = st.text_input("Ask a question:")
    if user_q:
        with st.spinner("Thinking..."):
            try:
                ns = get_namespace()
                context = rag.build_context(user_q, namespace=ns)
                if not context:
                    st.info("I don't know. Try uploading relevant PDFs and re-ask.")
                else:
                    response = conversation.predict(
                        input=f"Context:\n{context}\n\nQuestion:\n{user_q}"
                    )
                    st.write(response)
            except Exception:
                log.exception("Chat_failed")
                st.error("Something went wrong answering your question. See logs.")


if __name__ == "__main__":
    main()
