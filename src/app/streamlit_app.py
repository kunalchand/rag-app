import streamlit as st
from streamlit_chat import message

from service.pdf_service import get_pdf_text, get_text_chunks
from service.embeddings_service import EmbeddingsService
from service.pinecone_service import PineconeService
from service.chat_service import ChatService
from config import settings


def main():
    st.set_page_config(page_title=settings.APP_TITLE, page_icon=settings.APP_ICON)

    # Initialize services
    pdf_service = (get_pdf_text, get_text_chunks)
    embeddings_service = EmbeddingsService()
    pinecone_service = PineconeService()
    chat_service = ChatService()

    # Sidebar: PDF upload
    with st.sidebar:
        st.title("Submit Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF Documents", type="pdf", accept_multiple_files=True
        )
        if st.button("Send to Pinecone"):
            with st.spinner("Processing..."):
                pinecone_service.delete_all_vectors()
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                embeddings = embeddings_service.embed_texts(chunks)
                pinecone_service.upsert_vectors(chunks, embeddings)
                st.success("Done!")

    # Chat interface
    st.markdown("# PDF Langchain Chatbot")
    response_container = st.container()
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Question: ", key="input")
        if query:
            with st.spinner("typing..."):
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
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

    # Display chat history
    with response_container:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(st.session_state["requests"][i], is_user=True, key=f"{i}_user")


if __name__ == "__main__":
    main()
