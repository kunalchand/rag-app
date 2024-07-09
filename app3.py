import PyPDF2
import streamlit as st
from groq import Groq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone
from streamlit_chat import message


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    embedded = [hf.embed_query(chunk) for chunk in text_chunks]
    return embedded


def create_vectors_in_index_in_pinecone(text_chunks, embeddings, pinecone_index):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        # print(len(embedding))
        vectors.append({"id": str(i), "values": embedding, "metadata": {"text": chunk}})
    pinecone_index.upsert(vectors)


def get_similar_vector_from_pinecone(user_question, pinecone_index):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    vector_embedding = hf.embed_query(user_question)
    return pinecone_index.query(
        vector=vector_embedding,
        top_k=2,
        # include_values=True,
        include_metadata=True,
    )


def delete_all_vectors_in_pinecone(pinecone_index):
    if len([ids for ids in pinecone_index.list(namespace="")]) != 0:
        pinecone_index.delete(delete_all=True, namespace="")


def find_match(user_question, pinecone_index):
    result = get_similar_vector_from_pinecone(user_question, pinecone_index)
    if len(result["matches"]) < 2:
        return "Sorry, I could not answer this question. The PDFs you provided did not contain enough information related to the question. Please upload relevant PDFs for me to understand the context."
    else:
        return (
            result["matches"][0]["metadata"]["text"]
            + "\n"
            + result["matches"][1]["metadata"]["text"]
        )


def query_refiner(conversation, query, groq):
    prompt = f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"
    response = groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += "Human: " + st.session_state["requests"][i] + "\n"
        conversation_string += "Bot: " + st.session_state["responses"][i + 1] + "\n"
    return conversation_string


def initial_setup(chatGroq):
    if "responses" not in st.session_state:
        st.session_state["responses"] = ["How can I assist you?"]

    if "requests" not in st.session_state:
        st.session_state["requests"] = []

    if "buffer_memory" not in st.session_state:
        st.session_state["buffer_memory"] = ConversationBufferWindowMemory(
            k=3,
            return_messages=True,
        )

    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know' and suggest user to upload relevant pdf documents."""
    )

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template,
        ]
    )

    conversation = ConversationChain(
        memory=st.session_state.buffer_memory,
        prompt=prompt_template,
        llm=chatGroq,
        verbose=True,
    )

    return conversation


def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")

    # API Token Key Setup
    groq = Groq(
        api_key="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",
    )

    chatGroq = ChatGroq(
        model="llama3-70b-8192",
        api_key="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",
    )

    pc = Pinecone(
        api_key="bfdb67c3-5557-41c4-b0aa-afa02fad01d2",
        environment="us-east-1",
    )

    pinecone_index = pc.Index("rag-app")

    conversation = initial_setup(chatGroq)

    # Upload PDF Documents
    with st.sidebar:
        st.title("Submit Documents Here")
        pdf_docs = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_upChatGroqloader",
        )
        if st.button("Send to Pinecone", key="process_button"):
            with st.spinner("Processing..."):
                delete_all_vectors_in_pinecone(pinecone_index)

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                chunked_document_embeddings = get_embeddings(text_chunks)
                create_vectors_in_index_in_pinecone(
                    text_chunks, chunked_document_embeddings, pinecone_index
                )
                st.success("Done!")

    # Chatbot
    st.markdown("# PDF Langchain Chatbot")

    response_container = st.container()  # container for chat history
    textcontainer = st.container()  # container for text box

    with textcontainer:
        query = st.text_input("Question: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                # st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query, groq)
                # st.subheader("Refined Question:")
                # st.write(refined_query)
                context = find_match(refined_query, pinecone_index)
                # print(context)
                response = conversation.predict(
                    input=f"Context:\n {context} \n\n Query:\n{query}"
                )
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state["responses"]:
            for i in range(len(st.session_state["responses"])):
                message(st.session_state["responses"][i], key=str(i))
                if i < len(st.session_state["requests"]):
                    message(
                        st.session_state["requests"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )


if __name__ == "__main__":
    main()
