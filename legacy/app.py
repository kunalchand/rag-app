# Replicate API Token: ****

# pip install replicate
# export REPLICATE_API_TOKEN=****

import os

import PyPDF2
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import LlamaForCausalLM

# Set the REPLICATE_API_TOKEN environment variable programmatically
# os.environ["REPLICATE_API_TOKEN"] = "*****"

# import replicate

# # The meta/llama-2-70b-chat model can stream output as it's running.
# for event in replicate.stream(
#     "meta/llama-2-70b-chat",
#     input={"prompt": "Write a short poem"},
# ):
#     print(str(event), end="")


st.set_page_config(page_title="PDF Reader", page_icon="📄", layout="wide")

st.markdown("# PDF Reader")

api_key = st.text_input(
    "Enter your Replicate API Token", type="password", key="api_key_input"
)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    embeddings = LlamaCppEmbeddings(model="meta/llama-2-70b-chat")  # , api_key=api_key)
    vectordb = Chroma.from_documents(text_chunks, embeddings)
    vectordb.persist()
    return vectordb


def get_conversational_chain():
    prompt_template = """ Answer the question as detailed a spossible from the provided contenxt, make sre to provide all the details. If the answer is not available in the documents, don't provide wrong answer.
    context: \n{context}\n
    Question: {question}\n
    
    Answer:"""

    model = LlamaForCausalLM.from_pretrained(
        model="meta/llama-2-70b-chat", temperature=0.5, api_key=api_key
    )

    prompt = prompt_template(
        template=prompt_template, input_variable=["context", "question"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    embeddings = LlamaCppEmbeddings(model="meta/llama-2-70b-chat")  # , api_key=api_key)
    new_db = Chroma.from_documents([], embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )
    st.write("Reply: " + response["output_text"])


# StreamLit UI


def main():
    st.header("PDF Reader")

    user_question = st.text_input("Enter your question", key="user_question")

    if user_question and api_key:
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if st.button("Submit $ Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectordb = get_vector_store(text_chunks, api_key)
                st.success("Done!")


if __name__ == "__main__":
    main()
