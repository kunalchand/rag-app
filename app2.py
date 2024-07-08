# Groq API Token: *****

# pip install groq
# export GROQ_API_KEY=*****


# completion = client.chat.completions.create(
#     model="llama3-70b-8192",
#     messages=[
#         {"role": "user", "content": "Tell me a kock-knock joke"},
#     ],
# )

# print(completion)
# print(completion.choices[0].message.content)

import os
from math import e

import pinecone
import PyPDF2
import streamlit as st
from blinker import Namespace
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone

groq = Groq(
    api_key="****",
)

chatGroq = ChatGroq(
    model="llama3-70b-8192",
    api_key="*****",
)

pc = Pinecone(
    api_key="*****",
    # environment="****",
    environment="*****",
)

# pinecone_index = pc.Index("****")
pinecone_index = pc.Index("*****")


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


def create_vectors_in_index_in_pinecone(text_chunks, embeddings):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
        print(len(embedding))
        vectors.append({"id": str(i), "values": embedding, "metadata": {"text": chunk}})
    pinecone_index.upsert(vectors)


def get_similar_vector_from_pinecone(user_question):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    vector_embedding = hf.embed_query(user_question)
    print(len(vector_embedding))
    return pinecone_index.query(
        vector=vector_embedding,
        top_k=2,
        # include_values=True,
        include_metadata=True,
    )


def delete_all_vectors_in_pinecone():
    if len([ids for ids in pinecone_index.list(namespace="")]) == 0:
        pinecone_index.delete(delete_all=True, namespace="")


def find_match(user_question):
    result = get_similar_vector_from_pinecone(user_question)
    if len(result["matches"]) < 2:
        return "Sorry, I could not answer this question. The PDFs you provided did not contain enough information related to the question."
    else:
        return (
            result["matches"][0]["metadata"]["text"]
            + "\n"
            + result["matches"][1]["metadata"]["text"]
        )


def query_refiner(conversation, query):
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


from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from streamlit_chat import message

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
and if the answer is not contained within the text below, say 'I don't know'"""
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

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

st.markdown("# PDF Langchain Chatbot")

# api_key = st.text_input(
#     "Enter your Replicate API Token", type="password", key="api_key_input"
# )

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            context = find_match(refined_query)
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
                    st.session_state["requests"][i], is_user=True, key=str(i) + "_user"
                )


def main():
    # st.header("PDF Reader")

    # user_question = st.text_input("Enter your question", key="user_question")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_upChatGroqloader",
        )
        if st.button("Submit & Process", key="process_button"):
            with st.spinner("Processing..."):
                # delete_all_vectors_in_pinecone()

                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                # chunked_document_embeddings = get_embeddings(text_chunks)
                # create_vectors_in_index_in_pinecone(
                #     text_chunks, chunked_document_embeddings
                # )

                # print(len(text_chunks))
                # print(len(chunked_document_embeddings))

                # print(
                #     get_similar_vector_from_pinecone("Tell me about Nastenka's History")
                # )
                st.success("Length of raw_text: " + str(len(raw_text)))
                st.success("Length of text_chunks: " + str(len(text_chunks)))


if __name__ == "__main__":
    main()
