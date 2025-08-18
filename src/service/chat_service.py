import streamlit as st
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from config import settings


class ChatService:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.chat_groq = ChatGroq(
            model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY
        )
        self.conversation = self._init_conversation()

    def _init_conversation(self):
        if "responses" not in st.session_state:
            st.session_state["responses"] = [settings.DEFAULT_BOT_MESSAGE]
        if "requests" not in st.session_state:
            st.session_state["requests"] = []
        if "buffer_memory" not in st.session_state:
            st.session_state["buffer_memory"] = ConversationBufferWindowMemory(
                k=3, return_messages=True
            )

        system_msg = SystemMessagePromptTemplate.from_template(
            template="""Answer the question as truthfully as possible using the provided context. 
            If answer is not contained in context, say 'I don't know'. Suggest uploading PDFs only if unknown."""
        )
        human_msg = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg, MessagesPlaceholder(variable_name="history"), human_msg]
        )
        return ConversationChain(
            memory=st.session_state.buffer_memory,
            prompt=prompt_template,
            llm=self.chat_groq,
            verbose=True,
        )

    def query_refiner(self, conversation_str, query):
        prompt = (
            f"Given the following user query and conversation log, formulate a question most relevant "
            f"to provide the user with an answer from a knowledge base.\n\n"
            f"CONVERSATION LOG: \n{conversation_str}\n\nQuery: {query}\n\nRefined Query:"
        )
        response = self.groq_client.chat.completions.create(
            model=settings.GROQ_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @staticmethod
    def get_conversation_string():
        conversation_str = ""
        for i in range(len(st.session_state["responses"]) - 1):
            conversation_str += f"Human: {st.session_state['requests'][i]}\n"
            conversation_str += f"Bot: {st.session_state['responses'][i + 1]}\n"
        return conversation_str
