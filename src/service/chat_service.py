# FILE: src/service/chat_service.py
from typing import List
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
from src.config import settings


class ChatService:
    """
    ChatService handles conversation logic with the Groq LLM.
    Completely independent of Streamlit.
    """

    def __init__(self) -> None:
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.chat_groq = ChatGroq(
            model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY
        )
        self.conversation = self._init_conversation()
        self.requests: List[str] = []
        self.responses: List[str] = [settings.DEFAULT_BOT_MESSAGE]

    def _init_conversation(self) -> ConversationChain:
        """
        Initializes the LangChain ConversationChain with memory and prompt templates.
        """
        memory = ConversationBufferWindowMemory(k=3, return_messages=True)
        system_msg = SystemMessagePromptTemplate.from_template(
            template="""Answer the question as truthfully as possible using the provided context. 
            If answer is not contained in context, say 'I don't know'. Suggest uploading PDFs only if unknown."""
        )
        human_msg = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages(
            [system_msg, MessagesPlaceholder(variable_name="history"), human_msg]
        )
        return ConversationChain(
            memory=memory, prompt=prompt_template, llm=self.chat_groq, verbose=True
        )

    def query_refiner(self, conversation_str: str, query: str) -> str:
        """
        Refines a user query based on conversation history using Groq API.

        Args:
            conversation_str (str): Conversation history.
            query (str): User input query.

        Returns:
            str: Refined query.
        """
        prompt = (
            f"Given the following user query and conversation log, formulate a question most relevant "
            f"to provide the user with an answer from a knowledge base.\n\n"
            f"CONVERSATION LOG: \n{conversation_str}\n\nQuery: {query}\n\nRefined Query:"
        )
        response = self.groq_client.chat.completions.create(
            model=settings.GROQ_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_conversation_string(self) -> str:
        """
        Generates conversation history string from stored requests and responses.

        Returns:
            str: Formatted conversation string.
        """
        conversation_str = ""
        for i in range(len(self.responses) - 1):
            conversation_str += f"Human: {self.requests[i]}\n"
            conversation_str += f"Bot: {self.responses[i + 1]}\n"
        return conversation_str

    def add_interaction(self, user_query: str, bot_response: str) -> None:
        """
        Adds a user query and bot response to the conversation state.

        Args:
            user_query (str): User input.
            bot_response (str): Bot's answer.
        """
        self.requests.append(user_query)
        self.responses.append(bot_response)
