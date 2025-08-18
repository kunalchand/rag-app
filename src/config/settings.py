from dotenv import load_dotenv
import os

load_dotenv()

# App
APP_TITLE = "PDF Langchain Chatbot"
APP_ICON = "🤖"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# HuggingFace Embeddings
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
HF_DEVICE = os.getenv("HF_DEVICE")
HF_NORMALIZE = os.getenv("HF_NORMALIZE") == "True"

# Default Bot Message
DEFAULT_BOT_MESSAGE = (
    "Created by [Kunal Chand](https://kunalchand.github.io/portfolio/) with ❤️\n\n"
    "How can I assist you?"
)
