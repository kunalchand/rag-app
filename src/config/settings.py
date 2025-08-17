from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_env: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "rag-app")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")
    llm_model: str = os.getenv("LLM_MODEL", "llama3-70b-8192")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    namespace_strategy: str = os.getenv(
        "NAMESPACE_STRATEGY", "session"
    )  # or user/static


settings = Settings()
