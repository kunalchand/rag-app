# FILE: src/service/embeddings_service.py
from typing import List
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.config import settings


class EmbeddingsService:
    """
    EmbeddingsService wraps HuggingFace embeddings model.
    """

    def __init__(self) -> None:
        self.model = HuggingFaceBgeEmbeddings(
            model_name=settings.HF_MODEL_NAME,
            model_kwargs={"device": settings.HF_DEVICE},
            encode_kwargs={"normalize_embeddings": settings.HF_NORMALIZE},
        )

    def embed_texts(self, text_list: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts.

        Args:
            text_list (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return [self.model.embed_query(text) for text in text_list]

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.

        Args:
            query (str): Query string.

        Returns:
            List[float]: Embedding vector.
        """
        return self.model.embed_query(query)
