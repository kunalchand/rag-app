# FILE: src/service/pinecone_service.py
from typing import List, Dict, Any
from pinecone import Pinecone
from src.config import settings


class PineconeService:
    """
    PineconeService handles vector storage, deletion, upsertion, and queries.
    Independent of Streamlit.
    """

    def __init__(self) -> None:
        self.client = Pinecone(
            api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV
        )
        self.index = self.client.Index(settings.PINECONE_INDEX_NAME)

    def delete_all_vectors(self) -> None:
        """
        Deletes all vectors from the Pinecone index.
        """
        if len([ids for ids in self.index.list(namespace="")]) != 0:
            self.index.delete(delete_all=True, namespace="")

    def upsert_vectors(
        self, text_chunks: List[str], embeddings: List[List[float]]
    ) -> None:
        """
        Upserts text chunks and their embeddings into the Pinecone index.

        Args:
            text_chunks (List[str]): List of text chunks.
            embeddings (List[List[float]]): Corresponding embeddings.
        """
        vectors = [
            {"id": str(i), "values": emb, "metadata": {"text": chunk}}
            for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings))
        ]
        self.index.upsert(vectors)

    def query_vectors(self, vector: List[float], top_k: int = 2) -> Dict[str, Any]:
        """
        Queries the Pinecone index for the top_k similar vectors.

        Args:
            vector (List[float]): Query embedding vector.
            top_k (int): Number of top matches to return.

        Returns:
            Dict[str, Any]: Query results containing matches and metadata.
        """
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True)
