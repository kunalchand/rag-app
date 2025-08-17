from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from .schemas import VectorRecord


class PineconeStore:
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        embedding_model: str,
        device: str = "cpu",
    ):
        self.pc = Pinecone(api_key=api_key, environment=environment)
        self.index = self.pc.Index(index_name)
        self.embedder = HuggingFaceBgeEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Use embed_documents for batch efficiency
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)

    def upsert(
        self, records: List[VectorRecord], namespace: Optional[str] = None
    ) -> None:
        vectors = [
            {"id": r.id, "values": r.values, "metadata": r.metadata} for r in records
        ]
        self.index.upsert(vectors=vectors, namespace=namespace or "")

    def query(
        self,
        vector: List[float],
        top_k: int,
        namespace: Optional[str] = None,
        include_metadata: bool = True,
    ):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=namespace or "",
        )

    def clear_namespace(self, namespace: Optional[str] = None) -> None:
        ns = namespace or ""
        # Pinecone auto-creates namespace; clear when requested
        self.index.delete(delete_all=True, namespace=ns)
