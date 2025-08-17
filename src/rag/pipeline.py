from typing import List
from ..vectorstore.pinecone_store import PineconeStore
from ..vectorstore.schemas import VectorRecord


class RAGPipeline:
    def __init__(self, store: PineconeStore, retriever, id_prefix: str = "doc"):
        self.store = store
        self.retriever = retriever
        self.id_prefix = id_prefix

    def index_texts(self, texts: List[str], namespace: str) -> int:
        embeddings = self.store.embed_texts(texts)
        records = [
            VectorRecord(id=f"{self.id_prefix}-{i}", values=emb, metadata={"text": t})
            for i, (t, emb) in enumerate(zip(texts, embeddings))
        ]
        self.store.upsert(records, namespace=namespace)
        return len(records)

    def build_context(self, query: str, namespace: str) -> str:
        matches = self.retriever.retrieve_context(query, namespace)
        if not matches:
            return ""
        # join top matches; preserve ordering and avoid duplicates
        return "\n\n".join(
            [
                m["metadata"]["text"]
                for m in matches
                if "metadata" in m and "text" in m["metadata"]
            ]
        )
