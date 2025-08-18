from typing import List, Dict, Any

from src.vectorstore.pinecone_store import PineconeStore


class Retriever:
    def __init__(self, store: PineconeStore, top_k: int):
        self.store = store
        self.top_k = top_k

    def retrieve_context(self, query: str, namespace: str) -> List[Dict[str, Any]]:
        qvec = self.store.embed_query(query)
        results = self.store.query(qvec, top_k=self.top_k, namespace=namespace)
        return results.get("matches", [])
