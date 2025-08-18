from pinecone import Pinecone

from config import settings


class PineconeService:
    def __init__(self):
        self.client = Pinecone(
            api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENV
        )
        self.index = self.client.Index(settings.PINECONE_INDEX_NAME)

    def delete_all_vectors(self):
        if len([ids for ids in self.index.list(namespace="")]) != 0:
            self.index.delete(delete_all=True, namespace="")

    def upsert_vectors(self, text_chunks, embeddings):
        vectors = [
            {"id": str(i), "values": emb, "metadata": {"text": chunk}}
            for i, (chunk, emb) in enumerate(zip(text_chunks, embeddings))
        ]
        self.index.upsert(vectors)

    def query_vectors(self, vector, top_k=2):
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True)
