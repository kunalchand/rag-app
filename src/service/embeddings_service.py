from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from configs import settings


class EmbeddingsService:
    def __init__(self):
        self.model = HuggingFaceBgeEmbeddings(
            model_name=settings.HF_MODEL_NAME,
            model_kwargs={"device": settings.HF_DEVICE},
            encode_kwargs={"normalize_embeddings": settings.HF_NORMALIZE},
        )

    def embed_texts(self, text_list):
        return [self.model.embed_query(text) for text in text_list]

    def embed_query(self, query):
        return self.model.embed_query(query)
