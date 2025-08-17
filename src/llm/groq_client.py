from langchain_groq import ChatGroq


def build_groq_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(model=model, api_key=api_key)
