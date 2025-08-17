from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)
