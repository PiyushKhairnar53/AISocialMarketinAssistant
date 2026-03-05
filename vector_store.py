from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_cohere import CohereEmbeddings


BRAND_VOICE_FILE = "brand_voice.txt"


def _load_brand_voice_text() -> Optional[str]:
    """Load brand voice text from disk if present."""
    if not os.path.exists(BRAND_VOICE_FILE):
        return None

    try:
        with open(BRAND_VOICE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content or None
    except OSError:
        return None


@lru_cache(maxsize=1)
def _build_faiss_store(cohere_api_key: Optional[str]) -> Optional[FAISS]:
    """
    Build a FAISS vector store for brand voice RAG.

    If no brand_voice.txt exists, return None and let callers fall back to a
    default professional tone.
    """
    brand_text = _load_brand_voice_text()
    if not brand_text:
        return None

    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,
    )
    docs = [Document(page_content=brand_text, metadata={"source": BRAND_VOICE_FILE})]
    return FAISS.from_documents(docs, embedding=embeddings)


def get_brand_voice_retriever(cohere_api_key: Optional[str]) -> BaseRetriever:
    """
    Return a retriever for brand voice documents.

    If no FAISS store is available (no brand_voice.txt), return a dummy retriever
    that yields a single Document with a default tone.
    """
    store = _build_faiss_store(cohere_api_key)
    if store is not None:
        return store.as_retriever(search_kwargs={"k": 1})

    class _DefaultRetriever(BaseRetriever):  # type: ignore[misc]
        def _get_relevant_documents(self, query: str):  # type: ignore[override]
            return [
                Document(
                    page_content=(
                        "Use a clear, concise, and professional marketing voice. "
                        "Be friendly, data-aware, and action-oriented. "
                        "Avoid slang unless explicitly requested."
                    ),
                    metadata={"source": "default"},
                )
            ]

        async def _aget_relevant_documents(self, query: str):  # type: ignore[override]
            return self._get_relevant_documents(query)

    return _DefaultRetriever()


__all__ = ["get_brand_voice_retriever"]

