"""
pipeline/rag.py
---------------
Bonus: Mini-RAG (Retrieval-Augmented Generation).

Loads .txt / .md documents from a folder, indexes them with TF-IDF,
and retrieves the most relevant chunks to inject into the generation prompt.

No external vector database required – everything is in-memory.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .logger import logger


class RAGRetriever:
    """
    In-memory TF-IDF based document retriever.

    Usage
    -----
    retriever = RAGRetriever(docs_path="./rag_docs")
    retriever.build_index()
    context = retriever.retrieve("whey protein benefits", top_k=3)
    """

    def __init__(self, docs_path: str = "./rag_docs", chunk_size: int = 400) -> None:
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self._chunks: List[str] = []
        self._sources: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._built = False

    def build_index(self) -> None:
        """Load all documents from docs_path and build a TF-IDF index."""
        if not os.path.isdir(self.docs_path):
            logger.warning(
                "[rag] docs_path %s does not exist. RAG disabled.", self.docs_path
            )
            return

        raw_docs: List[Tuple[str, str]] = []
        for fname in os.listdir(self.docs_path):
            if fname.endswith((".txt", ".md")):
                fpath = os.path.join(self.docs_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw_docs.append((fname, fh.read()))
                except OSError as exc:
                    logger.warning("[rag] Could not read %s: %s", fpath, exc)

        if not raw_docs:
            logger.warning("[rag] No documents found in %s.", self.docs_path)
            return

        for fname, content in raw_docs:
            words = content.split()
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i : i + self.chunk_size])
                self._chunks.append(chunk)
                self._sources.append(fname)

        if not self._chunks:
            return

        self._vectorizer = TfidfVectorizer(
            stop_words=None,  # keep all words for multilingual support
            ngram_range=(1, 2),
            max_features=10_000,
        )
        self._matrix = self._vectorizer.fit_transform(self._chunks)
        self._built = True
        logger.info(
            "[rag] Index built: %d chunks from %d documents.",
            len(self._chunks),
            len(raw_docs),
        )

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """
        Return the top_k most relevant chunks for the query,
        formatted as a context string for LLM injection.
        """
        if not self._built or self._vectorizer is None:
            return ""

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[str] = []
        for idx in top_indices:
            if scores[idx] < 0.05:
                continue
            results.append(f"[Source: {self._sources[idx]}]\n{self._chunks[idx]}")

        if not results:
            return ""

        return "\n\n---\n\n".join(results)

    @property
    def is_ready(self) -> bool:
        return self._built
