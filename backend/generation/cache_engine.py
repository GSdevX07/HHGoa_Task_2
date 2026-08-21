"""
Fast Cache Engine for Voice RAG System
=======================================
Provides:
  1. ExactCache: Sub-millisecond normalized string lookup (<0.1ms)
  2. SemanticCache: In-memory cosine similarity embedding cache (<0.8ms)
  3. Pre-warming with MSMARCO-XI corpus queries & answers at startup
"""

import re
import time
import unicodedata
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger("generation.cache")

def normalize_query_text(text: str) -> str:
    """Normalize text: NFKC, lowercase, strip punctuation and extra whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    # Replace punctuation with space but keep alphanumeric and Indic script chars
    text = re.sub(r"[^\w\s\u0900-\u0D7F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ExactCache:
    """
    LRU Memory Cache for exact normalized queries.
    Target latency: < 0.1ms
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []

    def get(self, query: str, lang_code: str = "") -> Optional[Dict[str, Any]]:
        norm_key = f"{lang_code}:{normalize_query_text(query)}" if lang_code else normalize_query_text(query)
        if norm_key in self._cache:
            # Move to end of access order
            try:
                self._access_order.remove(norm_key)
            except ValueError:
                pass
            self._access_order.append(norm_key)
            entry = self._cache[norm_key]
            return {
                "answer": entry["answer"],
                "citations": entry.get("citations", []),
                "groundedness_score": entry.get("groundedness_score", 1.0),
                "is_grounded": True,
                "cache_type": "EXACT_CACHE_HIT",
            }
        return None

    def put(self, query: str, answer: str, citations: List[Any], lang_code: str = "", groundedness_score: float = 1.0):
        if not answer or len(answer.strip()) < 4:
            return
        norm_key = f"{lang_code}:{normalize_query_text(query)}" if lang_code else normalize_query_text(query)
        if norm_key in self._cache:
            self._cache[norm_key] = {
                "answer": answer,
                "citations": citations,
                "groundedness_score": groundedness_score,
                "timestamp": time.time(),
            }
            return

        if len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)

        self._cache[norm_key] = {
            "answer": answer,
            "citations": citations,
            "groundedness_score": groundedness_score,
            "timestamp": time.time(),
        }
        self._access_order.append(norm_key)

    def clear(self):
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class SemanticCache:
    """
    In-memory semantic vector cache using cosine similarity.
    Target latency: < 0.8ms
    """
    def __init__(self, threshold: float = 0.93, max_size: int = 2000):
        self.threshold = threshold
        self.max_size = max_size
        self._embeddings: List[np.ndarray] = []
        self._entries: List[Dict[str, Any]] = []

    def get(self, query_vector: np.ndarray) -> Optional[Tuple[Dict[str, Any], float]]:
        if not self._embeddings or query_vector is None:
            return None

        # Stack into matrix and compute cosine similarities (assuming unit vectors)
        matrix = np.vstack(self._embeddings)
        norm_q = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        similarities = np.dot(matrix, norm_q)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            entry = self._entries[best_idx]
            return {
                "answer": entry["answer"],
                "citations": entry.get("citations", []),
                "groundedness_score": entry.get("groundedness_score", 1.0),
                "is_grounded": True,
                "cache_type": "SEMANTIC_CACHE_HIT",
                "similarity": best_score,
            }, best_score
        return None

    def put(self, query_vector: np.ndarray, answer: str, citations: List[Any], groundedness_score: float = 1.0):
        if query_vector is None or not answer:
            return
        norm_v = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        if len(self._embeddings) >= self.max_size:
            self._embeddings.pop(0)
            self._entries.pop(0)

        self._embeddings.append(norm_v)
        self._entries.append({
            "answer": answer,
            "citations": citations,
            "groundedness_score": groundedness_score,
            "timestamp": time.time(),
        })

    def clear(self):
        self._embeddings.clear()
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
