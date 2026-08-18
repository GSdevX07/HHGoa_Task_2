"""
BM25 Retriever — rank_bm25 with multilingual tokenization
==========================================================
Loads pre-built BM25 index from disk (built by build_index.py).
Provides sparse keyword-based retrieval as the complement to dense retrieval.
"""

import os
import json
import pickle
import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("retrieval.bm25")


# ── Multilingual Tokenizer ────────────────────────────────────────────────────

# Unicode ranges for major Indic scripts
_INDIC_PATTERN = re.compile(
    r"[\w"
    r"\u0900-\u097F"   # Devanagari (Hindi, Marathi, Sanskrit)
    r"\u0980-\u09FF"   # Bengali / Assamese
    r"\u0A00-\u0A7F"   # Gurmukhi (Punjabi)
    r"\u0A80-\u0AFF"   # Gujarati
    r"\u0B00-\u0B7F"   # Odia
    r"\u0B80-\u0BFF"   # Tamil
    r"\u0C00-\u0C7F"   # Telugu
    r"\u0C80-\u0CFF"   # Kannada
    r"\u0D00-\u0D7F"   # Malayalam
    r"\u0E00-\u0E7F"   # Thai (included for completeness)
    r"\u0600-\u06FF"   # Arabic / Urdu
    r"]+"
)

# Common stopwords (English + Hindi) — very light list to avoid removing useful terms
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "into", "or", "and", "but",
    "not", "that", "this", "it", "its", "which", "who", "what", "when",
    "where", "how", "why", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "than", "then", "there", "these",
    "they", "those", "through", "up", "about", "between", "out", "if",
    # Hindi stopwords (common)
    "की", "के", "का", "में", "है", "हैं", "को", "से", "पर", "और",
    "यह", "वह", "एक", "भी", "तो", "ने", "हो", "था", "थे", "थी",
    "यहाँ", "वहाँ", "जो", "कि", "कर", "लिए", "जाता", "जाती",
})


def tokenize(text: str) -> List[str]:
    """
    Multilingual tokenizer suitable for Latin + Indic + Arabic scripts.
    Lowercases Latin text; preserves case for Indic scripts (they don't have case).
    Removes stopwords and single-character tokens.
    """
    # Extract all word-like tokens (handles mixed-script text)
    tokens = _INDIC_PATTERN.findall(text)
    result = []
    for tok in tokens:
        # Lowercase only ASCII portions (avoids corrupting Indic chars)
        tok_lower = tok.lower() if tok.isascii() else tok
        if len(tok_lower) > 1 and tok_lower not in _STOPWORDS:
            result.append(tok_lower)
    return result


# ── BM25 Retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Sparse BM25Okapi retriever with multilingual tokenization.

    Loads pre-built index from disk. Falls back to in-memory indexing
    from a list of chunks if no disk index exists.
    """

    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self._loaded_from_disk = False

    # ── Disk I/O ────────────────────────────────────────────────────────────

    def load_from_disk(self, index_dir: str) -> bool:
        """
        Load BM25 index from index_dir/bm25/.
        Returns True if successful.
        """
        bm25_dir = os.path.join(index_dir, "bm25")
        pkl_path = os.path.join(bm25_dir, "bm25_corpus.pkl")
        ids_path = os.path.join(bm25_dir, "bm25_chunk_ids.json")

        if not (os.path.exists(pkl_path) and os.path.exists(ids_path)):
            logger.info("BM25Retriever: no pre-built index found on disk.")
            return False

        try:
            t0 = time.perf_counter()

            with open(pkl_path, "rb") as f:
                self.bm25 = pickle.load(f)

            with open(ids_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            self._loaded_from_disk = True
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                f"BM25Retriever: loaded {len(self.chunks)} chunks from disk in {elapsed:.1f}ms"
            )
            return True

        except Exception as exc:
            logger.error(f"BM25Retriever: failed to load index — {exc}")
            return False

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build an in-memory BM25 index (used as fallback when no disk index exists).
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed. BM25 retrieval disabled. "
                           "Install with: pip install rank-bm25")
            self.bm25 = None
            return {"indexed_count": 0, "error": "rank-bm25 not installed"}

        t0 = time.perf_counter()
        self.chunks = chunks
        tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"BM25Retriever: indexed {len(chunks)} chunks in {elapsed:.1f}ms (in-memory)")
        return {"indexed_count": len(chunks), "indexing_ms": round(elapsed, 2)}

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieve top-k chunks using BM25 scoring.

        Returns:
            (results, retrieval_ms)
            results: list of dicts with chunk_id, text, parent_text, metadata, bm25_score, rank
        """
        t0 = time.perf_counter()

        if self.bm25 is None or not self.chunks:
            elapsed = (time.perf_counter() - t0) * 1000
            return [], round(elapsed, 3)

        query_tokens = tokenize(query)

        # BM25 returns scores for every document in the corpus
        scores = self.bm25.get_scores(query_tokens)  # numpy array of length N

        # Get top-k indices sorted by descending score
        import numpy as np
        k = min(top_k, len(scores))
        top_indices = (-scores).argsort()[:k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                # BM25 score of 0 means no keyword overlap — skip
                continue
            chunk = self.chunks[idx]
            results.append({
                "rank": rank + 1,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "parent_text": chunk.get("parent_text", chunk["text"]),
                "metadata": chunk.get("metadata", {}),
                "bm25_score": round(score, 6),
            })

        elapsed = (time.perf_counter() - t0) * 1000
        return results, round(elapsed, 3)

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None and bool(self.chunks)

    def __len__(self):
        return len(self.chunks)
