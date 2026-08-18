"""
Cross-Encoder Reranker — BAAI/bge-reranker-v2-m3
=================================================
State-of-the-art multilingual cross-encoder that rescores (query, passage) pairs
with full attention between query and passage tokens.

Model: BAAI/bge-reranker-v2-m3
  - 568M parameters, multilingual (100+ languages incl. all Indic scripts)
  - Significantly better than bi-encoder retrieval for final passage selection
  - Reasonable latency for reranking a small candidate pool (20 → top 3)

Architecture in the pipeline:
    HybridRetriever.search(top_k=20)
           ↓ 20 candidates
    CrossEncoderReranker.rerank(query, 20 candidates)
           ↓ top 3 passages
    LLM context window
"""

import os
import logging
import time
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("retrieval.reranker")

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """
    BAAI/bge-reranker-v2-m3 cross-encoder for precise passage reranking.

    The model is loaded lazily on first use to keep server startup fast.
    If the model is unavailable (no GPU / no disk space), the reranker
    gracefully falls back to returning the dense RRF ranking unchanged.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_length: int = 512,
        enabled: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            max_length: Max token length for each (query, passage) pair.
                        512 is safe for most passages after chunking.
            enabled:    Can be disabled via RERANKER_ENABLED=false env var.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.enabled = enabled and os.getenv("RERANKER_ENABLED", "true").lower() != "false"
        self._model = None
        self._load_attempted = False
        self._available = False

    # ── Lazy Model Load ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load model on first call. Returns True if model is available."""
        if self._load_attempted:
            return self._available

        self._load_attempted = True

        if not self.enabled:
            logger.info("CrossEncoderReranker: disabled via RERANKER_ENABLED=false")
            self._available = False
            return False

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"CrossEncoderReranker: loading '{self.model_name}' ...")
            t0 = time.perf_counter()
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                # Use CPU; GPU will be auto-detected if available
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"CrossEncoderReranker: model loaded in {elapsed:.0f}ms")
            self._available = True
        except Exception as exc:
            logger.warning(
                f"CrossEncoderReranker: could not load '{self.model_name}' — {exc}. "
                "Falling back to RRF ranking (no reranking)."
            )
            self._available = False

        return self._available

    # ── Reranking ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        use_parent_text: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Rerank candidate passages using cross-encoder scoring.

        Args:
            query:           The user's query string.
            candidates:      List of candidate dicts from HybridRetriever.
                             Each must have 'text', 'parent_text', and other fields.
            top_k:           Number of final passages to return.
            use_parent_text: If True, scores the parent passage (wider context)
                             but returns the child chunk for citation precision.
                             This implements the parent-child retrieval pattern
                             without inflating LLM context.

        Returns:
            (reranked_top_k, reranker_latency_ms)
        """
        t0 = time.perf_counter()

        if not candidates:
            return [], round((time.perf_counter() - t0) * 1000, 3)

        # Fallback: return top-k of the already-fused ranking
        if not self._ensure_loaded():
            elapsed = (time.perf_counter() - t0) * 1000
            fallback = candidates[:top_k]
            for i, c in enumerate(fallback):
                c["reranker_score"] = c.get("rrf_score", 0.0)
                c["rank"] = i + 1
            return fallback, round(elapsed, 3)

        # Build (query, passage) pairs for the cross-encoder
        # Use parent_text for scoring to leverage broader context,
        # but keep child text in the result for precise citations.
        scoring_texts = []
        for c in candidates:
            passage = c.get("parent_text", c["text"]) if use_parent_text else c["text"]
            # Truncate very long passages to avoid exceeding max_length
            # Cross-encoder tokenizer handles this, but being explicit avoids
            # unexpected slow batches.
            passage = passage[:2000]
            scoring_texts.append((query, passage))

        # Batch inference — faster than individual calls
        raw_scores = self._model.predict(
            scoring_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        # Attach scores and sort descending
        scored = []
        for score, candidate in zip(raw_scores, candidates):
            entry = dict(candidate)
            entry["reranker_score"] = round(float(score), 6)
            scored.append(entry)

        scored.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Re-assign ranks
        top = scored[:top_k]
        for i, entry in enumerate(top):
            entry["rank"] = i + 1

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"CrossEncoderReranker: {len(candidates)} → {len(top)} passages "
            f"in {elapsed:.1f}ms. "
            f"Top score: {top[0]['reranker_score']:.4f} (was RRF rank {top[0].get('rank', '?')})"
            if top else ""
        )

        return top, round(elapsed, 3)

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._available

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "enabled": self.enabled,
            "loaded": self._available,
            "max_length": self.max_length,
        }
