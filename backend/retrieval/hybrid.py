"""
Hybrid Retriever — Reciprocal Rank Fusion (RRF)
================================================
Combines dense vector retrieval (FAISS) and sparse keyword retrieval (BM25)
into a single ranked candidate pool using Reciprocal Rank Fusion.

Why RRF instead of a weighted score sum?
  Dense scores are cosine similarities in [0, 1].
  BM25 scores are TF-IDF frequencies on an arbitrary positive scale.
  A weighted linear combination would require per-corpus calibration.
  RRF normalises everything by rank position, making it naturally robust
  to scale differences. Standard k=60 from the Cormack et al. 2009 paper.
"""

import os
import logging
import time
from typing import List, Dict, Any, Tuple

from .dense import DenseRetriever
from .bm25 import BM25Retriever

logger = logging.getLogger("retrieval.hybrid")

# RRF constant — higher k reduces the impact of rank-1 vs rank-2 differences
_RRF_K = 60


class HybridRetriever:
    """
    Hybrid retrieval using Reciprocal Rank Fusion over dense + BM25 results.

    Pipeline:
        query
          ├── DenseRetriever.search(top_k=candidate_pool)
          └── BM25Retriever.search(top_k=candidate_pool)
                  ↓
           RRF score fusion
                  ↓
           merged top-k candidates (passed to reranker)
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        candidate_pool: int = 20,
    ):
        """
        Args:
            dense:          Pre-initialised DenseRetriever instance.
            bm25:           Pre-initialised BM25Retriever instance.
            candidate_pool: Number of results to fetch from each retriever
                            before fusion. Controls recall vs. latency trade-off.
                            Default 20 → reranker sees up to 20 merged candidates.
        """
        self.dense = dense
        self.bm25 = bm25
        self.candidate_pool = candidate_pool
        self._use_bm25 = True  # toggled if BM25 unavailable

    # ── Core Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Execute hybrid retrieval and return fused, ranked candidates.

        Returns:
            (results, stage_latencies_ms)
            results: list of dicts with chunk_id, text, parent_text,
                     metadata, dense_score, bm25_score, rrf_score, rank
            stage_latencies_ms: {"dense_ms", "bm25_ms", "fusion_ms", "total_ms"}
        """
        t_total = time.perf_counter()

        # ── Dense retrieval ────────────────────────────────────────────────
        dense_results, dense_ms = self.dense.search(
            query, top_k=self.candidate_pool
        )

        # ── BM25 retrieval ─────────────────────────────────────────────────
        bm25_results, bm25_ms = [], 0.0
        if self._use_bm25 and self.bm25.is_ready:
            bm25_results, bm25_ms = self.bm25.search(
                query, top_k=self.candidate_pool
            )
        elif not self.bm25.is_ready:
            # BM25 not available — log once and disable for this session
            if self._use_bm25:
                logger.info("HybridRetriever: BM25 not ready, using dense-only retrieval.")
                self._use_bm25 = False

        # ── RRF fusion ─────────────────────────────────────────────────────
        t_fusion = time.perf_counter()
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results, top_k)
        fusion_ms = (time.perf_counter() - t_fusion) * 1000

        total_ms = (time.perf_counter() - t_total) * 1000

        latencies = {
            "dense_ms": round(dense_ms, 3),
            "bm25_ms": round(bm25_ms, 3),
            "fusion_ms": round(fusion_ms, 3),
            "total_ms": round(total_ms, 3),
        }

        logger.debug(
            f"HybridRetriever: dense={len(dense_results)} BM25={len(bm25_results)} "
            f"→ fused={len(fused)}  [{total_ms:.1f}ms]"
        )

        return fused, latencies

    # ── RRF Implementation ───────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score for chunk c:
            rrf(c) = Σ_list  1 / (k + rank_c_in_list)

        Chunks that appear in both lists get contributions from both,
        naturally boosting results with multi-signal agreement.
        """
        # Map chunk_id → accumulated RRF score and base data
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}

        def _accumulate(results: List[Dict[str, Any]], score_key: str):
            for rank_0based, item in enumerate(results):
                cid = item["chunk_id"]
                rrf_contribution = 1.0 / (_RRF_K + rank_0based + 1)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution
                if cid not in chunk_data:
                    chunk_data[cid] = {
                        "chunk_id": cid,
                        "text": item["text"],
                        "parent_text": item.get("parent_text", item["text"]),
                        "metadata": item.get("metadata", {}),
                        "dense_score": 0.0,
                        "bm25_score": 0.0,
                        "rrf_score": 0.0,
                    }
                chunk_data[cid][score_key] = item.get(score_key, 0.0)

        _accumulate(dense_results, "dense_score")
        _accumulate(bm25_results, "bm25_score")

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused = []
        for final_rank, cid in enumerate(sorted_ids[:top_k]):
            entry = dict(chunk_data[cid])
            entry["rrf_score"] = round(rrf_scores[cid], 8)
            entry["rank"] = final_rank + 1
            fused.append(entry)

        return fused

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.dense.is_ready

    def load_from_disk(self, index_dir: str) -> Dict[str, bool]:
        """Convenience: load both retrievers from disk."""
        dense_ok = self.dense.load_from_disk(index_dir)
        bm25_ok = self.bm25.load_from_disk(index_dir)
        return {"dense": dense_ok, "bm25": bm25_ok}

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Convenience: index both retrievers in-memory."""
        self.dense.index_chunks(chunks)
        self.bm25.index_chunks(chunks)
