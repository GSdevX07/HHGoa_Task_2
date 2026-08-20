"""
Dense Retriever — FAISS + SentenceTransformer
=============================================
Loads pre-built FAISS index from disk and performs
dense vector similarity search with LRU query caching.
"""

import os
import json
import logging
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

# Force PyTorch backend and disable TensorFlow/Keras 3 hooks in Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np

logger = logging.getLogger("retrieval.dense")


class DenseRetriever:
    """
    Dense embedding retriever backed by FAISS IndexFlatIP.

    At startup, loads a pre-built index from disk (built by build_index.py).
    Falls back to in-memory indexing if no pre-built index is found.

    Query caching: LRU cache (default 1 000 entries) for repeated queries.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_size: int = 1_000,
    ):
        self.model_name = model_name
        self.model = None
        self.vector_dim = 384

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.faiss_index = None

        self._query_cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._loaded_from_disk = False

        self._init_model()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_model(self):
        """Load SentenceTransformer; gracefully skip if unavailable."""
        try:
            import torch
            torch.set_num_threads(4)
            torch.set_grad_enabled(False)
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model.eval()
            self.vector_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"DenseRetriever: loaded model '{self.model_name}' (dim={self.vector_dim})")
        except Exception as exc:
            logger.warning(f"DenseRetriever: could not load SentenceTransformer — {exc}. "
                           "Falling back to hash-based bi-encoder.")
            self.model = None

    # ── Disk I/O ────────────────────────────────────────────────────────────

    def load_from_disk(self, index_dir: str) -> bool:
        """
        Load pre-built index from index_dir/dense/.
        Returns True if successful.
        """
        dense_dir = os.path.join(index_dir, "dense")
        meta_path = os.path.join(dense_dir, "chunks_metadata.json")
        npy_path = os.path.join(dense_dir, "embeddings.npy")
        faiss_path = os.path.join(dense_dir, "faiss.index")

        if not (os.path.exists(meta_path) and os.path.exists(npy_path)):
            logger.info("DenseRetriever: no pre-built index found on disk.")
            return False

        try:
            t0 = time.perf_counter()

            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)

            self.embeddings_matrix = np.load(npy_path)
            self.vector_dim = self.embeddings_matrix.shape[1]

            # Load FAISS index if available
            if os.path.exists(faiss_path):
                try:
                    import faiss
                    self.faiss_index = faiss.read_index(faiss_path)
                    logger.info(f"DenseRetriever: loaded FAISS index ({self.faiss_index.ntotal} vectors)")
                except Exception as exc:
                    logger.warning(f"DenseRetriever: FAISS load failed ({exc}) — will use NumPy fallback")
                    self.faiss_index = None
            else:
                self._rebuild_faiss()

            self._query_cache.clear()
            self._loaded_from_disk = True
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                f"DenseRetriever: loaded {len(self.chunks)} chunks from disk in {elapsed:.1f}ms"
            )
            return True

        except Exception as exc:
            logger.error(f"DenseRetriever: failed to load index — {exc}")
            return False

    def _rebuild_faiss(self):
        """Rebuild FAISS index from in-memory embeddings (no disk write)."""
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return
        try:
            import faiss
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            self.faiss_index.add(self.embeddings_matrix)
        except Exception:
            self.faiss_index = None

    def save_to_disk(self, index_dir: str):
        """Persist current in-memory index to disk."""
        dense_dir = os.path.join(index_dir, "dense")
        os.makedirs(dense_dir, exist_ok=True)

        # Save metadata
        with open(os.path.join(dense_dir, "chunks_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        # Save embeddings
        if self.embeddings_matrix is not None:
            np.save(os.path.join(dense_dir, "embeddings.npy"), self.embeddings_matrix)

        # Save FAISS
        if self.faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_index, os.path.join(dense_dir, "faiss.index"))
            except Exception as exc:
                logger.warning(f"DenseRetriever: could not save FAISS index — {exc}")

        logger.info(f"DenseRetriever: saved {len(self.chunks)} chunks to {dense_dir}")

    # ── In-memory Indexing (Fallback) ───────────────────────────────────────

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build an in-memory index from a list of chunk dicts.
        Used when no pre-built disk index exists (demo / CI mode).
        """
        t0 = time.perf_counter()
        self.chunks = chunks
        self._query_cache.clear()

        if not chunks:
            self.embeddings_matrix = None
            return {"indexed_count": 0, "indexing_ms": 0.0}

        texts = [c["text"] for c in chunks]
        self.embeddings_matrix = self._embed(texts)
        self._rebuild_faiss()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"DenseRetriever: indexed {len(chunks)} chunks in {elapsed:.1f}ms (in-memory)")
        return {
            "indexed_count": len(chunks),
            "vector_dim": self.vector_dim,
            "indexing_ms": round(elapsed, 2),
        }

    # ── Embedding ───────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings for `texts`."""
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)

        if self.model is not None:
            try:
                import torch
                torch.set_num_threads(4)
                with torch.no_grad():
                    return self.model.encode(
                        texts,
                        batch_size=128,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    ).astype(np.float32)
            except Exception:
                return self.model.encode(
                    texts,
                    batch_size=128,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)

        # Hash-based fallback (zero-dependency)
        matrix = np.zeros((len(texts), self.vector_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = abs(hash(word)) % self.vector_dim
                matrix[i, h] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)

    # ── Search ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        similarity_threshold: float = 0.0,
        query_vector: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Search for top-k similar chunks.

        Returns:
            (results, retrieval_ms)
            results: list of dicts with keys chunk_id, text, parent_text,
                     metadata, dense_score, rank
        """
        t0 = time.perf_counter()
        query_key = query.strip().lower()

        # LRU cache check
        if query_vector is None and query_key in self._query_cache:
            self._query_cache.move_to_end(query_key)
            cached = self._query_cache[query_key]
            elapsed = (time.perf_counter() - t0) * 1000
            return cached, round(elapsed, 3)

        results = []

        if self.embeddings_matrix is not None and len(self.embeddings_matrix) > 0:
            if query_vector is not None:
                q_vec = query_vector
            else:
                q_vec = self._embed([query])[0]  # shape (dim,)

            if self.faiss_index is not None:
                k = min(top_k, self.faiss_index.ntotal)
                scores, indices = self.faiss_index.search(
                    q_vec.reshape(1, -1), k
                )
                raw_scores = scores[0].tolist()
                raw_indices = indices[0].tolist()
            else:
                sims = np.dot(self.embeddings_matrix, q_vec)
                k = min(top_k, len(sims))
                raw_indices = np.argsort(sims)[::-1][:k].tolist()
                raw_scores = [float(sims[i]) for i in raw_indices]

            for rank, (idx, score) in enumerate(zip(raw_indices, raw_scores)):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                if score < similarity_threshold:
                    continue
                chunk = self.chunks[idx]
                results.append({
                    "rank": rank + 1,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "parent_text": chunk.get("parent_text", chunk["text"]),
                    "metadata": chunk.get("metadata", {}),
                    "dense_score": round(float(score), 6),
                })

        # Update LRU cache
        self._query_cache[query_key] = results
        if len(self._query_cache) > self._cache_size:
            self._query_cache.popitem(last=False)

        elapsed = (time.perf_counter() - t0) * 1000
        return results, round(elapsed, 3)

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks) and self.embeddings_matrix is not None

    def __len__(self):
        return len(self.chunks)
