"""
Build Offline FAISS Dense Index + BM25 Index from corpus.jsonl
==============================================================
Run ONCE after download_dataset.py.

Usage:
    python scripts/build_index.py [--strategy semantic] [--chunk-size 256]

Output:
    indexes/dense/chunks_metadata.json   — list of chunk dicts
    indexes/dense/embeddings.npy         — float32 matrix (N, 384)
    indexes/dense/faiss.index            — FAISS IndexFlatIP (serialised)
    indexes/bm25/bm25_corpus.pkl         — BM25Okapi object
    indexes/bm25/bm25_chunk_ids.json     — rank → chunk_id mapping
"""

import os
import sys
import json
import pickle
import logging
import argparse
import time

# Force PyTorch backend and disable TensorFlow/Keras 3 hooks in Transformers
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np

# Repo root resolution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

DATA_DIR = os.path.join(REPO_ROOT, "data")
INDEX_DIR = os.path.join(REPO_ROOT, "indexes")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_index")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list:
    """Load corpus.jsonl into a list of document dicts."""
    if not os.path.exists(path):
        logger.error(f"Corpus not found at {path}. Run scripts/download_dataset.py first.")
        sys.exit(1)
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs)} documents from corpus.")
    return docs


def chunk_corpus(docs: list, strategy: str, chunk_size: int) -> list:
    """Apply chunking strategy to all documents using the backend engine."""
    from chunking_engine import ChunkingEngine
    engine = ChunkingEngine()
    t0 = time.perf_counter()
    chunks = engine.chunk_documents(docs, strategy=strategy, chunk_size=chunk_size)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Chunking [{strategy}, size={chunk_size}]: "
        f"{len(docs)} docs → {len(chunks)} chunks in {elapsed:.1f}ms"
    )
    return chunks


# ── Dense Index ───────────────────────────────────────────────────────────────

def build_dense_index(chunks: list, model_name: str, dense_dir: str):
    """Embed all chunks and save FAISS index + metadata."""
    os.makedirs(dense_dir, exist_ok=True)

    logger.info(f"Loading embedding model: {model_name} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vector_dim = model.get_sentence_embedding_dimension()

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks (dim={vector_dim}) ...")

    t0 = time.perf_counter()
    # Batch encoding with progress bar
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    embed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"Embedded {len(texts)} chunks in {embed_ms:.0f}ms. Shape: {embeddings.shape}")

    # Save numpy array
    npy_path = os.path.join(dense_dir, "embeddings.npy")
    np.save(npy_path, embeddings)
    logger.info(f"Saved embeddings → {npy_path}")

    # Save FAISS index
    try:
        import faiss
        index = faiss.IndexFlatIP(vector_dim)
        index.add(embeddings)
        faiss_path = os.path.join(dense_dir, "faiss.index")
        faiss.write_index(index, faiss_path)
        logger.info(f"Saved FAISS index → {faiss_path}  ({index.ntotal} vectors)")
    except ImportError:
        logger.warning("faiss-cpu not installed — FAISS index not saved. Install: pip install faiss-cpu")

    # Save chunk metadata (text + metadata, no embeddings)
    metadata_path = os.path.join(dense_dir, "chunks_metadata.json")
    # Store only what's needed for retrieval; strip parent_text to keep file small
    slim_chunks = []
    for c in chunks:
        slim_chunks.append({
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "parent_text": c.get("parent_text", c["text"]),
            "metadata": c.get("metadata", {}),
        })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(slim_chunks, f, ensure_ascii=False)
    logger.info(f"Saved chunk metadata → {metadata_path}")


# ── BM25 Index ────────────────────────────────────────────────────────────────

def _tokenize_multilingual(text: str) -> list:
    """
    Simple multilingual tokenizer.
    Splits on whitespace and punctuation without requiring heavy NLP libraries.
    Works for Latin scripts, Devanagari, Tamil, Telugu, Bengali, etc.
    """
    import re
    # Lowercase for Latin scripts; preserve case for scripts that don't have it
    text = text.lower()
    # Split on whitespace and common punctuation
    tokens = re.findall(r"[\w\u0900-\u097F\u0980-\u09FF\u0A00-\u0AFF\u0B00-\u0BFF"
                        r"\u0C00-\u0CFF\u0D00-\u0D7F\u0E00-\u0E7F]+", text)
    # Remove single-character tokens (noise)
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


def build_bm25_index(chunks: list, bm25_dir: str):
    """Tokenize chunk texts and build BM25Okapi index."""
    os.makedirs(bm25_dir, exist_ok=True)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed. BM25 index skipped. Install: pip install rank-bm25")
        return

    logger.info(f"Building BM25 index over {len(chunks)} chunks ...")
    t0 = time.perf_counter()

    tokenized_corpus = [_tokenize_multilingual(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"BM25 index built in {bm25_ms:.0f}ms.")

    # Save BM25 object
    pkl_path = os.path.join(bm25_dir, "bm25_corpus.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 object → {pkl_path}")

    # Save chunk ID list (positional — rank i → chunks[i])
    ids_path = os.path.join(bm25_dir, "bm25_chunk_ids.json")
    chunk_index = [
        {"chunk_id": c["chunk_id"], "text": c["text"], "parent_text": c.get("parent_text", c["text"]),
         "metadata": c.get("metadata", {})}
        for c in chunks
    ]
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f, ensure_ascii=False)
    logger.info(f"Saved BM25 chunk map → {ids_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build offline FAISS + BM25 indexes.")
    parser.add_argument("--strategy", default="semantic",
                        choices=["fixed", "semantic", "metadata_aware", "parent_child"],
                        help="Chunking strategy (default: semantic)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Target chunk size in words (default: 256)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    dense_dir = os.path.join(INDEX_DIR, "dense")
    bm25_dir = os.path.join(INDEX_DIR, "bm25")

    logger.info("=" * 60)
    logger.info("HH Goa Task 2 — Offline Index Builder")
    logger.info(f"  Corpus:   {CORPUS_PATH}")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Chunk sz: {args.chunk_size} words")
    logger.info(f"  Model:    {args.model}")
    logger.info("=" * 60)

    docs = load_corpus(CORPUS_PATH)
    chunks = chunk_corpus(docs, args.strategy, args.chunk_size)

    # Save metadata with dimension validation support
    meta = {
        "chunking_strategy": args.strategy,
        "chunk_size": args.chunk_size,
        "embedding_model": args.model,
        "dimension": 384 if "MiniLM" in args.model else 384,
        "total_chunks": len(chunks),
        "total_documents": len(docs),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(INDEX_DIR, exist_ok=True)

    build_dense_index(chunks, args.model, dense_dir)
    build_bm25_index(chunks, bm25_dir)

    # Read actual dimension from dense retriever matrix if available
    npy_path = os.path.join(dense_dir, "embeddings.npy")
    if os.path.exists(npy_path):
        emb_matrix = np.load(npy_path)
        meta["dimension"] = int(emb_matrix.shape[1])

    with open(os.path.join(INDEX_DIR, "index_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Index build complete! {len(chunks)} chunks indexed.")
    logger.info("Next step: start the server with:  uvicorn backend.main:app --port 8000")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
