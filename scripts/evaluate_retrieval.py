"""
Chunking Strategy Evaluation Script
====================================
Evaluates all 4 chunking strategies against the corpus using retrieval metrics.
Measures Recall@5, MRR, and latency per strategy.

Usage:
    python scripts/evaluate_retrieval.py [--queries 100]

Output:
    Prints a comparison table to stdout.
    Saves results to indexes/chunking_eval.json
"""

import os
import sys
import json
import time
import logging
import argparse
import re

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

DATA_DIR = os.path.join(REPO_ROOT, "data")
INDEX_DIR = os.path.join(REPO_ROOT, "indexes")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_retrieval")


def load_corpus(path: str) -> list:
    docs = []
    if not os.path.exists(path):
        # Fall back to built-in samples
        from dataset_loader import BUILTIN_MSMARCO_XI_SAMPLES
        return BUILTIN_MSMARCO_XI_SAMPLES
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def recall_at_k(retrieved_texts: list, gold_passages: list, k: int = 5) -> float:
    """Check if any gold passage appears (substring match) in top-k retrieved texts."""
    top_k = retrieved_texts[:k]
    for gold in gold_passages:
        gold_norm = normalize(gold)
        for rt in top_k:
            if gold_norm in normalize(rt) or normalize(rt) in gold_norm:
                return 1.0
    return 0.0


def reciprocal_rank(retrieved_texts: list, gold_passages: list) -> float:
    """MRR: 1/rank of the first relevant result."""
    for rank, rt in enumerate(retrieved_texts, 1):
        rt_norm = normalize(rt)
        for gold in gold_passages:
            gold_norm = normalize(gold)
            if gold_norm in rt_norm or rt_norm in gold_norm:
                return 1.0 / rank
    return 0.0


def evaluate_strategy(strategy: str, chunk_size: int, docs: list, model, max_queries: int) -> dict:
    """Chunk, embed, index, and evaluate retrieval for one strategy."""
    from chunking_engine import ChunkingEngine

    engine = ChunkingEngine()

    # Chunk
    t_chunk = time.perf_counter()
    chunks = engine.chunk_documents(docs, strategy=strategy, chunk_size=chunk_size)
    chunk_ms = (time.perf_counter() - t_chunk) * 1000

    if not chunks:
        return {"strategy": strategy, "error": "No chunks generated"}

    # Embed
    texts = [c["text"] for c in chunks]
    t_embed = time.perf_counter()
    embeddings = model.encode(texts, batch_size=256, normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
    embed_ms = (time.perf_counter() - t_embed) * 1000

    # Build mini FAISS index
    try:
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        use_faiss = True
    except ImportError:
        use_faiss = False

    # Collect queries that have known answer passages
    eval_pairs = []
    for doc in docs:
        query = (doc.get("query_en") or doc.get("query", "")).strip()
        passage = (doc.get("passage_en") or doc.get("passage", "")).strip()
        if query and passage:
            eval_pairs.append({"query": query, "gold_passage": passage})

    eval_pairs = eval_pairs[:max_queries]
    if not eval_pairs:
        return {"strategy": strategy, "error": "No evaluation pairs found"}

    recalls, rrs, latencies = [], [], []

    for pair in eval_pairs:
        q_vec = model.encode([pair["query"]], normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
        t_ret = time.perf_counter()
        if use_faiss:
            scores, indices = index.search(q_vec, 5)
            top_indices = indices[0].tolist()
        else:
            sims = np.dot(embeddings, q_vec[0])
            top_indices = np.argsort(sims)[::-1][:5].tolist()
        ret_ms = (time.perf_counter() - t_ret) * 1000

        retrieved_texts = [chunks[i]["text"] for i in top_indices if 0 <= i < len(chunks)]
        recalls.append(recall_at_k(retrieved_texts, [pair["gold_passage"]], k=5))
        rrs.append(reciprocal_rank(retrieved_texts, [pair["gold_passage"]]))
        latencies.append(ret_ms)

    chunk_lengths_words = [len(c["text"].split()) for c in chunks]

    return {
        "strategy": strategy,
        "chunk_size_target_words": chunk_size,
        "total_chunks": len(chunks),
        "avg_chunk_words": round(sum(chunk_lengths_words) / len(chunk_lengths_words), 1),
        "min_chunk_words": min(chunk_lengths_words),
        "max_chunk_words": max(chunk_lengths_words),
        "recall_at_5": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(rrs) / len(rrs), 4),
        "avg_retrieval_ms": round(sum(latencies) / len(latencies), 3),
        "p50_retrieval_ms": round(float(np.percentile(latencies, 50)), 3),
        "p100_retrieval_ms": round(float(np.max(latencies)), 3),
        "chunking_ms": round(chunk_ms, 1),
        "embedding_ms": round(embed_ms, 1),
        "eval_queries": len(eval_pairs),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies.")
    parser.add_argument("--queries", type=int, default=100,
                        help="Max queries to evaluate per strategy (default: 100)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Target chunk size in words (default: 256)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Embedding model (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    print("=" * 72)
    print("  HH Goa Task 2 — Chunking Strategy Evaluation")
    print("=" * 72)
    print(f"  Corpus:     {CORPUS_PATH}")
    print(f"  Eval queries per strategy: {args.queries}")
    print(f"  Embedding model: {args.model}")
    print("-" * 72)

    docs = load_corpus(CORPUS_PATH)
    print(f"  Loaded {len(docs)} documents.\n")

    print("  Loading embedding model ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    strategies = ["fixed", "semantic", "metadata_aware", "parent_child"]
    results = []

    for strategy in strategies:
        print(f"  Evaluating [{strategy}] ...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = evaluate_strategy(strategy, args.chunk_size, docs, model, args.queries)
        total_s = time.perf_counter() - t0
        print(f"done ({total_s:.1f}s)")
        results.append(result)

    # Print comparison table
    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    header = f"  {'Strategy':<18} {'Chunks':>7} {'AvgWords':>9} {'Recall@5':>9} {'MRR':>7} {'P50 ret ms':>11}"
    print(header)
    print("  " + "-" * 68)
    for r in results:
        if "error" in r:
            print(f"  {r['strategy']:<18}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['strategy']:<18} "
            f"{r['total_chunks']:>7} "
            f"{r['avg_chunk_words']:>9.1f} "
            f"{r['recall_at_5']:>9.4f} "
            f"{r['mrr']:>7.4f} "
            f"{r['p50_retrieval_ms']:>11.3f}"
        )
    print("=" * 72)

    # Best strategy by MRR
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["mrr"])
        print(f"\n  Best strategy by MRR: [{best['strategy']}]  MRR={best['mrr']:.4f}")
        print(f"  → Use this in build_index.py: --strategy {best['strategy']}\n")

    # Save results
    os.makedirs(INDEX_DIR, exist_ok=True)
    out_path = os.path.join(INDEX_DIR, "chunking_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
