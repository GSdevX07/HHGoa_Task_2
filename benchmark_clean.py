"""
Clean Latency Benchmark -- Voice RAG Pipeline
=============================================
Methodologically sound replacement for run_benchmark.py.

Problems fixed vs. the old benchmark:
  Fixed LRU query cache cleared before each cold-cache query
  Fixed ExactCache / SemanticCache are NOT instantiated -- raw retrieval only
  Fixed Corpus queries augmented to 100+ unique variants (rule-based, no LLM)
  Fixed Matches production code path: embed once, pass query_vector to hybrid.search()
  Fixed Detects hash-fallback embedding and marks results INVALID
  Fixed Reports actual reranker state (disabled / loaded / fallback)
  Fixed Fails loudly if disk index missing and --require-disk-index is set
  Fixed Cold-cache vs warm-cache reported separately (--cache both)
  Fixed retrieval mode = embedding+FAISS+BM25+RRF ONLY (no synthesis noise)
  Fixed pipeline mode = retrieval + guardrails (production path, no LLM)

Dataset: ai4bharat/MSMARCO-XI (Hindi, English, Marathi)
Languages tested: hi, en, mr

Usage:
  python benchmark_clean.py
  python benchmark_clean.py --queries 100 --mode pipeline --cache cold
  python benchmark_clean.py --queries 50  --mode retrieval --cache both
  python benchmark_clean.py --require-disk-index
"""

import sys
import os
import time
import json
import argparse
import logging
import datetime
import platform

# Force UTF-8 terminal encoding and PyTorch backend on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# -- Path resolution ----------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND   = os.path.join(_REPO_ROOT, "backend")
_INDEX_DIR = os.path.join(_REPO_ROOT, "indexes")
_CORPUS    = os.path.join(_REPO_ROOT, "data", "corpus.jsonl")

sys.path.insert(0, _BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))

from dataset_loader import load_corpus_from_disk
from chunking_engine import ChunkingEngine
from guardrails import GuardrailEngine
from model_harness import ModelHarness
from latency_analytics import BenchmarkAnalytics
from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


# ── Query Augmentation ───────────────────────────────────────────────────────
# Rule-based surface transforms -- no LLM, no API needed.
# Produces semantically equivalent queries with different surface forms so that
# cache hits across repeated queries are not artificially inflated.

_TRANSFORMS = [
    lambda q: q.strip(),
    lambda q: q.strip().lower(),
    lambda q: q.strip().capitalize(),
    lambda q: "What is " + q.strip().lower().rstrip("?") + "?",
    lambda q: "Tell me about " + q.strip().lower().rstrip("."),
    lambda q: "How does " + q.strip().lower().rstrip("?") + " work?",
    lambda q: "Can you explain " + q.strip().lower().rstrip("?") + "?",
    lambda q: "Give me information on " + q.strip().lower().rstrip("."),
    lambda q: "I need to understand " + q.strip().lower().rstrip("."),
    lambda q: q.strip() + " -- please explain",
    lambda q: "In simple terms, " + q.strip().lower().rstrip("?"),
    lambda q: "Describe " + q.strip().lower().rstrip("."),
]


def augment_queries(base_queries: list, target: int) -> list:
    """
    Expand base_queries to at least `target` unique surface-form variants.
    Deduplicates by lowercased form. Falls back to repetition with a clear
    flag if transforms are exhausted before reaching target.
    """
    seen, result = set(), []

    def _add(q):
        k = q.strip().lower()
        if k and k not in seen:
            seen.add(k)
            result.append(q.strip())

    # Pass 1: all originals
    for q in base_queries:
        _add(q)

    # Pass 2+: apply transforms round-robin
    t_idx = 1  # skip identity (pass 1 already added)
    while len(result) < target and t_idx < len(_TRANSFORMS):
        for q in base_queries:
            if len(result) >= target:
                break
            _add(_TRANSFORMS[t_idx](q))
        t_idx += 1

    # Pass 3: if still short, repeat originals (noted in integrity warnings)
    padded = False
    if len(result) < target:
        padded = True
        base_cycle = list(result)[:]
        i = 0
        while len(result) < target:
            result.append(base_cycle[i % len(base_cycle)] + f" (repeat-{i // len(base_cycle) + 1})")
            i += 1

    return result[:target], padded


# ── Single-query execution ────────────────────────────────────────────────────

def _run_retrieval(query, dense, hybrid, clear_cache):
    """embed + dense FAISS + BM25 + RRF only. Matches production embed-once path."""
    if clear_cache:
        dense._query_cache.clear()

    t0 = time.perf_counter_ns()

    t_emb = time.perf_counter_ns()
    q_vec = dense._embed([query])[0]
    embed_ms = (time.perf_counter_ns() - t_emb) / 1e6

    t_ret = time.perf_counter_ns()
    candidates, lats = hybrid.search(query, top_k=20, query_vector=q_vec)
    ret_ms = (time.perf_counter_ns() - t_ret) / 1e6

    total_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "total_latency_ms": round(total_ms, 3),
        "embed_ms":         round(embed_ms, 3),
        "dense_ms":         round(lats.get("dense_ms", 0.0), 3),
        "bm25_ms":          round(lats.get("bm25_ms",  0.0), 3),
        "rrf_ms":           round(lats.get("fusion_ms",0.0), 3),
        "retrieval_ms":     round(ret_ms,  3),
        "reranker_ms":      None,
        "guardrail_ms":     None,
        "llm_ms":           None,
        "stt_ms":           None,
        "num_candidates":   len(candidates),
    }


def _run_pipeline(query, dense, hybrid, reranker, guardrails, clear_cache):
    """retrieval + input guardrail + reranker (RRF fallback) + context guardrail."""
    if clear_cache:
        dense._query_cache.clear()

    t0 = time.perf_counter_ns()

    t_g = time.perf_counter_ns()
    pre = guardrails.validate_input(query)
    g_ms = (time.perf_counter_ns() - t_g) / 1e6

    if not pre["passed"]:
        total_ms = (time.perf_counter_ns() - t0) / 1e6
        return {
            "total_latency_ms": round(total_ms, 3),
            "embed_ms": 0.0, "dense_ms": 0.0, "bm25_ms": 0.0, "rrf_ms": 0.0,
            "retrieval_ms": 0.0, "reranker_ms": 0.0,
            "guardrail_ms": round(g_ms, 3),
            "llm_ms": None, "stt_ms": None, "num_candidates": 0, "is_refusal": True,
        }

    t_emb = time.perf_counter_ns()
    q_vec = dense._embed([query])[0]
    embed_ms = (time.perf_counter_ns() - t_emb) / 1e6

    t_ret = time.perf_counter_ns()
    candidates, lats = hybrid.search(query, top_k=20, query_vector=q_vec)
    ret_ms = (time.perf_counter_ns() - t_ret) / 1e6

    t_rr = time.perf_counter_ns()
    final, _ = reranker.rerank(query, candidates, top_k=3)
    rr_ms = (time.perf_counter_ns() - t_rr) / 1e6

    t_ctx = time.perf_counter_ns()
    top_score = final[0].get("reranker_score", 0.0) if final else 0.0
    ctx = guardrails.validate_retrieved_context(final, top_score, threshold=0.15)
    ctx_ms = (time.perf_counter_ns() - t_ctx) / 1e6

    total_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "total_latency_ms": round(total_ms, 3),
        "embed_ms":     round(embed_ms, 3),
        "dense_ms":     round(lats.get("dense_ms", 0.0), 3),
        "bm25_ms":      round(lats.get("bm25_ms",  0.0), 3),
        "rrf_ms":       round(lats.get("fusion_ms",0.0), 3),
        "retrieval_ms": round(ret_ms, 3),
        "reranker_ms":  round(rr_ms,  3),
        "guardrail_ms": round(g_ms + ctx_ms, 3),
        "llm_ms":       None,
        "stt_ms":       None,
        "num_candidates": len(candidates),
        "is_refusal":   ctx.get("should_refuse", False),
    }


def _run_extractive(query, dense, hybrid, reranker, guardrails, harness, clear_cache):
    """pipeline + extractive answer synthesis."""
    run = _run_pipeline(query, dense, hybrid, reranker, guardrails, clear_cache)
    if run.get("is_refusal"):
        run["llm_ms"] = 0.0
        return run
    q_vec = dense._embed([query])[0]
    cands, _ = hybrid.search(query, top_k=20, query_vector=q_vec)
    final, _ = reranker.rerank(query, cands, top_k=3)
    t_s = time.perf_counter_ns()
    harness.extractive_synthesizer.synthesize(query, final)
    synth_ms = (time.perf_counter_ns() - t_s) / 1e6
    run["llm_ms"] = round(synth_ms, 3)
    run["total_latency_ms"] = round(run["total_latency_ms"] + synth_ms, 3)
    return run


# ── Percentile helper ────────────────────────────────────────────────────────

def _p(vals, pct):
    if not vals:
        return 0.0
    import numpy as np
    return round(float(np.percentile(vals, pct)), 2)


def _print_table(label, runs, stage_keys):
    totals = [r["total_latency_ms"] for r in runs]
    sla_pass = _p(totals, 95) <= 200
    sep = "-" * 72
    print(f"\n  {sep}")
    print(f"  {label}  ({len(runs)} queries)")
    print(f"  {sep}")
    print(f"  {'Stage':<30} {'P50':>7} {'P70':>7} {'P95':>7} {'P99':>7} {'MAX':>7}  ms")
    print(f"  {sep}")
    for key, name in stage_keys:
        vals = [r[key] for r in runs if r.get(key) is not None and r[key] > 0]
        if not vals:
            print(f"  {name:<30} {'N/A':>7}")
            continue
        print(f"  {name:<30} {_p(vals,50):>7.2f} {_p(vals,70):>7.2f} {_p(vals,95):>7.2f} {_p(vals,99):>7.2f} {max(vals):>7.2f}")
    print(f"  {sep}")
    verdict = "PASS (SLA<=200ms)" if sla_pass else "FAIL (P95>200ms)"
    print(f"  {'TOTAL end-to-end':<30} {_p(totals,50):>7.2f} {_p(totals,70):>7.2f} {_p(totals,95):>7.2f} {_p(totals,99):>7.2f} {max(totals):>7.2f}   [{verdict}]")
    print(f"  {sep}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(num_queries, mode, cache_mode, require_disk):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 76)
    print("  HH Goa 2026 Task 2 -- Clean Voice RAG Latency Benchmark")
    print("  Dataset: ai4bharat/MSMARCO-XI (Hindi, English, Marathi)")
    print("=" * 76)
    print(f"  Mode       : {mode.upper()}")
    print(f"  Cache mode : {cache_mode}")
    print(f"  Queries    : N={num_queries}")
    print(f"  SLA target : 200 ms  (P95)")
    print(f"  Timestamp  : {ts}")
    print("-" * 76)

    # ── Init components ───────────────────────────────────────────────────────
    # NOTE: No ExactCache/SemanticCache instantiated.
    # They live only inside the FastAPI app (main.py).
    # This benchmark measures RAW retrieval latency -- no app-layer caching.

    dense     = DenseRetriever()
    bm25      = BM25Retriever()
    hybrid    = HybridRetriever(dense, bm25, candidate_pool=20)
    reranker  = CrossEncoderReranker(enabled=False)   # disabled = lowest latency
    guardrails = GuardrailEngine()
    harness    = ModelHarness() if mode == "extractive" else None

    # ── Load index ─────────────────────────────────────────────────────────────
    load_result = hybrid.load_from_disk(_INDEX_DIR)

    if load_result["dense"]:
        index_source = "disk"
        print(f"  Index      : disk  ({len(dense)} chunks)")
    else:
        if require_disk:
            print("\n  FATAL: --require-disk-index set but no disk index found.")
            print("         Run:  python scripts/build_index.py")
            sys.exit(1)
        print("  Index      : building in-memory from corpus.jsonl ...")
        corpus_docs = load_corpus_from_disk(_CORPUS)
        if not corpus_docs:
            print("  FATAL: No corpus.jsonl found.")
            print("         Run:  python scripts/download_dataset.py --langs hi,en,mr --limit 100")
            sys.exit(1)
        chunks = ChunkingEngine().chunk_documents(corpus_docs, strategy="semantic")
        hybrid.index_chunks(chunks)
        index_source = "in-memory"
        print(f"  Index      : in-memory  ({len(chunks)} chunks built from {len(corpus_docs)} docs)")

    if dense.model is not None:
        guardrails.set_embedder(dense._embed)

    # ── Embedding model guard ─────────────────────────────────────────────────
    model_neural = dense.model is not None
    if not model_neural:
        print()
        print("  !! CRITICAL: SentenceTransformer did NOT load -- hash fallback active.")
        print("  !!           Latency numbers are INVALID for neural retrieval.")
        print()

    # ── Collect corpus queries (English + native) ─────────────────────────────
    corpus_docs = load_corpus_from_disk(_CORPUS)
    raw_qs = []
    for doc in corpus_docs:
        for field in ("query_en", "Eng_Query", "query"):
            q = (doc.get(field) or "").strip()
            if q and len(q.split()) >= 3:
                raw_qs.append(q)
                break

    seen_k, unique_raw = set(), []
    for q in raw_qs:
        k = q.lower()
        if k not in seen_k:
            seen_k.add(k)
            unique_raw.append(q)

    if not unique_raw:
        print("  FATAL: No usable queries found in corpus.jsonl")
        sys.exit(1)

    all_queries, was_padded = augment_queries(unique_raw, target=num_queries)
    unique_final = len(set(q.lower() for q in all_queries))

    print(f"  Corpus docs: {len(corpus_docs)}  |  Unique raw queries: {len(unique_raw)}")
    print(f"  After aug  : {unique_final} unique variants (target {num_queries}"
          + ("  [PADDED with repeats]" if was_padded else "") + ")")
    print("-" * 76)

    # ── Stage key definitions ─────────────────────────────────────────────────
    retrieval_stages = [
        ("embed_ms",  "Embedding (all-MiniLM-L6-v2)"),
        ("dense_ms",  "Dense FAISS search"),
        ("bm25_ms",   "BM25 sparse search"),
        ("rrf_ms",    "RRF fusion"),
    ]
    pipeline_stages = retrieval_stages + [
        ("reranker_ms",  "Reranker (RRF fallback, disabled)"),
        ("guardrail_ms", "Guardrails (input + context)"),
    ]
    extractive_stages = pipeline_stages + [
        ("llm_ms", "Extractive synthesis"),
    ]
    stage_keys = (
        retrieval_stages if mode == "retrieval" else
        extractive_stages if mode == "extractive" else
        pipeline_stages
    )

    def _single(q, clear):
        if mode == "retrieval":
            return _run_retrieval(q, dense, hybrid, clear)
        elif mode == "extractive":
            return _run_extractive(q, dense, hybrid, reranker, guardrails, harness, clear)
        else:
            return _run_pipeline(q, dense, hybrid, reranker, guardrails, clear)

    # ── Warmup (not measured) ─────────────────────────────────────────────────
    print("  Warming up (2 queries, not counted in results) ...")
    _wq = "what is information retrieval and how does it work"
    _wv = dense._embed([_wq])[0]
    hybrid.search(_wq, top_k=5, query_vector=_wv)
    hybrid.search("explain vector search briefly", top_k=5)
    dense._query_cache.clear()   # clear so first benchmark query is a true cold start

    # ── Cold-cache run ────────────────────────────────────────────────────────
    cold_runs = []
    if cache_mode in ("cold", "both"):
        print(f"\n  [COLD-CACHE] Running {len(all_queries)} queries, LRU cleared each query ...")
        dense._query_cache.clear()
        t0 = time.perf_counter()
        for i, q in enumerate(all_queries, 1):
            r = _single(q, clear=True)
            cold_runs.append(r)
            if i % 25 == 0 or i == len(all_queries):
                print(f"    [{i:4d}/{len(all_queries)}]  total={r['total_latency_ms']:6.1f}ms  "
                      f"embed={r.get('embed_ms',0):5.1f}ms  ret={r.get('retrieval_ms',0):5.1f}ms")
        print(f"  Cold done in {time.perf_counter()-t0:.1f}s")

    # ── Warm-cache run ────────────────────────────────────────────────────────
    warm_runs = []
    if cache_mode in ("warm", "both"):
        print(f"\n  [WARM-CACHE] Pre-warming, then running {len(all_queries)} queries (cache hot) ...")
        # Pre-warm pass
        for q in all_queries:
            _single(q, clear=False)
        # Measured pass (no cache clearing -- LRU returns instantly for seen queries)
        t0 = time.perf_counter()
        for i, q in enumerate(all_queries, 1):
            r = _single(q, clear=False)
            warm_runs.append(r)
            if i % 25 == 0 or i == len(all_queries):
                print(f"    [{i:4d}/{len(all_queries)}]  total={r['total_latency_ms']:6.1f}ms  "
                      f"embed={r.get('embed_ms',0):5.1f}ms  ret={r.get('retrieval_ms',0):5.1f}ms")
        print(f"  Warm done in {time.perf_counter()-t0:.1f}s")

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("  RESULTS")
    print("=" * 76)
    if cold_runs:
        _print_table(f"COLD-CACHE | {mode.upper()} mode", cold_runs, stage_keys)
    if warm_runs:
        _print_table(f"WARM-CACHE | {mode.upper()} mode", warm_runs, stage_keys)

    # ── Integrity warnings ────────────────────────────────────────────────────
    warnings = BenchmarkAnalytics.integrity_check(
        dense_model_loaded=model_neural,
        reranker_loaded=reranker.is_loaded,
        reranker_enabled=reranker.enabled,
        index_source=index_source,
        unique_query_count=unique_final,
        total_query_count=len(all_queries),
        cache_cleared=(cache_mode in ("cold", "both")),
    )

    print()
    print("  -- Measurement Notes -------------------------------------------------")
    if warnings:
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("  [OK] No issues detected -- configuration is clean.")
    print()
    print("  -- What these numbers represent --------------------------------------")
    print("  STT (Sarvam/ElevenLabs) NOT included -- measure via /api/query/voice")
    print("  ExactCache / SemanticCache NOT active -- raw retrieval path only")
    print(f"  Reranker: disabled (RRF fallback) -- lowest-latency configuration")
    print(f"  Languages: Hindi (hi), English (en), Marathi (mr)")
    print()

    # ── Save JSON report ──────────────────────────────────────────────────────
    report_runs = cold_runs if cold_runs else warm_runs
    totals = [r["total_latency_ms"] for r in report_runs]

    def _sp50(k):
        v = [r[k] for r in report_runs if r.get(k) is not None and r[k] > 0]
        return _p(v, 50) if v else None

    report = {
        "benchmark_meta": {
            "timestamp": ts,
            "mode": mode,
            "cache_mode": cache_mode,
            "dataset": "ai4bharat/MSMARCO-XI",
            "languages": ["hi", "en", "mr"],
            "total_queries_run": len(report_runs),
            "unique_query_count": unique_final,
            "index_source": index_source,
            "sla_target_ms": 200,
            "stt_included": False,
            "llm_included": mode == "extractive",
            "reranker_enabled": reranker.enabled,
            "reranker_actually_loaded": reranker.is_loaded,
            "embedding_model_neural": model_neural,
            "embedding_model": dense.model_name if dense.model else "hash-fallback",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "integrity_warnings": warnings,
            "note": (
                "STT latency excluded -- voice transcription via Sarvam/ElevenLabs "
                "is measured separately through the /api/query/voice endpoint. "
                "ExactCache and SemanticCache are NOT active in this benchmark."
            ),
        },
        "summary": {
            "p50_ms":  _p(totals, 50),
            "p70_ms":  _p(totals, 70),
            "p95_ms":  _p(totals, 95),
            "p99_ms":  _p(totals, 99),
            "p100_ms": round(max(totals), 2),
            "mean_ms": round(sum(totals) / len(totals), 2),
            "sla_200ms_compliance_pct": round(
                100.0 * sum(1 for x in totals if x <= 200) / len(totals), 2),
            "sla_p95_pass": _p(totals, 95) <= 200,
        },
        "stage_p50_breakdown_ms": {
            "embedding":      _sp50("embed_ms"),
            "dense_faiss":    _sp50("dense_ms"),
            "bm25":           _sp50("bm25_ms"),
            "rrf_fusion":     _sp50("rrf_ms"),
            "retrieval_total":_sp50("retrieval_ms"),
            "reranker":       _sp50("reranker_ms"),
            "guardrail":      _sp50("guardrail_ms"),
            "synthesis":      _sp50("llm_ms"),
        },
        "warm_cache_summary": {
            "p50_ms": _p([r["total_latency_ms"] for r in warm_runs], 50),
            "p95_ms": _p([r["total_latency_ms"] for r in warm_runs], 95),
        } if warm_runs else None,
        "sample_runs": report_runs[:15],
    }

    fname = f"benchmark_report_clean_{mode}_{ts}.json"
    out = os.path.join(_REPO_ROOT, fname)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved: {fname}")
    print("=" * 76)

    if report["summary"]["p95_ms"] > 200:
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Clean Voice RAG Latency Benchmark -- ai4bharat/MSMARCO-XI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--queries", type=int, default=100,
        help="Number of queries (default 100). Rule-based augmentation fills gaps above unique count.")
    p.add_argument("--mode", default="pipeline",
        choices=["retrieval", "pipeline", "extractive"],
        help="retrieval=embed+FAISS+BM25+RRF; pipeline=+guardrails+reranker; extractive=+synthesis")
    p.add_argument("--cache", default="both",
        choices=["cold", "warm", "both"],
        help="cold=LRU cleared each query; warm=pre-warmed; both=report both (default)")
    p.add_argument("--require-disk-index", action="store_true",
        help="Exit if disk index not found instead of falling back to in-memory build")
    a = p.parse_args()
    run_benchmark(num_queries=a.queries, mode=a.mode,
                  cache_mode=a.cache, require_disk=a.require_disk_index)
