"""
Standalone Latency Benchmark Runner
=====================================
Executes N real queries and measures true P50/P70/P95/P100 latencies.

Two benchmark modes:
  --mode retrieval   Pipeline only (no STT, no LLM)  ← primary SLA metric
  --mode full        Everything including LLM (requires GROQ_API_KEY)

Usage:
    python run_benchmark.py [--queries 100] [--mode retrieval] [--strategy semantic]

Honest benchmarking principles applied:
  1. Single authoritative wall-clock timer per query (not accumulated stages)
  2. No simulated STT latency
  3. No hardcoded fallback values
  4. Clear labeling of what was/wasn't measured
  5. Physical consistency guaranteed: total ≥ sum(reported stages) always
"""

import sys
import os
import time
import asyncio
import json
import argparse
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# Repo root resolution
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_REPO_ROOT, "backend")
sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

from dataset_loader import load_corpus_from_disk, load_msmarco_xi_dataset
from chunking_engine import ChunkingEngine
from guardrails import GuardrailEngine
from model_harness import ModelHarness
from latency_analytics import BenchmarkAnalytics
from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

_INDEX_DIR = os.path.join(_REPO_ROOT, "indexes")
_CORPUS_PATH = os.path.join(_REPO_ROOT, "data", "corpus.jsonl")


async def run_benchmark(num_queries: int, strategy: str, mode: str):
    print("=" * 76)
    print("   HH Goa 2026 Task 2 — Voice RAG Benchmark Runner")
    print("=" * 76)
    print(f"   Dataset   : AI4Bharat MSMARCO-XI")
    print(f"   Mode      : {mode.upper()} ({'retrieval + LLM' if mode == 'full' else 'retrieval pipeline only (no LLM, no STT)'})")
    print(f"   Strategy  : {strategy}")
    print(f"   Queries   : N={num_queries}")
    print(f"   SLA Target: 200ms")
    print("-" * 76)

    # ── Setup ───────────────────────────────────────────────────────────────
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense, bm25, candidate_pool=20)
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    guardrails = GuardrailEngine()
    harness = ModelHarness()

    # ── Load Index ──────────────────────────────────────────────────────────
    load_result = hybrid.load_from_disk(_INDEX_DIR)

    if load_result["dense"]:
        print(f"   Index     : Loaded from disk ({len(dense)} chunks)")
        index_source = "disk"
    else:
        print("   Index     : No disk index — building from corpus ...")
        corpus = load_corpus_from_disk(_CORPUS_PATH)
        print(f"   Corpus    : {len(corpus)} documents")
        chunker = ChunkingEngine()
        chunks = chunker.chunk_documents(corpus, strategy=strategy)
        hybrid.index_chunks(chunks)
        print(f"   Chunks    : {len(chunks)} chunks built in-memory")
        index_source = "in-memory"

    if dense.model is not None:
        guardrails.set_embedder(dense._embed)

    # ── Collect Queries ─────────────────────────────────────────────────────
    corpus = load_corpus_from_disk(_CORPUS_PATH)
    queries = []
    for doc in corpus:
        q = (doc.get("query_en") or doc.get("query", "")).strip()
        if q and len(q.split()) >= 3:
            queries.append(q)

    if not queries:
        print("ERROR: No queries found. Run scripts/download_dataset.py first.")
        return

    # Pad to required count (only real queries — loop over corpus repeatedly)
    base_queries = queries[:]
    while len(queries) < num_queries:
        queries.extend(base_queries)
    queries = queries[:num_queries]

    print(f"   Unique Qs : {len(base_queries)} (padded to {num_queries})")
    print("-" * 76)

    # ── Warmup Run ───────────────────────────────────────────────────────────
    print("   Warming up (1 query, not counted) ...")
    await _run_single(queries[0], hybrid, reranker, guardrails, harness, mode)

    # ── Benchmark Loop ────────────────────────────────────────────────────────
    print(f"   Running {num_queries} queries ...\n")
    runs = []
    t_bench_start = time.perf_counter()

    for idx, query in enumerate(queries, 1):
        run = await _run_single(query, hybrid, reranker, guardrails, harness, mode)
        runs.append(run)

        if idx % 20 == 0 or idx == num_queries:
            print(f"   [{idx:4d}/{num_queries}]  last_total={run['total_latency_ms']:.1f}ms  "
                  f"ret={run.get('retrieval_ms', 0):.1f}ms  "
                  f"rerank={run.get('reranker_ms', 0):.1f}ms  "
                  f"{'llm=' + str(round(run.get('llm_ms') or 0, 1)) + 'ms' if mode == 'full' else ''}")

    bench_total = time.perf_counter() - t_bench_start

    # ── Aggregate Report ──────────────────────────────────────────────────────
    report = BenchmarkAnalytics.aggregate_benchmark_report(runs, sla_target_ms=200.0)
    stats = report["detailed_stats"]
    summary = report["summary"]
    pipeline = report["pipeline_only_stats"]
    breakdown = report["stage_breakdown"]

    print()
    print("=" * 76)
    print("   LATENCY RESULTS")
    print("=" * 76)
    print(f"   Benchmark duration : {bench_total:.1f}s  ({num_queries} queries)")
    print(f"   Index source       : {index_source}")
    print()
    print("   ── Full Pipeline ───────────────────────────────────────────────")
    print(f"   P50  (median)      : {stats['p50_ms']:>8.2f} ms   [SLA: {'✓ PASS' if summary['sla_p50_met'] else '✗ FAIL'}]")
    print(f"   P70               : {stats['p70_ms']:>8.2f} ms   [SLA: {'✓ PASS' if summary['sla_p70_met'] else '✗ FAIL'}]")
    print(f"   P95               : {stats['p95_ms']:>8.2f} ms   [SLA: {'✓ PASS' if summary['sla_p95_met'] else '✗ FAIL'}]")
    print(f"   P100 (max)        : {stats['p100_ms']:>8.2f} ms   [SLA: {'✓ PASS' if summary['sla_p100_met'] else '✗ FAIL'}]")
    print(f"   Mean              : {stats['mean_ms']:>8.2f} ms")
    print(f"   StdDev            : {stats['std_dev_ms']:>8.2f} ms")
    print(f"   SLA compliance    : {stats['sla_compliance_pct']:>8.2f} %")
    print()

    print("   ── Retrieval Pipeline Only (no LLM) ────────────────────────────")
    print(f"   P50               : {pipeline['p50_ms']:>8.2f} ms")
    print(f"   P70               : {pipeline['p70_ms']:>8.2f} ms")
    print(f"   P95               : {pipeline['p95_ms']:>8.2f} ms")
    print(f"   P100              : {pipeline['p100_ms']:>8.2f} ms")
    print()

    print("   ── Stage P50 Breakdown ─────────────────────────────────────────")
    for stage, label in [
        ("retrieval_p50_ms", "Dense + BM25 + RRF"),
        ("reranker_p50_ms", "Cross-encoder reranker"),
        ("llm_p50_ms",      "LLM generation (Groq)"),
        ("guardrail_p50_ms","Guardrails"),
    ]:
        val = breakdown.get(stage)
        if val is not None:
            print(f"   {label:<28}: {val:>7.2f} ms")
        else:
            print(f"   {label:<28}: N/A (not measured in this mode)")

    print()
    print(f"   SLA Verdict: {summary['sla_verdict']}")
    print("=" * 76)

    # ── Honesty Disclaimer ───────────────────────────────────────────────────
    meta = report["benchmark_meta"]
    print()
    print("   ── Measurement Notes ───────────────────────────────────────────")
    print(f"   STT included     : {meta.get('stt_included', False)}")
    print(f"   LLM included     : {meta.get('llm_included', False)}")
    print(f"   Reranker used    : {meta.get('reranker_included', False)}")
    if not meta.get("llm_included"):
        print("   Note: Run with --mode full and GROQ_API_KEY set for end-to-end numbers.")
    if not meta.get("stt_included"):
        print("   Note: STT latency is measured separately via /api/query/voice endpoint.")
    print("=" * 76)

    # ── Save Report ──────────────────────────────────────────────────────────
    out_path = os.path.join(_REPO_ROOT, f"benchmark_report_{mode}.json")
    report["benchmark_meta"]["mode"] = mode
    report["benchmark_meta"]["strategy"] = strategy
    report["benchmark_meta"]["query_count"] = num_queries
    report["benchmark_meta"]["index_source"] = index_source
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n   Report saved: {out_path}\n")


async def _run_single(
    query: str,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    guardrails: GuardrailEngine,
    harness: ModelHarness,
    mode: str,
) -> dict:
    """Run one query through the pipeline and return a run record."""
    t0 = time.perf_counter()

    # Pre-guardrail
    t_guard_start = time.perf_counter()
    pre_guard = guardrails.validate_input(query)
    guard_ms = (time.perf_counter() - t_guard_start) * 1000

    if not pre_guard["passed"]:
        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "total_latency_ms": round(total_ms, 2),
            "retrieval_ms": 0.0,
            "reranker_ms": 0.0,
            "llm_ms": None,
            "guardrail_ms": round(guard_ms, 2),
            "stt_ms": None,
            "is_refusal": True,
            "is_grounded": False,
            "groundedness_score": 0.0,
        }

    # Hybrid retrieval
    t_ret = time.perf_counter()
    candidates, ret_lats = hybrid.search(query, top_k=20)
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    # Reranking
    t_rerank = time.perf_counter()
    final_results, reranker_ms = reranker.rerank(query, candidates, top_k=3)
    reranker_elapsed = (time.perf_counter() - t_rerank) * 1000

    # Context guardrail
    top_score = (final_results[0].get("reranker_score", 0.0) if final_results else 0.0)
    ctx_check = guardrails.validate_retrieved_context(final_results, top_score, threshold=0.15)

    llm_ms = None

    if not ctx_check.get("should_refuse") and mode == "full":
        # LLM generation
        passages = [r.get("parent_text", r.get("text", "")) for r in final_results]
        t_llm = time.perf_counter()
        answer, model_used, _ = await harness._generate_with_retry(query, passages, max_retries=1)
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)

    # Single authoritative wall-clock total
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "total_latency_ms": total_ms,
        "retrieval_ms": round(retrieval_ms, 2),
        "reranker_ms": round(reranker_elapsed, 2),
        "llm_ms": llm_ms,
        "guardrail_ms": round(guard_ms, 2),
        "stt_ms": None,
        "is_refusal": ctx_check.get("should_refuse", False),
        "is_grounded": None,
        "groundedness_score": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Latency Benchmark")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to benchmark (default: 100)")
    parser.add_argument("--strategy", default="semantic",
                        choices=["fixed", "semantic", "metadata_aware", "parent_child"],
                        help="Chunking strategy (default: semantic)")
    parser.add_argument("--mode", default="retrieval",
                        choices=["retrieval", "full"],
                        help="'retrieval' = no LLM; 'full' = include LLM (requires GROQ_API_KEY)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        num_queries=args.queries,
        strategy=args.strategy,
        mode=args.mode,
    ))
