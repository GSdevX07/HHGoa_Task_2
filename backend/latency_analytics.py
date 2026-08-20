"""
Latency Analytics Engine for Voice RAG Pipeline
================================================
High-precision per-stage timing using time.perf_counter_ns.
Computes P50, P70, P95, P100 latency distributions.

Key fix from original: a single authoritative wall-clock timer wraps the
entire pipeline — stage timings are additive sub-measurements, not separate
clocks that could produce total < sum(stages).
"""

import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("latency_analytics")


class LatencyTracker:
    """
    Per-request latency tracker using monotonic nanosecond clock.

    Usage:
        tracker = LatencyTracker()
        # ... do work ...
        tracker.mark("retrieval")
        # ... do more work ...
        tracker.mark("llm")
        summary = tracker.get_summary()
    """

    def __init__(self):
        self._start_ns = time.perf_counter_ns()
        self._last_mark_ns = self._start_ns
        self._stage_durations: Dict[str, float] = {}

    def mark(self, stage_name: str) -> float:
        """
        Record the end of a stage. Returns the stage duration in ms.
        Stages are measured as time since the previous mark (or start).
        """
        now_ns = time.perf_counter_ns()
        duration_ms = round((now_ns - self._last_mark_ns) / 1_000_000, 3)
        self._stage_durations[stage_name] = duration_ms
        self._last_mark_ns = now_ns
        return duration_ms

    def elapsed_ms(self) -> float:
        """Total elapsed time since tracker creation in milliseconds."""
        return round((time.perf_counter_ns() - self._start_ns) / 1_000_000, 3)

    def get_summary(self) -> Dict[str, float]:
        """Return stage durations plus total elapsed time."""
        return {
            **self._stage_durations,
            "total_elapsed_ms": self.elapsed_ms(),
        }

    def reset(self):
        self._start_ns = time.perf_counter_ns()
        self._last_mark_ns = self._start_ns
        self._stage_durations = {}


class BenchmarkAnalytics:
    """
    Aggregate latency statistics across multiple benchmark query runs.

    Computes: P50, P70, P95, P100, Mean, StdDev, Min, SLA compliance.
    Reports separately for retrieval-only and end-to-end (incl. LLM) pipelines.
    """

    # ── Percentile Calculation ────────────────────────────────────────────────

    @staticmethod
    def calculate_percentiles(
        latencies: List[float],
        sla_target_ms: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Calculate a full latency distribution for a list of measurements.

        Returns:
            sample_count, p50_ms, p70_ms, p95_ms, p100_ms, mean_ms,
            min_ms, std_dev_ms, sla_compliance_pct,
            sla_p50_met, sla_p70_met, sla_p95_met, sla_p100_met,
            raw_samples_ms
        """
        if not latencies:
            return {
                "sample_count": 0,
                "p50_ms": 0.0, "p70_ms": 0.0, "p95_ms": 0.0, "p100_ms": 0.0,
                "mean_ms": 0.0, "min_ms": 0.0, "std_dev_ms": 0.0,
                "sla_compliance_pct": 100.0,
                "sla_p50_met": True, "sla_p70_met": True,
                "sla_p95_met": True, "sla_p100_met": True,
                "raw_samples_ms": [],
            }

        arr = np.array(latencies, dtype=np.float64)
        sorted_arr = np.sort(arr)

        p50  = float(np.percentile(arr, 50))
        p70  = float(np.percentile(arr, 70))
        p95  = float(np.percentile(arr, 95))
        p99  = float(np.percentile(arr, 99))
        p100 = float(np.max(arr))
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        p0   = float(np.min(arr))

        under_sla = int(np.sum(arr <= sla_target_ms))
        compliance = round((under_sla / len(arr)) * 100.0, 2)

        return {
            "sample_count": len(latencies),
            "p50_ms": round(p50, 2),
            "p70_ms": round(p70, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "p100_ms": round(p100, 2),
            "mean_ms": round(mean, 2),
            "min_ms": round(p0, 2),
            "std_dev_ms": round(std, 2),
            "sla_target_ms": sla_target_ms,
            "sla_compliance_pct": compliance,
            # Individual percentile SLA checks (all must pass for full compliance)
            "sla_p50_met": p50 <= sla_target_ms,
            "sla_p70_met": p70 <= sla_target_ms,
            "sla_p95_met": p95 <= sla_target_ms,
            "sla_p99_met": p99 <= sla_target_ms,
            "sla_p100_met": p100 <= sla_target_ms,
            "sla_full_distribution_met": compliance == 100.0,
            "raw_samples_ms": [round(float(x), 2) for x in sorted_arr],
        }

    # ── Benchmark Report Aggregator ───────────────────────────────────────────

    @staticmethod
    def aggregate_benchmark_report(
        runs: List[Dict[str, Any]],
        sla_target_ms: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Aggregate multiple query benchmark runs into a comprehensive report.

        Each run dict must contain at minimum:
            query_id, query, total_latency_ms

        Optional per-stage keys (used for stage breakdown if present):
            retrieval_ms, reranker_ms, llm_ms, guardrail_ms, stt_ms

        Returns a report with:
            - Summary (P50/P70/P95/P100, SLA)
            - Stage breakdowns (retrieval-only, llm-only, etc.)
            - Individual run samples
            - Honest note about what was/wasn't simulated
        """
        if not runs:
            return {"error": "No benchmark runs provided."}

        # ── Extract latency series ────────────────────────────────────────────
        def _series(key: str) -> List[float]:
            return [r[key] for r in runs if key in r and r[key] is not None and r[key] > 0]

        total_lats     = _series("total_latency_ms")
        retrieval_lats = _series("retrieval_ms")
        reranker_lats  = _series("reranker_ms")
        llm_lats       = _series("llm_ms")
        guardrail_lats = _series("guardrail_ms")
        stt_lats       = _series("stt_ms")

        # Pipeline-only latency = retrieval + reranker + guardrails (no LLM, no STT)
        pipeline_only = []
        for r in runs:
            ret = r.get("retrieval_ms", 0.0) or 0.0
            rer = r.get("reranker_ms", 0.0) or 0.0
            grd = r.get("guardrail_ms", 0.0) or 0.0
            pipeline_only.append(ret + rer + grd)

        overall        = BenchmarkAnalytics.calculate_percentiles(total_lats, sla_target_ms)
        pipeline_stats = BenchmarkAnalytics.calculate_percentiles(pipeline_only, sla_target_ms)

        # ── SLA verdict ───────────────────────────────────────────────────────
        p50_met  = overall["sla_p50_met"]
        p70_met  = overall["sla_p70_met"]
        p99_met  = overall.get("sla_p99_met", True)
        p100_met = overall["sla_p100_met"]

        if p50_met and p70_met and p99_met and p100_met:
            sla_verdict = "PASS — full distribution meets sub-200ms SLA"
        elif p50_met and p70_met:
            sla_verdict = "PARTIAL — P50 and P70 meet SLA; tail latency exceeds"
        elif p50_met:
            sla_verdict = "PARTIAL — P50 meets SLA; tail latency exceeds"
        else:
            sla_verdict = "FAIL — P50 exceeds sub-200ms SLA"

        return {
            "benchmark_meta": {
                "total_queries": len(runs),
                "sla_target_ms": sla_target_ms,
                "stt_included": len(stt_lats) > 0,
                "llm_included": len(llm_lats) > 0,
                "reranker_included": len(reranker_lats) > 0,
            },
            "summary": {
                "p50_total_ms": overall["p50_ms"],
                "p70_total_ms": overall["p70_ms"],
                "p95_total_ms": overall["p95_ms"],
                "p99_total_ms": overall.get("p99_ms", overall["p95_ms"]),
                "p100_total_ms": overall["p100_ms"],
                "mean_total_ms": overall["mean_ms"],
                "sla_compliance_pct": overall["sla_compliance_pct"],
                "sla_verdict": sla_verdict,
                "sla_p50_met": p50_met,
                "sla_p70_met": p70_met,
                "sla_p95_met": overall["sla_p95_met"],
                "sla_p99_met": p99_met,
                "sla_p100_met": p100_met,
            },
            "pipeline_only_stats": {
                "description": "Retrieval + Reranker + Guardrails (no STT, no LLM)",
                "p50_ms": pipeline_stats["p50_ms"],
                "p70_ms": pipeline_stats["p70_ms"],
                "p95_ms": pipeline_stats["p95_ms"],
                "p99_ms": pipeline_stats.get("p99_ms", pipeline_stats["p95_ms"]),
                "p100_ms": pipeline_stats["p100_ms"],
            },
            "stage_breakdown": {
                "retrieval_p50_ms": BenchmarkAnalytics.calculate_percentiles(retrieval_lats)["p50_ms"] if retrieval_lats else None,
                "reranker_p50_ms":  BenchmarkAnalytics.calculate_percentiles(reranker_lats)["p50_ms"]  if reranker_lats else None,
                "llm_p50_ms":       BenchmarkAnalytics.calculate_percentiles(llm_lats)["p50_ms"]       if llm_lats else None,
                "guardrail_p50_ms": BenchmarkAnalytics.calculate_percentiles(guardrail_lats)["p50_ms"] if guardrail_lats else None,
                "stt_p50_ms":       BenchmarkAnalytics.calculate_percentiles(stt_lats)["p50_ms"]       if stt_lats else None,
            },
            "detailed_stats": overall,
            "sample_runs": runs[:10],  # First 10 runs for inspection
        }

    # ── Integrity Check ───────────────────────────────────────────────────────

    @staticmethod
    def integrity_check(
        dense_model_loaded: bool,
        reranker_loaded: bool,
        reranker_enabled: bool,
        index_source: str,
        unique_query_count: int,
        total_query_count: int,
        cache_cleared: bool,
    ) -> List[str]:
        """
        Inspect a benchmark configuration and return a list of human-readable
        warnings about measurement validity.

        Returns an empty list if the configuration is clean.
        """
        warnings: List[str] = []

        if not dense_model_loaded:
            warnings.append(
                "CRITICAL: SentenceTransformer model NOT loaded — using hash-based "
                "fallback embedding. All latency numbers are INVALID for neural retrieval."
            )

        if reranker_enabled and not reranker_loaded:
            warnings.append(
                "WARNING: Reranker was enabled but failed to load — fell back to RRF ranking. "
                "Reranker latency numbers are RRF latency, not BGE cross-encoder latency."
            )

        if not reranker_enabled:
            warnings.append(
                "NOTE: Reranker is DISABLED (RERANKER_ENABLED=false). "
                "Reranker latency = RRF fallback only. BGE cross-encoder was NOT benchmarked."
            )

        if index_source == "in-memory":
            warnings.append(
                "WARNING: Index was built in-memory from corpus (not loaded from disk). "
                "This may differ from the production index configuration."
            )
        elif index_source == "builtin_samples":
            warnings.append(
                "CRITICAL: Running on built-in sample corpus. This is DEMO MODE, "
                "not a production benchmark."
            )

        if not cache_cleared:
            warnings.append(
                "NOTE: Dense LRU query cache was NOT cleared between queries. "
                "Results include cache-hit speedups and are NOT cold-cache latency."
            )

        if unique_query_count < total_query_count:
            repeat_factor = round(total_query_count / unique_query_count, 1)
            warnings.append(
                f"NOTE: Only {unique_query_count} unique queries available; "
                f"padded to {total_query_count} ({repeat_factor}x repetition). "
                f"Repeated queries may benefit from LRU cache even after initial clear."
            )

        return warnings

