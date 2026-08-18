"""
Latency Analytics Engine for RAG Pipeline
Measures per-stage duration using high-precision monotonic clocks (time.perf_counter_ns).
Computes exact P50, P70, and P100 latency distributions across benchmark query suites.
"""

import time
import math
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("latency_analytics")

class LatencyTracker:
    """Tracks latency metrics for individual query executions."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_ns = time.perf_counter_ns()
        self.stage_timestamps = {}
        self.stage_durations = {}

    def mark_stage(self, stage_name: str):
        """Marks completion timestamp of a pipeline stage."""
        now = time.perf_counter_ns()
        prev_ts = max(self.stage_timestamps.values()) if self.stage_timestamps else self.start_ns
        duration_ms = round((now - prev_ts) / 1_000_000, 3)
        
        self.stage_timestamps[stage_name] = now
        self.stage_durations[stage_name] = duration_ms

    def get_summary(self) -> Dict[str, float]:
        """Returns stage latency breakdown and total latency in milliseconds."""
        now = time.perf_counter_ns()
        total_ms = round((now - self.start_ns) / 1_000_000, 3)
        return {
            **self.stage_durations,
            "total_latency_ms": total_ms
        }

class BenchmarkAnalytics:
    """Computes statistical P50 / P70 / P100 latency metrics across benchmark runs."""

    @staticmethod
    def calculate_percentiles(latencies: List[float]) -> Dict[str, Any]:
        """
        Calculates P50 (median), P70, P100 (max), Mean, Min, and SLA Compliance (< 200ms).
        """
        if not latencies:
            return {
                "sample_count": 0,
                "p50_ms": 0.0,
                "p70_ms": 0.0,
                "p100_ms": 0.0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "sla_target_200ms_compliance_pct": 100.0
            }

        sorted_lats = sorted(latencies)
        n = len(sorted_lats)

        # Percentile computation (Interpolated)
        p50 = float(np.percentile(sorted_lats, 50))
        p70 = float(np.percentile(sorted_lats, 70))
        p100 = float(np.max(sorted_lats))
        p0 = float(np.min(sorted_lats))
        mean_val = float(np.mean(sorted_lats))
        std_val = float(np.std(sorted_lats))

        # Under 200ms SLA calculation
        under_200 = sum(1 for x in sorted_lats if x <= 200.0)
        compliance_pct = round((under_200 / n) * 100.0, 2)

        return {
            "sample_count": n,
            "p50_ms": round(p50, 2),
            "p70_ms": round(p70, 2),
            "p100_ms": round(p100, 2),
            "mean_ms": round(mean_val, 2),
            "min_ms": round(p0, 2),
            "std_dev_ms": round(std_val, 2),
            "sla_target_200ms_compliance_pct": compliance_pct,
            "raw_samples_ms": [round(x, 2) for x in sorted_lats]
        }

    @staticmethod
    def aggregate_benchmark_report(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates multiple query runs into an overall latency benchmark report.
        Includes total latency percentiles and breakdown per stage.
        """
        if not runs:
            return {"error": "No benchmark runs provided."}

        total_latencies = [r["total_latency_ms"] for r in runs]
        stt_latencies = [r.get("stt_latency_ms", 0.0) for r in runs]
        retrieval_latencies = [r.get("retrieval_latency_ms", 0.0) for r in runs]
        harness_latencies = [r.get("harness_latency_ms", 0.0) for r in runs]

        overall_stats = BenchmarkAnalytics.calculate_percentiles(total_latencies)

        return {
            "summary": {
                "total_queries_tested": len(runs),
                "p50_total_latency_ms": overall_stats["p50_ms"],
                "p70_total_latency_ms": overall_stats["p70_ms"],
                "p100_total_latency_ms": overall_stats["p100_ms"],
                "mean_total_latency_ms": overall_stats["mean_ms"],
                "sub_200ms_target_met": overall_stats["p50_ms"] <= 200.0,
                "sla_compliance_pct": overall_stats["sla_target_200ms_compliance_pct"]
            },
            "stage_breakdown": {
                "stt_p50_ms": BenchmarkAnalytics.calculate_percentiles(stt_latencies)["p50_ms"],
                "retrieval_p50_ms": BenchmarkAnalytics.calculate_percentiles(retrieval_latencies)["p50_ms"],
                "harness_p50_ms": BenchmarkAnalytics.calculate_percentiles(harness_latencies)["p50_ms"]
            },
            "detailed_stats": overall_stats,
            "individual_runs": runs[:10] # Top 10 sample traces
        }
