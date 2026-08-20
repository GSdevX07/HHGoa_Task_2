# Benchmark Methodology — HH Goa 2026 Task 2: Voice RAG

## What Is Measured

This project reports **four distinct latency metrics**, each clearly separated:

| Metric | What it covers | Script / Endpoint |
|--------|----------------|-------------------|
| **Retrieval latency** | Embedding + Dense FAISS + BM25 + RRF | `benchmark_clean.py --mode retrieval` |
| **Pipeline latency** | Retrieval + Guardrails + Reranker | `benchmark_clean.py --mode pipeline` |
| **Full non-LLM latency** | Pipeline + Extractive synthesis | `benchmark_clean.py --mode extractive` |
| **Voice E2E latency** | STT + full pipeline (server-side) | Live `/api/query/voice` endpoint |

## What Is NOT Included in the Benchmark Numbers

The `benchmark_clean.py` results **explicitly exclude**:

- **STT (Speech-to-Text)**: Sarvam AI / ElevenLabs transcription is an external HTTP call.
  Measured separately via the live `/api/query/voice` endpoint during demo.
- **ExactCache / SemanticCache**: The application-layer caches in `main.py` are NOT
  instantiated during benchmarking. Results show raw retrieval latency, not cached lookups.
- **Network upload time**: Browser-to-server audio transfer is user/network dependent.

## Dataset

**ai4bharat/MSMARCO-XI** — MS MARCO translated into Indic languages.

- Languages benchmarked: **Hindi (hi), English (en), Marathi (mr)**
- Queries sourced from the validation split (`Eng_Query` and native `query` fields)
- Rule-based paraphrase augmentation used to reach ≥ 100 unique surface-form variants
  (no LLM, no external API — deterministic transforms only)

## Cache Control

Two separate runs are always reported:

| Run | Description | Use case |
|-----|-------------|----------|
| **COLD-CACHE** | Dense LRU cache cleared before every single query | Worst-case / honest latency |
| **WARM-CACHE** | All queries pre-run once, then measured with hot cache | Steady-state production latency |

The cold-cache number is the primary SLA claim.
The warm-cache number shows the benefit of the LRU query cache in production.

## Reranker State

The BGE cross-encoder (`BAAI/bge-reranker-v2-m3`) is **disabled** in both the
benchmark and the default production configuration (`RERANKER_ENABLED=false`).
When disabled, the pipeline falls back to RRF ranking (fast, no model inference).

To benchmark WITH the reranker enabled, set `RERANKER_ENABLED=true` before running.
Expect P50 to increase by ~200–800 ms on CPU.

## Embedding Model

Model: `all-MiniLM-L6-v2` (SentenceTransformers, 384-dim, CPU-optimized)

The benchmark detects if the model failed to load and fell back to the hash-based
fallback embedding. If fallback is active, results are marked **INVALID** in the
integrity warnings section of the JSON report.

## Index Source

| Source | Description |
|--------|-------------|
| `disk` | Pre-built FAISS + BM25 index loaded from `indexes/` — production configuration |
| `in-memory` | Index built from `corpus.jsonl` at benchmark start — same data, slightly different path |

The `--require-disk-index` flag forces the benchmark to exit if the disk index is missing.

## How to Reproduce

```bash
# 1. Download corpus (Hindi, English, Marathi — 100 each)
python scripts/download_dataset.py --langs hi,en,mr --limit 100

# 2. Build disk index
python scripts/build_index.py

# 3. Run clean benchmark (pipeline mode, cold + warm cache, 100 queries)
python benchmark_clean.py --mode pipeline --queries 100 --cache both

# 4. Run retrieval-only benchmark (pure SLA metric)
python benchmark_clean.py --mode retrieval --queries 100 --cache cold
```

## What the Old Numbers Mean (for reference)

The old `benchmark_report.json` (P50=0.15ms) and `benchmark_report_extractive.json`
(P50=27.69ms) are **NOT valid system latency figures**. They are artifacts of:

- Dense LRU query cache returning immediately for repeated corpus queries
- ExactCache pre-warmed with all corpus queries at startup
- Reranker disabled (0.01ms = RRF fallback, not BGE inference)
- In-memory index (not production disk index)
- Repeated queries (8–25 unique × N repetitions)

These files are retained for historical reference only and should not be cited
in presentations or submissions.
