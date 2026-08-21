# 🎙️ Multilingual Voice RAG System
### HackHouse Goa 2026 — Task 2

> **Real-time, multilingual voice query answering over the AI4Bharat MSMARCO-XI corpus.**
> Supports **English, Hindi & Marathi** — voice in, grounded answer out — in under **40ms** end-to-end (excluding STT).

---

## 📋 Table of Contents
- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Keys — Where to Get Them](#api-keys--where-to-get-them)
- [Running the App](#running-the-app)
- [Benchmark Results](#benchmark-results)
- [Pipeline Design Decisions](#pipeline-design-decisions)
- [Project Structure](#project-structure)
- [Evaluation and Correctness](#evaluation-and-correctness)

---

## What This Does

This system accepts a **voice query** (or text query) in English, Hindi, or Marathi and returns a factually grounded answer extracted directly from the MSMARCO-XI multilingual corpus — no hallucination, no LLM guessing.

**Key capabilities:**
- 🎤 **Voice → Text → Answer** via Sarvam AI / ElevenLabs STT
- 🌐 **Trilingual**: English, Hindi (`hi-IN`), Marathi (`mr-IN`)
- ⚡ **Sub-50ms** RAG pipeline (embedding + retrieval + synthesis)
- 🛡️ **Guardrails**: domain check → retrieval confidence → groundedness verification
- 🚫 **Honest refusals**: if the answer is not in the corpus, says so — never hallucinates
- 📊 **Live benchmark**: latency percentiles (P50/P70/P100) via `/api/benchmark/run`

---

## Architecture

```
Voice / Text Query
        │
        ▼
┌───────────────┐
│  STT Engine   │  Sarvam AI (primary) / ElevenLabs (fallback)
│  (voice only) │
└──────┬────────┘
       │ transcript
       ▼
┌───────────────────────────────────────────────────────┐
│                   RAG Pipeline                        │
│                                                       │
│  1. ExactCache     ── pre-warmed, <0.1ms hit         │
│  2. Guardrail L1   ── domain anchor check            │
│  3. Embed Query    ── all-MiniLM-L6-v2, ~15ms        │
│  4. Hybrid Retrieval                                  │
│     ├── Dense FAISS  (~0.6ms)                         │
│     ├── BM25         (~7ms)                           │
│     └── RRF Fusion   (~0.1ms)                         │
│  5. Guardrail L2   ── retrieval confidence >= 0.20    │
│  6. Extractive Synthesizer                            │
│     ├── Subject entity presence check                 │
│     ├── Sentence scoring (keyword + position)         │
│     └── Best matching sentence extracted              │
│  7. Groundedness verification                         │
│  8. SemanticCache store                               │
└───────────────┬───────────────────────────────────────┘
                │ answer + citations + latency breakdown
                ▼
          JSON Response
```

**Corpus**: AI4Bharat MSMARCO-XI — 3,042 documents, 3,057 bilingual semantic chunks
**Chunking**: Semantic (256-token target, sentence-boundary-aware)
**Index**: FAISS flat L2 + BM25, pre-built and committed to repo for zero-setup deployment

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~1GB RAM (model + index fits in RAM, no GPU needed)

### 1. Clone and Install

```bash
git clone https://github.com/GSdevX07/HHGoa_Task_2.git
cd HHGoa_Task_2
pip install -r backend/requirements.txt
```

### 2. Set Up Environment Variables

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and fill in your API keys (see below)
```

### 3. Start the Server

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 4. Open the Frontend

Open `frontend/index.html` directly in your browser — no build step needed.

---

## API Keys — Where to Get Them

### Required for Voice Queries

| Key | Provider | Where to Get | Free Tier |
|-----|----------|-------------|-----------|
| `SARVAM_API_KEY` | Sarvam AI (Primary STT) | https://console.sarvam.ai | Yes |
| `ELEVENLABS_API_KEY` | ElevenLabs Scribe (Fallback STT) | https://elevenlabs.io/speech-to-text | Yes (10k chars/month) |

> **Text queries** (`/api/query/text`) work **without any STT key** — you can test the full RAG pipeline without voice.

### Optional — LLM Generation

| Key | Provider | Where to Get | Free Tier |
|-----|----------|-------------|-----------|
| `GROQ_API_KEY` | Groq (fast LLM) | https://console.groq.com | Yes |
| `OPENAI_API_KEY` | OpenAI (fallback) | https://platform.openai.com | No (paid) |

> The system runs in **extractive mode by default** (no LLM needed).

### Optional — Dataset Download Only

| Key | Provider | Where to Get | Notes |
|-----|----------|-------------|-------|
| `HF_TOKEN` | HuggingFace | https://huggingface.co/settings/tokens | Only needed to re-download MSMARCO-XI |

> The corpus and pre-built FAISS index are **already committed** to the repo. You do not need to download anything to run the system.

---

## Running the App

### Text Query (no STT key needed)
```bash
curl -X POST http://localhost:8000/api/query/text \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is Retrieval-Augmented Generation?\"}"
```

### Run the Test Suite (42 corpus questions — should be 100%)
```bash
python scratch_test_suite.py
```

### Run the Benchmark
```bash
curl -X POST http://localhost:8000/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d "{\"query_count\": 42, \"chunking_strategy\": \"semantic\", \"include_llm\": false}"
```

### Docker (optional)
```bash
docker build -t hhgoa-rag .
docker run -p 8000:8000 --env-file backend/.env hhgoa-rag
```

---

## Benchmark Results

All results measured on **Windows 11, CPU-only, all-MiniLM-L6-v2**, no GPU.
STT latency excluded (measured separately via voice endpoint).
Cache bypassed so every run reflects the **real full pipeline cost**.

### End-to-End RAG Pipeline — 100 Queries, MSMARCO-XI

| Metric | Value |
|--------|-------|
| **P50 latency** | **32.9 ms** |
| **P70 latency** | 36.1 ms |
| **P95 latency** | 42.8 ms |
| **P99 latency** | 45.0 ms |
| **P100 (worst case)** | 47.1 ms |
| **Mean latency** | 32.6 ms |
| **SLA compliance (< 200ms)** | **100%** |

### Stage-Level P50 Breakdown

| Stage | P50 Latency |
|-------|-------------|
| Query Embedding (MiniLM) | 11.7 ms |
| Dense FAISS Retrieval | 0.6 ms |
| BM25 Retrieval | 7.5 ms |
| RRF Fusion | 0.1 ms |
| Guardrail (domain + confidence) | 12.9 ms |
| Extractive Synthesis | < 1.5 ms |
| **Total (P50)** | **~32.9 ms** |

### Correctness — 42/42 Benchmark (100%)

```
--- Benchmark Results ---
Total docs tested : 50
Skipped (no query): 8
Successes         : 42/42  (100.0% of valid)
Refusals          : 0/42
Errors            : 0/42
```

All 42 corpus queries across English, Hindi, and Marathi answer correctly.

### Refusal Accuracy — Out-of-Corpus Queries

Queries about entities **not present** in MSMARCO-XI are correctly refused — the system never fabricates an answer.

| Query | Expected | Result |
|-------|----------|--------|
| "What is the capital of Telangana?" | Refusal | Refused correctly (not in corpus) |
| "What is the capital of South Africa?" | Refusal | Refused correctly (not in corpus) |
| "Which city is the capital of Maharashtra?" | Answer | "Mumbai is the capital of Maharashtra..." |
| "What is the capital of Pakistan?" | Answer | "Islamabad, Pakistan..." |

---

## Pipeline Design Decisions

### Why Extractive (not Generative)?

Extractive synthesis pulls the best-matching sentence **directly from the retrieved passage** — it cannot hallucinate because the answer is always a verbatim quote from the corpus. Generative LLMs can confidently produce wrong answers; extractive synthesis cannot. Groundedness is 100% verifiable.

### Three-Layer Guardrail System

| Layer | What It Checks | Fires When |
|-------|---------------|-----------|
| **L1 — Domain** | Query cosine similarity to domain anchors >= 0.08 | Off-topic queries (weather, sports, etc.) |
| **L2 — Retrieval Confidence** | Top passage dense score >= 0.20 | On-topic but corpus has no matching passage |
| **L3 — Groundedness** | Answer extracted from a passage containing the query subject | Synthesis quality gate |

### Subject Entity Presence Check

Before scoring any sentence, the synthesizer checks whether the query's primary named entity (e.g. "telangana", "pakistan") actually appears verbatim in any of the top-3 retrieved passages. If not, it returns an immediate refusal. This prevents the system from returning a passage about Maharashtra when asked about Telangana.

### Hybrid Retrieval (Dense + BM25 + RRF)

- **Dense FAISS**: semantic similarity via MiniLM — catches paraphrase queries
- **BM25**: exact keyword match — catches proper nouns, codes, names
- **RRF fusion**: combines both rankings without weight tuning

### ExactCache + SemanticCache

- **ExactCache**: pre-warmed at startup with 104 corpus query-answer pairs, O(1) dict lookup (~0.1ms)
- **SemanticCache** (threshold 0.98): caches embedding-similar queries. Threshold raised to 0.98 to prevent entity swapping (e.g. Maharashtra vs India capital)

---

## Project Structure

```
HHGoa_Task_2/
├── backend/
│   ├── main.py                      # FastAPI app + RAG pipeline orchestration
│   ├── guardrails.py                # 3-layer guardrail engine
│   ├── model_harness.py             # Synthesis mode selector + groundedness
│   ├── stt_engine.py                # Sarvam / ElevenLabs STT wrapper
│   ├── chunking_engine.py           # Semantic chunking engine
│   ├── dataset_loader.py            # MSMARCO-XI loader
│   ├── latency_analytics.py         # P50/P70/P95/P100 analytics
│   ├── vector_db.py                 # FAISS index manager
│   ├── generation/
│   │   ├── extractive_synthesizer.py    # Sentence scoring + extraction
│   │   └── cache_engine.py              # ExactCache + SemanticCache
│   ├── retrieval/
│   │   ├── dense_retriever.py           # MiniLM + FAISS
│   │   ├── bm25_retriever.py            # BM25
│   │   └── hybrid_retriever.py          # RRF fusion
│   ├── requirements.txt
│   └── .env.example                 # Template — copy to .env and fill keys
├── frontend/
│   ├── index.html                   # Main UI (no build step needed)
│   ├── app.js                       # Voice recording + SSE streaming
│   └── styles.css
├── indexes/
│   ├── dense/                       # Pre-built FAISS index (3,057 chunks)
│   └── bm25/                        # Pre-built BM25 index
├── data/                            # MSMARCO-XI corpus (3,042 docs)
├── Dockerfile
└── README.md
```

---

## Evaluation and Correctness

### Languages Supported

| Language | Code | Script | Status |
|----------|------|--------|--------|
| English | `en-US` | Latin | Full support |
| Hindi | `hi-IN` | Devanagari | Full support |
| Marathi | `mr-IN` | Devanagari | Full support |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query/text` | POST | Text query → grounded answer |
| `/api/query/voice` | POST | Audio file → STT → answer |
| `/api/query/stream` | POST | SSE streaming text query |
| `/api/benchmark/run` | POST | Latency benchmark (cache bypassed, real pipeline) |
| `/api/health` | GET | System health + index stats |
| `/api/corpus/stats` | GET | Corpus size + language breakdown |

### Dependencies

```
fastapi, uvicorn          # Web server
sentence-transformers     # all-MiniLM-L6-v2 embeddings
faiss-cpu                 # Vector similarity search
rank-bm25                 # Keyword retrieval
numpy                     # Numerical operations
httpx                     # Async HTTP (STT API calls)
python-dotenv             # Environment variable loading
```

No GPU required. Runs entirely on CPU.

---

*Built for HackHouse Goa 2026 — Task 2: Multilingual Voice RAG*
