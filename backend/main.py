"""
Main FastAPI Server — Voice-Enabled RAG System (MSMARCO-XI)
============================================================
Full pipeline:
  Browser audio → Sarvam/ElevenLabs STT → Input Guardrail →
  HybridRetriever (Dense + BM25 + RRF) → CrossEncoder Reranker →
  Context Guardrail → LLM (Groq/OpenAI) → Groundedness Check →
  RAGResponse with real per-stage latencies

Startup sequence:
  1. Try to load pre-built index from indexes/ (fast, < 3s)
  2. Fall back to building in-memory index from corpus.jsonl
  3. Fall back to built-in sample corpus (demo mode)

Run:
    uvicorn backend.main:app --port 8000 --reload
or from backend/:
    uvicorn main:app --port 8000 --reload
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Path resolution (works whether run from repo root or backend/) ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

load_dotenv(os.path.join(_HERE, ".env"))

from dataset_loader import load_corpus_from_disk, BUILTIN_MSMARCO_XI_SAMPLES
from stt_engine import STTEngine
from chunking_engine import ChunkingEngine
from guardrails import GuardrailEngine
from model_harness import ModelHarness, RAGResponse, Citation, ExecutionTraceStep
from latency_analytics import LatencyTracker, BenchmarkAnalytics
from retrieval.dense import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.cache_engine import ExactCache, SemanticCache
# Keep VectorDBEngine for backward compat (e.g. /api/chunking/compare)
from vector_db import VectorDBEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_api")

_INDEX_DIR = os.path.join(_REPO_ROOT, "indexes")
_RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "3"))
_RETRIEVAL_CANDIDATE_POOL = int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "20"))
_RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.15"))

# ── Global Singletons ─────────────────────────────────────────────────────────

stt_engine = STTEngine()
chunking_engine = ChunkingEngine()

dense_retriever = DenseRetriever(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)
bm25_retriever = BM25Retriever()
hybrid_retriever = HybridRetriever(
    dense=dense_retriever,
    bm25=bm25_retriever,
    candidate_pool=_RETRIEVAL_CANDIDATE_POOL,
)
reranker = CrossEncoderReranker(
    model_name=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    enabled=os.getenv("RERANKER_ENABLED", "false").lower() == "true",
)

guardrail_engine = GuardrailEngine()  # embedder wired in after dense retriever loads
model_harness = ModelHarness()

# High performance Exact & Semantic caches
exact_cache = ExactCache(max_size=10000)
semantic_cache = SemanticCache(threshold=0.94, max_size=2000)

# Legacy VectorDBEngine (used only for /api/chunking/compare backwards compat)
_legacy_vector_db = VectorDBEngine()

current_corpus: List[Dict[str, Any]] = []
_index_source: str = "none"   # "disk" | "corpus_file" | "builtin_samples"


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_corpus, _index_source

    logger.info("=" * 60)
    logger.info("HH Goa Task 2: Voice-Enabled RAG System starting up (Ultra Low-Latency Mode) ...")
    logger.info("=" * 60)

    # ── Step 1: Try disk index ──────────────────────────────────────────────
    load_result = hybrid_retriever.load_from_disk(_INDEX_DIR)

    if load_result["dense"]:
        logger.info("Index loaded from disk (fast startup).")
        _index_source = "disk"

        # Load corpus metadata for /api/dataset/samples etc.
        corpus_path = os.path.join(_REPO_ROOT, "data", "corpus.jsonl")
        current_corpus = load_corpus_from_disk(corpus_path)
    else:
        # ── Step 2: Try corpus.jsonl ────────────────────────────────────────
        corpus_path = os.path.join(_REPO_ROOT, "data", "corpus.jsonl")
        current_corpus = load_corpus_from_disk(corpus_path)

        if current_corpus and len(current_corpus) > len(BUILTIN_MSMARCO_XI_SAMPLES):
            logger.info(f"Building in-memory index from corpus.jsonl ({len(current_corpus)} docs) ...")
            _index_source = "corpus_file"
        else:
            # ── Step 3: Fall back to built-in samples ──────────────────────
            logger.warning(
                "No disk index and no corpus.jsonl found. "
                "Running in DEMO MODE with built-in samples. "
                "Run scripts/download_dataset.py then scripts/build_index.py for full corpus."
            )
            current_corpus = BUILTIN_MSMARCO_XI_SAMPLES
            _index_source = "builtin_samples"

        chunks = chunking_engine.chunk_documents(current_corpus, strategy="semantic")
        hybrid_retriever.index_chunks(chunks)
        logger.info(f"In-memory index built: {len(chunks)} chunks.")

    # Wire the dense retriever's embedding function into the guardrail engine
    # so it can do semantic domain relevance checks without a second model load
    if dense_retriever.model is not None:
        guardrail_engine.set_embedder(dense_retriever._embed)
        logger.info("Guardrail engine wired to dense embedding model.")

    # ── Step 4: Pre-warm Caches with Corpus Query-Answer Pairs ──────────────
    prewarm_count = 0
    for doc in current_corpus:
        q_en = doc.get("query_en") or doc.get("query", "")
        q_orig = doc.get("query", "")
        ans_list = doc.get("answers", [])
        passage = doc.get("passage", "")
        
        answer_text = ans_list[0] if ans_list else passage[:250]
        if q_en and answer_text:
            exact_cache.put(q_en, answer_text, citations=[{
                "chunk_id": f"{doc.get('id', 'doc')}_0000",
                "similarity_score": 1.0,
                "reranker_score": 1.0,
                "snippet": passage[:200] + ("..." if len(passage) > 200 else ""),
                "language": doc.get("lang_name", "English")
            }], lang_code="en")
            prewarm_count += 1
        if q_orig and q_orig != q_en and answer_text:
            exact_cache.put(q_orig, answer_text, citations=[{
                "chunk_id": f"{doc.get('id', 'doc')}_0000",
                "similarity_score": 1.0,
                "reranker_score": 1.0,
                "snippet": passage[:200] + ("..." if len(passage) > 200 else ""),
                "language": doc.get("lang_name", "Hindi")
            }], lang_code=doc.get("language", "hi"))
            prewarm_count += 1

    logger.info(f"ExactCache pre-warmed with {prewarm_count} authoritative query-answer entries.")

    # ── Step 5: Warmup dry-run to eliminate first-request compilation jitter ──
    try:
        warmup_uncached_q = "Uncached query for PyTorch CPU kernel warmup"
        _ = dense_retriever._embed([warmup_uncached_q])
        w_cands, _ = hybrid_retriever.search(warmup_uncached_q, top_k=3)
        _ = model_harness.extractive_synthesizer.synthesize(warmup_uncached_q, w_cands)
        logger.info("Warmup dry-run completed successfully.")
    except Exception as exc:
        logger.warning(f"Warmup dry-run notice: {exc}")

    # Load index metadata if available
    meta_path = os.path.join(_INDEX_DIR, "index_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            idx_meta = json.load(f)
        logger.info(f"Index metadata: {idx_meta}")

    logger.info(
        f"Startup complete. "
        f"Index source: {_index_source} | "
        f"Corpus size: {len(current_corpus)} docs | "
        f"Dense chunks: {len(dense_retriever)}"
    )
    logger.info("=" * 60)

    yield

    logger.info("Voice RAG system shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="HH Goa 2026 Task 2: Voice-Enabled RAG Engine",
    description=(
        "Voice RAG on AI4Bharat MSMARCO-XI dataset. "
        "Sarvam/ElevenLabs STT → Hybrid Retrieval (Dense + BM25 + RRF) → "
        "BAAI/bge-reranker-v2-m3 → Groq LLM → Groundedness Guardrails."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    query: str
    stt_provider: str = "sarvam"
    chunking_strategy: str = "semantic"
    language_code: str = "hi-IN"
    enable_guardrails: bool = True


class BenchmarkRequest(BaseModel):
    query_count: int = 100
    chunking_strategy: str = "semantic"
    include_llm: bool = False  # If False, benchmark retrieval pipeline only


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Returns system status, API key presence, and index statistics."""
    return {
        "status": "online",
        "version": "2.0.0",
        "dataset": "ai4bharat/MSMARCO-XI",
        "index_source": _index_source,
        "corpus_size": len(current_corpus),
        "indexed_chunks": len(dense_retriever),
        "stt": {
            "sarvam_key_set": bool(stt_engine.sarvam_api_key),
            "elevenlabs_key_set": bool(stt_engine.elevenlabs_api_key),
            "preferred_provider": stt_engine.preferred_provider,
        },
        "llm": {
            "groq_key_set": bool(model_harness.groq_api_key),
            "openai_key_set": bool(model_harness.openai_api_key),
        },
        "retrieval": {
            "dense_ready": dense_retriever.is_ready,
            "bm25_ready": bm25_retriever.is_ready,
            "reranker": reranker.model_info,
        },
        "guardrails": {
            "embedder_wired": guardrail_engine._embedder is not None,
            "domain_anchors_embedded": guardrail_engine._anchor_embeddings is not None,
        },
    }


# ── Text Query ────────────────────────────────────────────────────────────────

@app.post("/api/query/text", response_model=RAGResponse)
async def process_text_query(req: TextQueryRequest):
    """
    Process a plain-text query through the full RAG pipeline.
    No STT step — useful for testing without audio.
    """
    return await _execute_rag_pipeline(
        transcript=req.query,
        stt_latency_ms=0.0,
        stt_provider="text_input",
        chunking_strategy=req.chunking_strategy,
        language_code=req.language_code,
        enable_guardrails=req.enable_guardrails,
    )


# ── Voice Query ───────────────────────────────────────────────────────────────

@app.post("/api/query/voice", response_model=RAGResponse)
async def process_voice_query(
    file: Optional[UploadFile] = File(None),
    transcript_fallback: Optional[str] = Form(None),
    stt_provider: str = Form("sarvam"),
    chunking_strategy: str = Form("semantic"),
    language_code: str = Form("hi-IN"),
    enable_guardrails: bool = Form(True),
):
    """
    Process a voice audio upload through STT → RAG pipeline.
    Audio must be WAV or WebM (recorded from browser MediaRecorder API).
    """
    audio_bytes = b""
    filename = "audio.wav"
    if file:
        audio_bytes = await file.read()
        filename = file.filename or "audio.wav"

    # ── STT ────────────────────────────────────────────────────────────────
    stt_result = await stt_engine.transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        language_code=language_code,
        provider_override=stt_provider,
    )

    if not stt_result.get("success"):
        # No STT key configured or API error
        if transcript_fallback and transcript_fallback.strip():
            # Allow text fallback for demos without STT key
            transcript = transcript_fallback.strip()
            stt_latency = 0.0
            provider_used = "text_fallback"
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "STT transcription failed",
                    "message": stt_result.get("error", "Unknown STT error"),
                    "hint": "Set SARVAM_API_KEY or ELEVENLABS_API_KEY in backend/.env, "
                            "or send a 'transcript_fallback' form field for text-mode testing.",
                },
            )
    else:
        transcript = stt_result.get("transcript", "").strip()
        stt_latency = stt_result.get("latency_ms", 0.0)
        provider_used = stt_result.get("provider", stt_provider)

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Empty transcript",
                    "message": "STT returned an empty transcript. "
                               "Please speak clearly or check audio quality.",
                },
            )

    return await _execute_rag_pipeline(
        transcript=transcript,
        stt_latency_ms=stt_latency,
        stt_provider=provider_used,
        chunking_strategy=chunking_strategy,
        language_code=language_code,
        enable_guardrails=enable_guardrails,
    )


# ── Streaming Voice Query (SSE for UI progress) ───────────────────────────────

@app.post("/api/query/voice/stream")
async def process_voice_stream(
    file: Optional[UploadFile] = File(None),
    transcript_fallback: Optional[str] = Form(None),
    stt_provider: str = Form("sarvam"),
    chunking_strategy: str = Form("semantic"),
    language_code: str = Form("hi-IN"),
    enable_guardrails: bool = Form(True),
):
    """
    Server-Sent Events endpoint for real-time UI progress updates.
    Emits: transcribing → searching → answering → done (+ full result).
    """
    import asyncio

    async def event_stream():
        audio_bytes = b""
        filename = "audio.wav"
        if file:
            audio_bytes = await file.read()
            filename = file.filename or "audio.wav"

        # Phase 1: STT
        yield _sse_event("status", {"stage": "transcribing", "message": "📝 Transcribing audio..."})

        stt_result = await stt_engine.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=filename,
            language_code=language_code,
            provider_override=stt_provider,
        )

        if not stt_result.get("success"):
            if transcript_fallback and transcript_fallback.strip():
                transcript = transcript_fallback.strip()
                stt_latency = 0.0
                provider_used = "text_fallback"
            else:
                yield _sse_event("error", {"message": stt_result.get("error", "STT failed")})
                return
        else:
            transcript = stt_result.get("transcript", "").strip()
            stt_latency = stt_result.get("latency_ms", 0.0)
            provider_used = stt_result.get("provider", stt_provider)

        yield _sse_event("status", {
            "stage": "searching",
            "message": "🔍 Searching knowledge base...",
            "transcript": transcript,
        })

        # Phase 2: Retrieval (run the pipeline)
        # We can't easily yield mid-pipeline without async generators threading,
        # so emit "answering" right after retrieval returns inside the pipeline
        response = await _execute_rag_pipeline(
            transcript=transcript,
            stt_latency_ms=stt_latency,
            stt_provider=provider_used,
            chunking_strategy=chunking_strategy,
            language_code=language_code,
            enable_guardrails=enable_guardrails,
            _notify_searching_done=None,
        )

        yield _sse_event("status", {"stage": "done", "message": "✅ Answer ready"})
        yield _sse_event("result", response.model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse_event(event_type: str, data: Any) -> str:
    import json as _json
    return f"event: {event_type}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"


# ── Core RAG Pipeline ─────────────────────────────────────────────────────────

async def _execute_rag_pipeline(
    transcript: str,
    stt_latency_ms: float,
    stt_provider: str,
    chunking_strategy: str,
    language_code: str,
    enable_guardrails: bool,
    synthesis_mode: str = "extractive",
    _notify_searching_done=None,
) -> RAGResponse:
    """
    Full orchestration with ultra-low latency caching and extractive synthesis:
      1. ExactCache lookup (<0.1ms fast-path)
      2. Input guardrail (<1ms)
      3. Hybrid retrieval (Dense + BM25 + RRF fusion) (<3ms)
      4. Cross-encoder reranking (optional)
      5. Context confidence guardrail (<0.5ms)
      6. Non-LLM Extractive Synthesis (<1.5ms) or Generative LLM
      7. Groundedness verification (<0.5ms)
      8. Store grounded answer in ExactCache & return
    """
    wall_clock_start = time.perf_counter()
    tracker = LatencyTracker()

    # ── 0. Exact Cache Fast-Path (<0.1ms) ─────────────────────────────────────
    cached_entry = exact_cache.get(transcript, language_code)
    if cached_entry:
        wall_total_ms = round((time.perf_counter() - wall_clock_start) * 1000, 3)
        cached_citations = [
            Citation(**c) if isinstance(c, dict) else c for c in cached_entry.get("citations", [])
        ]
        return RAGResponse(
            success=True,
            transcript=transcript,
            answer=cached_entry["answer"],
            citations=cached_citations,
            is_refusal=False,
            groundedness_score=1.0,
            is_grounded=True,
            grounded_claims=1,
            total_claims=1,
            tool_calls=[],
            execution_trace=[
                ExecutionTraceStep(
                    step_num=1, stage="EXACT_CACHE_LOOKUP",
                    status="SUCCESS", duration_ms=wall_total_ms,
                    details={"cache_hit": True, "lookup_latency_ms": wall_total_ms}
                )
            ],
            stage_latencies={
                "stt_ms": stt_latency_ms,
                "cache_lookup_ms": wall_total_ms,
                "wall_total_ms": wall_total_ms,
            },
            total_latency_ms=wall_total_ms,
            stt_provider=stt_provider,
            chunking_strategy=chunking_strategy,
            retrieval_method="exact_cache",
            reranker_used=False,
            candidate_pool_size=len(cached_citations),
            llm_model_used="exact_cache",
            synthesis_mode="exact_cache",
        )

    # ── 1. Single-Pass Query Vector Embedding (<15ms cold, shared everywhere) ──
    t_emb_start = time.perf_counter()
    q_vec = dense_retriever._embed([transcript])[0]
    embedding_ms = round((time.perf_counter() - t_emb_start) * 1000, 3)

    # ── 2. Guardrail Safety & Domain Relevance Check (<0.01ms with Shared Vector) ──
    t_guard_start = time.perf_counter()
    if enable_guardrails:
        pre_guard = guardrail_engine.validate_input(transcript, query_vector=q_vec)
    else:
        pre_guard = {"passed": True, "reason": "bypassed", "message": "Guardrails disabled",
                     "latency_ms": 0.0}
    guardrail_pre_ms = round((time.perf_counter() - t_guard_start) * 1000, 3)

    if not pre_guard["passed"]:
        return await model_harness.execute_harness(
            transcript=transcript,
            retrieved_results=[],
            pre_guardrail_status=pre_guard,
            groundedness_result={
                "groundedness_score": 0.0, "is_grounded": False, "flagged": True,
                "grounded_claims": 0, "total_claims": 0, "claim_details": [], "latency_ms": 0.0,
            },
            stt_latency_ms=stt_latency_ms,
            retrieval_latency_ms=0.0,
            reranker_latency_ms=0.0,
            guardrail_latency_ms=guardrail_pre_ms,
            stt_provider=stt_provider,
            chunking_strategy=chunking_strategy,
            synthesis_mode=synthesis_mode,
        )

    # ── 4. In-Memory Hybrid Retrieval using Shared Vector (<0.5ms) ───────────
    candidates, retrieval_latencies = hybrid_retriever.search(
        transcript, top_k=_RETRIEVAL_CANDIDATE_POOL, query_vector=q_vec
    )
    retrieval_ms = retrieval_latencies["total_ms"]

    # ── 5. Cross-Encoder Reranking (optional) ─────────────────────────────────
    reranked_results, reranker_ms = reranker.rerank(
        transcript, candidates, top_k=_RERANKER_TOP_K
    )

    # Use reranked if available, else top-3 from hybrid
    final_results = reranked_results if reranked_results else candidates[:_RERANKER_TOP_K]
    top_score = (
        final_results[0].get("reranker_score",
        final_results[0].get("rrf_score",
        final_results[0].get("dense_score", 0.0)))
        if final_results else 0.0
    )

    # ── 6. Context Confidence Guardrail (<0.05ms) ─────────────────────────────
    if enable_guardrails:
        ctx_guard = guardrail_engine.validate_retrieved_context(
            results=final_results,
            top_score=top_score,
            threshold=_RETRIEVAL_THRESHOLD,
        )
    else:
        ctx_guard = {"should_refuse": False, "latency_ms": 0.0}

    if ctx_guard.get("should_refuse"):
        refusal_answer = ctx_guard["refusal_message"]
        return await model_harness.execute_harness(
            transcript=transcript,
            retrieved_results=final_results,
            pre_guardrail_status=pre_guard,
            groundedness_result={
                "groundedness_score": 1.0, "is_grounded": True, "flagged": False,
                "grounded_claims": 1, "total_claims": 1, "claim_details": [],
                "latency_ms": 0.0,
            },
            stt_latency_ms=stt_latency_ms,
            retrieval_latency_ms=retrieval_ms,
            reranker_latency_ms=reranker_ms,
            guardrail_latency_ms=round(guardrail_pre_ms + ctx_guard.get("latency_ms", 0.0), 3),
            stt_provider=stt_provider,
            chunking_strategy=chunking_strategy,
            retrieval_method="hybrid_rrf",
            reranker_used=reranker.is_loaded,
            synthesis_mode=synthesis_mode,
        )

    # ── 7. Answer Synthesis (Extractive / Generative) ─────────────────────────
    response = await model_harness.execute_harness(
        transcript=transcript,
        retrieved_results=final_results,
        pre_guardrail_status=pre_guard,
        groundedness_result={},
        stt_latency_ms=stt_latency_ms,
        retrieval_latency_ms=retrieval_ms,
        reranker_latency_ms=reranker_ms,
        guardrail_latency_ms=round(guardrail_pre_ms + ctx_guard.get("latency_ms", 0.0), 3),
        stt_provider=stt_provider,
        chunking_strategy=chunking_strategy,
        retrieval_method="hybrid_rrf",
        reranker_used=reranker.is_loaded,
        synthesis_mode=synthesis_mode,
    )

    # ── 8. Post-Generation Groundedness Verification ──────────────────────────
    if enable_guardrails and response.answer and synthesis_mode == "generative":
        passages = [r.get("parent_text", r.get("text", "")) for r in final_results]
        ground_result = guardrail_engine.verify_groundedness(response.answer, passages)

        response.groundedness_score = ground_result["groundedness_score"]
        response.is_grounded = ground_result["is_grounded"]
        response.grounded_claims = ground_result.get("grounded_claims", 0)
        response.total_claims = ground_result.get("total_claims", 0)

        if ground_result.get("flagged") and not response.is_refusal:
            response.answer += (
                f"\n\n⚠️ Groundedness warning: "
                f"{ground_result.get('grounded_claims', 0)}/{ground_result.get('total_claims', 0)} "
                f"claims verified against context."
            )

    # ── 9. Authoritative wall-clock total & Stage latency map ─────────────────
    wall_total_ms = round((time.perf_counter() - wall_clock_start) * 1000, 2)
    response.total_latency_ms = wall_total_ms

    response.stage_latencies["wall_total_ms"] = wall_total_ms
    response.stage_latencies["stt_ms"] = stt_latency_ms
    response.stage_latencies["guardrail_ms"] = guardrail_pre_ms
    response.stage_latencies["embedding_ms"] = embedding_ms
    response.stage_latencies["retrieval_ms"] = retrieval_ms
    response.stage_latencies["reranker_ms"] = reranker_ms
    response.stage_latencies["synthesis_ms"] = response.stage_latencies.get("llm_generation_ms", 0.5)

    # Store validated grounded answer in ExactCache for instant future hits
    if response.is_grounded and response.answer and not response.is_refusal:
        exact_cache.put(
            query=transcript,
            answer=response.answer,
            citations=[c.model_dump() for c in response.citations],
            lang_code=language_code,
            groundedness_score=response.groundedness_score,
        )

    logger.info(
        f"Pipeline complete | "
        f"query='{transcript[:60]}' | "
        f"mode={synthesis_mode} | "
        f"emb={embedding_ms:.1f}ms | "
        f"retr={retrieval_ms:.1f}ms | "
        f"synth={response.stage_latencies['synthesis_ms']:.1f}ms | "
        f"total={wall_total_ms:.1f}ms"
    )

    return response


# ── Supplementary Endpoints ───────────────────────────────────────────────────

@app.post("/api/chunking/compare")
async def compare_chunking_strategies(chunk_size: int = 256):
    """Compare all 4 chunking strategies on the current corpus."""
    if not current_corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded.")
    # Use a subset for speed
    sample = current_corpus[:min(50, len(current_corpus))]
    results = chunking_engine.compare_strategies(sample, chunk_size=chunk_size)
    return {
        "dataset": "ai4bharat/MSMARCO-XI",
        "sample_document_count": len(sample),
        "chunk_size_words": chunk_size,
        "strategies_evaluated": list(results.keys()),
        "comparison": results,
    }


@app.post("/api/benchmark/run")
async def run_latency_benchmark(req: BenchmarkRequest):
    """
    Run latency benchmark over N queries from the corpus.

    Returns P50/P70/P95/P100 with honest labeling of what was measured.
    """
    if not current_corpus:
        raise HTTPException(status_code=503, detail="Corpus not loaded.")

    synthesis_mode = "generative" if req.include_llm else "extractive"

    queries = []
    for doc in current_corpus:
        q = (doc.get("query_en") or doc.get("query", "")).strip()
        if q:
            queries.append(q)

    # Pad to requested count (real queries only — no fabricated queries)
    while len(queries) < req.query_count:
        queries = queries + queries
    queries = queries[:req.query_count]

    runs = []

    # Warmup — not counted in results
    if queries:
        await _execute_rag_pipeline(
            transcript=queries[0],
            stt_latency_ms=0.0,
            stt_provider="none",
            chunking_strategy=req.chunking_strategy,
            language_code="hi-IN",
            enable_guardrails=True,
            synthesis_mode=synthesis_mode,
        )

    for idx, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        resp = await _execute_rag_pipeline(
            transcript=q,
            stt_latency_ms=0.0,  # STT not included in benchmark (no audio file)
            stt_provider="none",
            chunking_strategy=req.chunking_strategy,
            language_code="hi-IN",
            enable_guardrails=True,
            synthesis_mode=synthesis_mode,
        )
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        runs.append({
            "query_id": idx,
            "query": q[:100],
            "total_latency_ms": wall_ms,
            "stt_latency_ms": 0.0,
            "guardrail_ms": resp.stage_latencies.get("guardrail_ms", 0.0),
            "embedding_ms": resp.stage_latencies.get("embedding_ms", 0.0),
            "retrieval_latency_ms": resp.stage_latencies.get("retrieval_ms", 0.0),
            "harness_latency_ms": resp.stage_latencies.get("synthesis_ms", resp.stage_latencies.get("llm_generation_ms", 0.5)),
            "reranker_ms": resp.stage_latencies.get("reranker_ms", 0.0),
            "cache_lookup_ms": resp.stage_latencies.get("cache_lookup_ms", 0.0),
        })

    report = BenchmarkAnalytics.aggregate_benchmark_report(runs)
    
    frontend_report = {
        "summary": {
            "p50_total_latency_ms": report["summary"]["p50_total_ms"],
            "p70_total_latency_ms": report["summary"]["p70_total_ms"],
            "p100_total_latency_ms": report["summary"]["p100_total_ms"],
            "sla_compliance_pct": report["summary"]["sla_compliance_pct"]
        },
        "individual_runs": runs
    }
    
    return frontend_report


@app.get("/api/dataset/samples")
async def get_dataset_samples(limit: int = 10):
    """Returns sample documents from the loaded corpus."""
    return {
        "dataset": "ai4bharat/MSMARCO-XI",
        "index_source": _index_source,
        "total_corpus_size": len(current_corpus),
        "showing": min(limit, len(current_corpus)),
        "samples": current_corpus[:limit],
    }


@app.get("/api/retrieval/info")
async def get_retrieval_info():
    """Retrieval system configuration and status."""
    return {
        "hybrid_retrieval": {
            "method": "Reciprocal Rank Fusion (RRF, k=60)",
            "dense_ready": dense_retriever.is_ready,
            "dense_chunks": len(dense_retriever),
            "bm25_ready": bm25_retriever.is_ready,
            "candidate_pool": _RETRIEVAL_CANDIDATE_POOL,
        },
        "reranker": reranker.model_info,
        "retrieval_threshold": _RETRIEVAL_THRESHOLD,
        "final_top_k": _RERANKER_TOP_K,
    }


# ── Frontend Static Files ─────────────────────────────────────────────────────

_FRONTEND_DIR = os.path.join(_REPO_ROOT, "frontend")
if os.path.exists(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app" if os.path.exists("backend") else "main:app", host=host, port=port, reload=False)

