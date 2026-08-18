"""
Model Harness — Structured Orchestration for Voice RAG
======================================================
Provides:
  • Pydantic-validated input/output schemas
  • Async LLM generation with exponential backoff retries
  • Per-stage execution tracing
  • Tool call recording
  • Groq (primary) → OpenAI (fallback) → deterministic refusal (last resort)

Critically: NO hardcoded evaluation scores. All groundedness values are
computed by the GuardrailEngine and passed in at call time.
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional

import httpx
from pydantic import BaseModel, Field

from generation.prompts import build_groq_payload, build_openai_payload, REFUSAL_PHRASES

logger = logging.getLogger("model_harness")

# Groq models in priority order (fastest first)
_GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # Best quality, still very fast on Groq
    "llama3-8b-8192",            # Fallback: smaller but <100ms
]


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: str
    similarity_score: float
    reranker_score: float = 0.0
    snippet: str
    language: str


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    output: Any
    duration_ms: float


class ExecutionTraceStep(BaseModel):
    step_num: int
    stage: str
    status: str  # "SUCCESS" | "REJECTED" | "RETRY" | "FAILED" | "SKIPPED"
    duration_ms: float
    details: Dict[str, Any]


class RAGResponse(BaseModel):
    success: bool
    transcript: str
    answer: str
    citations: List[Citation]
    is_refusal: bool
    refusal_reason: Optional[str] = None

    # Groundedness — set by GuardrailEngine, never hardcoded
    groundedness_score: float
    is_grounded: bool
    grounded_claims: int = 0
    total_claims: int = 0

    tool_calls: List[ToolCallResult]
    execution_trace: List[ExecutionTraceStep]

    stage_latencies: Dict[str, float]
    total_latency_ms: float

    stt_provider: str
    chunking_strategy: str

    # Retrieval metadata
    retrieval_method: str = "hybrid_rrf"
    reranker_used: bool = False
    candidate_pool_size: int = 0
    llm_model_used: str = ""


# ── Model Harness ─────────────────────────────────────────────────────────────

class ModelHarness:
    """
    Orchestrates the full RAG response cycle:

    pre-guardrail → retrieval (logged externally) → tool calls →
    LLM generation (with retry) → groundedness check → RAGResponse
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._groq_timeout = float(os.getenv("GROQ_TIMEOUT_S", "6.0"))
        self._openai_timeout = float(os.getenv("OPENAI_TIMEOUT_S", "8.0"))

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def execute_harness(
        self,
        transcript: str,
        retrieved_results: List[Dict[str, Any]],
        pre_guardrail_status: Dict[str, Any],
        groundedness_result: Dict[str, Any],
        stt_latency_ms: float,
        retrieval_latency_ms: float,
        reranker_latency_ms: float,
        guardrail_latency_ms: float,
        stt_provider: str,
        chunking_strategy: str,
        retrieval_method: str = "hybrid_rrf",
        reranker_used: bool = False,
        max_retries: int = 2,
    ) -> RAGResponse:
        """
        Execute the full model harness.

        Args:
            transcript:            User's question (from STT or text input).
            retrieved_results:     Top-k chunks from hybrid retrieval + reranker.
            pre_guardrail_status:  Result from GuardrailEngine.validate_input().
            groundedness_result:   Result from GuardrailEngine.verify_groundedness().
                                   (Computed externally after generation, passed back in.)
            stt_latency_ms:        Real measured STT latency.
            retrieval_latency_ms:  Real measured hybrid retrieval latency.
            reranker_latency_ms:   Real measured reranker latency.
            guardrail_latency_ms:  Combined guardrail latency (pre + post).
            stt_provider:          "Sarvam AI" | "ElevenLabs Scribe" | "none"
            chunking_strategy:     Active chunking strategy name.
            retrieval_method:      "hybrid_rrf" | "dense_only"
            reranker_used:         Whether the cross-encoder reranker ran.
            max_retries:           LLM retry attempts on transient failures.

        Returns:
            A fully populated RAGResponse.
        """
        total_start = time.perf_counter()
        trace: List[ExecutionTraceStep] = []
        tool_calls: List[ToolCallResult] = []
        step = 1

        # ── Step 1: STT trace ─────────────────────────────────────────────────
        trace.append(ExecutionTraceStep(
            step_num=step, stage="STT_TRANSCRIPTION",
            status="SUCCESS" if stt_latency_ms > 0 else "SKIPPED",
            duration_ms=stt_latency_ms,
            details={"provider": stt_provider, "transcript_length": len(transcript)},
        ))
        step += 1

        # ── Step 2: Pre-guardrail trace ───────────────────────────────────────
        trace.append(ExecutionTraceStep(
            step_num=step, stage="PRE_GUARDRAIL",
            status="SUCCESS" if pre_guardrail_status["passed"] else "REJECTED",
            duration_ms=pre_guardrail_status.get("latency_ms", 0.0),
            details={k: v for k, v in pre_guardrail_status.items() if k != "latency_ms"},
        ))
        step += 1

        # Early return on pre-guardrail rejection
        if not pre_guardrail_status["passed"]:
            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            return RAGResponse(
                success=False,
                transcript=transcript,
                answer=pre_guardrail_status["message"],
                citations=[],
                is_refusal=True,
                refusal_reason=pre_guardrail_status["reason"],
                groundedness_score=0.0,
                is_grounded=False,
                tool_calls=[],
                execution_trace=trace,
                stage_latencies={
                    "stt_ms": stt_latency_ms,
                    "pre_guardrail_ms": pre_guardrail_status.get("latency_ms", 0.0),
                    "total_ms": total_ms,
                },
                total_latency_ms=total_ms,
                stt_provider=stt_provider,
                chunking_strategy=chunking_strategy,
            )

        # ── Step 3: Retrieval trace ───────────────────────────────────────────
        trace.append(ExecutionTraceStep(
            step_num=step, stage="HYBRID_RETRIEVAL",
            status="SUCCESS",
            duration_ms=retrieval_latency_ms,
            details={
                "retrieved_count": len(retrieved_results),
                "method": retrieval_method,
                "reranker_used": reranker_used,
                "reranker_latency_ms": reranker_latency_ms,
                "top_score": retrieved_results[0].get("reranker_score",
                             retrieved_results[0].get("rrf_score", 0.0))
                             if retrieved_results else 0.0,
            },
        ))
        step += 1

        # ── Step 4: Context assembly tool call ────────────────────────────────
        tool_t = time.perf_counter()
        passages = self._extract_passages(retrieved_results)

        context_summary = (
            f"Assembled {len(passages)} passages | "
            f"Total words: {sum(len(p.split()) for p in passages)} | "
            f"Languages: {list({r.get('metadata', {}).get('language', 'en') for r in retrieved_results})}"
        )
        tool_calls.append(ToolCallResult(
            tool_name="context_assembler",
            arguments={"top_k": len(retrieved_results), "use_parent_text": True},
            output=context_summary,
            duration_ms=round((time.perf_counter() - tool_t) * 1000, 3),
        ))

        trace.append(ExecutionTraceStep(
            step_num=step, stage="CONTEXT_ASSEMBLY",
            status="SUCCESS",
            duration_ms=round((time.perf_counter() - tool_t) * 1000, 3),
            details={"passage_count": len(passages), "context_summary": context_summary},
        ))
        step += 1

        # ── Step 5: LLM generation with retry ────────────────────────────────
        llm_start = time.perf_counter()
        answer, model_used, attempt_count = await self._generate_with_retry(
            transcript, passages, max_retries
        )
        llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

        gen_status = "SUCCESS" if answer and not self._is_empty_answer(answer) else "FALLBACK"
        trace.append(ExecutionTraceStep(
            step_num=step, stage="LLM_GENERATION",
            status=gen_status,
            duration_ms=llm_ms,
            details={
                "model_used": model_used,
                "attempts": attempt_count,
                "answer_length": len(answer),
                "is_refusal": any(p in answer.lower() for p in REFUSAL_PHRASES),
            },
        ))
        step += 1

        # ── Step 6: Groundedness trace ────────────────────────────────────────
        trace.append(ExecutionTraceStep(
            step_num=step, stage="GROUNDEDNESS_VERIFICATION",
            status="SUCCESS" if groundedness_result.get("is_grounded") else "FLAGGED",
            duration_ms=groundedness_result.get("latency_ms", 0.0),
            details={
                "groundedness_score": groundedness_result.get("groundedness_score", 0.0),
                "grounded_claims": groundedness_result.get("grounded_claims", 0),
                "total_claims": groundedness_result.get("total_claims", 0),
                "flagged": groundedness_result.get("flagged", False),
            },
        ))
        step += 1

        # ── Build citations ───────────────────────────────────────────────────
        citations = []
        for r in retrieved_results[:3]:
            citations.append(Citation(
                chunk_id=r["chunk_id"],
                similarity_score=round(r.get("dense_score", r.get("rrf_score", 0.0)), 4),
                reranker_score=round(r.get("reranker_score", 0.0), 4),
                snippet=r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                language=r.get("metadata", {}).get("lang_name", "English"),
            ))

        # ── Compose final stage latencies ─────────────────────────────────────
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        stage_latencies = {
            "stt_ms": round(stt_latency_ms, 3),
            "pre_guardrail_ms": round(pre_guardrail_status.get("latency_ms", 0.0), 3),
            "retrieval_ms": round(retrieval_latency_ms, 3),
            "reranker_ms": round(reranker_latency_ms, 3),
            "context_assembly_ms": round(tool_calls[0].duration_ms, 3),
            "llm_generation_ms": round(llm_ms, 3),
            "groundedness_ms": round(groundedness_result.get("latency_ms", 0.0), 3),
            "guardrail_total_ms": round(guardrail_latency_ms, 3),
            "harness_total_ms": total_ms,
        }

        return RAGResponse(
            success=True,
            transcript=transcript,
            answer=answer,
            citations=citations,
            is_refusal=self._is_refusal_answer(answer),
            refusal_reason="llm_refused" if self._is_refusal_answer(answer) else None,
            groundedness_score=groundedness_result.get("groundedness_score", 0.0),
            is_grounded=groundedness_result.get("is_grounded", False),
            grounded_claims=groundedness_result.get("grounded_claims", 0),
            total_claims=groundedness_result.get("total_claims", 0),
            tool_calls=tool_calls,
            execution_trace=trace,
            stage_latencies=stage_latencies,
            total_latency_ms=total_ms,
            stt_provider=stt_provider,
            chunking_strategy=chunking_strategy,
            retrieval_method=retrieval_method,
            reranker_used=reranker_used,
            candidate_pool_size=len(retrieved_results),
            llm_model_used=model_used,
        )

    # ── LLM Generation ────────────────────────────────────────────────────────

    async def _generate_with_retry(
        self,
        query: str,
        passages: List[str],
        max_retries: int,
    ) -> tuple[str, str, int]:
        """
        Try Groq, then OpenAI, with exponential backoff on transient failures.

        Returns: (answer_text, model_name_used, total_attempts)
        """
        attempt = 0
        last_error = None

        # Try each Groq model
        if self.groq_api_key:
            for model in _GROQ_MODELS:
                for retry in range(max_retries + 1):
                    attempt += 1
                    try:
                        answer = await self._call_groq(query, passages, model)
                        return answer, model, attempt
                    except Exception as exc:
                        last_error = exc
                        logger.warning(f"Groq [{model}] attempt {retry + 1} failed: {exc}")
                        if retry < max_retries:
                            await asyncio.sleep(0.1 * (2 ** retry))

        # Try OpenAI
        if self.openai_api_key:
            for retry in range(max_retries + 1):
                attempt += 1
                try:
                    answer = await self._call_openai(query, passages)
                    return answer, "gpt-4o-mini", attempt
                except Exception as exc:
                    last_error = exc
                    logger.warning(f"OpenAI attempt {retry + 1} failed: {exc}")
                    if retry < max_retries:
                        await asyncio.sleep(0.1 * (2 ** retry))

        # All providers failed — return explicit no-answer signal
        logger.error(f"All LLM providers failed. Last error: {last_error}")
        return "", "none", attempt

    async def _call_groq(self, query: str, passages: List[str], model: str) -> str:
        """Call Groq chat completions API."""
        payload = build_groq_payload(query, passages, model=model)
        async with httpx.AsyncClient(timeout=self._groq_timeout) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}",
                         "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _call_openai(self, query: str, passages: List[str]) -> str:
        """Call OpenAI chat completions API."""
        payload = build_openai_payload(query, passages)
        async with httpx.AsyncClient(timeout=self._openai_timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}",
                         "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_passages(results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract passage text for LLM context.
        Uses parent_text (wider context) when available — this is the
        parent-child retrieval pattern in action.
        """
        passages = []
        for r in results:
            text = r.get("parent_text") or r.get("text", "")
            if text:
                passages.append(text.strip())
        return passages

    @staticmethod
    def _is_empty_answer(answer: str) -> bool:
        return not answer or len(answer.strip()) < 5

    @staticmethod
    def _is_refusal_answer(answer: str) -> bool:
        if not answer:
            return True
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
