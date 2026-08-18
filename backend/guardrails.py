"""
Multi-Layer Guardrail Engine for RAG Model Protection
1. Pre-execution Safety & Prompt-Injection Guard
2. Domain Relevance & Off-Topic Query Filter
3. Context Retrieval Confidence & Refusal Trigger
4. Post-execution Hallucination & Groundedness Verifier
"""

import re
import time
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("guardrails")

class GuardrailEngine:
    def __init__(self):
        # Prompt Injection & Toxic Patterns
        self.unsafe_patterns = [
            re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+(a|an)\s+unrestricted", re.IGNORECASE),
            re.compile(r"system\s+prompt|admin\s+mode|root\s+access", re.IGNORECASE),
            re.compile(r"bypass\s+safety|hack\s+|exploit\s+", re.IGNORECASE)
        ]

        # Off-topic triggers out of scope for MSMARCO-XI corpus
        self.off_topic_patterns = [
            re.compile(r"buy\s+(crypto|bitcoin|ethereum|stocks)", re.IGNORECASE),
            re.compile(r"tell\s+me\s+a\s+dirty\s+joke", re.IGNORECASE),
            re.compile(r"write\s+a\s+virus|malware|keylogger", re.IGNORECASE)
        ]

    def validate_input(self, query: str) -> Dict[str, Any]:
        """
        Pre-execution guardrail check.
        Inspects query for prompt injection, unsafe input, and domain relevance.
        """
        start_time = time.perf_counter()
        query_clean = query.strip()

        if not query_clean:
            return {
                "passed": False,
                "reason": "empty_query",
                "message": "Query text cannot be empty.",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        # Check Unsafe / Prompt Injection
        for pattern in self.unsafe_patterns:
            if pattern.search(query_clean):
                return {
                    "passed": False,
                    "reason": "unsafe_prompt_injection",
                    "message": "Security Alert: Query contains prompt injection or unsafe instructions.",
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }

        # Check Off-Topic / Out of Domain
        for pattern in self.off_topic_patterns:
            if pattern.search(query_clean):
                return {
                    "passed": False,
                    "reason": "off_topic_query",
                    "message": "Off-Topic Refusal: Query is out of domain for the MSMARCO-XI dataset.",
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
                }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "passed": True,
            "reason": "clean_input",
            "message": "Input passed all pre-execution safety and domain guardrails.",
            "latency_ms": elapsed_ms
        }

    def validate_retrieved_context(
        self,
        retrieval_results: List[Dict[str, Any]],
        top_score: float,
        threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Retrieval confidence guardrail.
        Triggers explicit refusal when top retrieved document score is below similarity threshold.
        """
        start_time = time.perf_counter()

        if not retrieval_results or top_score < threshold:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "should_refuse": True,
                "refusal_reason": "low_context_confidence",
                "refusal_message": (
                    "I apologize, but I cannot find relevant grounded context in the MSMARCO-XI dataset "
                    f"to answer your question reliably (Top relevance score {top_score:.2f} < threshold {threshold:.2f})."
                ),
                "latency_ms": elapsed_ms
            }

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "should_refuse": False,
            "refusal_reason": None,
            "refusal_message": None,
            "latency_ms": elapsed_ms
        }

    def verify_groundedness(
        self,
        generated_answer: str,
        retrieved_passages: List[str]
    ) -> Dict[str, Any]:
        """
        Post-execution hallucination & groundedness verifier.
        Calculates word overlap & key entity inclusion between LLM answer and retrieved context.
        """
        start_time = time.perf_counter()

        if not generated_answer or not retrieved_passages:
            return {
                "groundedness_score": 0.0,
                "is_grounded": False,
                "flagged": True,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        combined_context = " ".join(retrieved_passages).lower()
        context_words = set(re.findall(r"\w+", combined_context))

        answer_words = re.findall(r"\w+", generated_answer.lower())
        # Filter out short stopwords
        filtered_answer_words = [w for w in answer_words if len(w) > 3]

        if not filtered_answer_words:
            score = 1.0
        else:
            grounded_count = sum(1 for w in filtered_answer_words if w in context_words)
            score = round(grounded_count / len(filtered_answer_words), 2)

        is_grounded = score >= 0.50
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "groundedness_score": score,
            "is_grounded": is_grounded,
            "flagged": not is_grounded,
            "warning": None if is_grounded else "Warning: Potential hallucination detected. Answer contains words not found in retrieved passages.",
            "latency_ms": elapsed_ms
        }
