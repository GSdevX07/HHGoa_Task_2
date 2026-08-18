"""
Multi-Layer Guardrail Engine for Voice-Enabled RAG System
=========================================================
Four independently composable guardrail layers:

  Layer 1 — Pre-execution safety:
    • Prompt injection detection (regex patterns)
    • Domain relevance check (embedding cosine similarity vs. domain anchors)
    • Query length / quality validation

  Layer 2 — Retrieval confidence:
    • Refuses to generate when top retrieval score is below threshold
    • Prevents hallucination on uncovered topics

  Layer 3 — Post-generation groundedness:
    • Sentence-level claim verification using embedding similarity
    • Each claim in the answer is matched against retrieved passages
    • Ungrounded claims are flagged; if ratio is too high, answer is rejected

  Layer 4 — (Wired in main.py) Output formatting safety

Design goals:
  • All checks run in < 10ms total (no external API calls)
  • Domain relevance uses embedding cache — anchors are embedded once at startup
  • Claim-level grounding replaces the naïve bag-of-words overlap
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("guardrails")


# ── Domain Anchor Passages ────────────────────────────────────────────────────
# These are representative in-domain passages. Queries whose embeddings are
# far from ALL anchors are considered off-topic.
_DOMAIN_ANCHORS = [
    "Information retrieval from documents using vector search and dense embeddings.",
    "Question answering over multilingual passages from the MSMARCO dataset.",
    "History, geography, science, culture, people, places, and events in India.",
    "Indic languages including Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Malayalam, Kannada.",
    "Technology, artificial intelligence, machine learning, and data science topics.",
    "Government, politics, economy, education, and public institutions of India.",
    "Sports, arts, literature, religion, and traditions in South Asia.",
    "Natural language processing, search engines, and information systems.",
    "General knowledge questions similar to MS MARCO benchmark queries.",
]

# Cosine similarity threshold below which a query is flagged off-topic
_DOMAIN_THRESHOLD = 0.18


class GuardrailEngine:
    """
    Multi-layer RAG guardrail system.

    The embedding model is the same as the dense retriever (all-MiniLM-L6-v2)
    so there's no extra model weight to load — we just reuse the shared instance.
    """

    def __init__(self, embedder=None):
        """
        Args:
            embedder: Optional callable that takes a list of strings and
                      returns a (N, dim) float32 numpy array of normalised
                      embeddings. If None, falls back to keyword heuristics.
                      Typically pass the DenseRetriever._embed method.
        """
        self._embedder = embedder
        self._anchor_embeddings: Optional[np.ndarray] = None

        # ── Prompt Injection Patterns ─────────────────────────────────────────
        self._injection_patterns: List[re.Pattern] = [
            re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", re.I),
            re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
            re.compile(r"system\s*(prompt|message|override|hack)", re.I),
            re.compile(r"(admin|root|sudo|developer)\s*(mode|access|override)", re.I),
            re.compile(r"bypass\s+(safety|filter|guardrail|restriction)", re.I),
            re.compile(r"jailbreak|dan\s+mode|do\s+anything\s+now", re.I),
            re.compile(r"output\s+(raw|hidden|internal)\s+(prompt|instructions)", re.I),
            re.compile(r"forget\s+(all\s+)?(your\s+)?(previous\s+)?(rules|training)", re.I),
            re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
        ]

        # Pre-embed domain anchors if embedder is available
        if self._embedder is not None:
            self._embed_anchors()

    def set_embedder(self, embedder):
        """Set or replace the embedding callable and pre-embed anchors."""
        self._embedder = embedder
        self._anchor_embeddings = None
        self._embed_anchors()

    def _embed_anchors(self):
        """Embed domain anchor phrases once at startup."""
        if self._embedder is None:
            return
        try:
            self._anchor_embeddings = self._embedder(_DOMAIN_ANCHORS)
            logger.info(f"GuardrailEngine: pre-embedded {len(_DOMAIN_ANCHORS)} domain anchors.")
        except Exception as exc:
            logger.warning(f"GuardrailEngine: anchor embedding failed — {exc}. "
                           "Domain relevance check will use keyword heuristics.")
            self._anchor_embeddings = None

    # ── Layer 1: Input Validation ─────────────────────────────────────────────

    def validate_input(self, query: str) -> Dict[str, Any]:
        """
        Pre-execution guardrail: validate query safety and domain relevance.

        Returns a dict with:
            passed (bool), reason (str), message (str), latency_ms (float)
        """
        t0 = time.perf_counter()
        query_clean = query.strip()

        # Empty query
        if not query_clean:
            return self._fail("empty_query", "Query cannot be empty.", t0)

        # Too short (likely noise / mic artifact)
        if len(query_clean.split()) < 2:
            return self._fail(
                "query_too_short",
                "Query is too short to be meaningful. Please ask a complete question.",
                t0,
            )

        # Too long (potential prompt stuffing)
        if len(query_clean) > 2000:
            return self._fail(
                "query_too_long",
                "Query exceeds the maximum allowed length of 2000 characters.",
                t0,
            )

        # Prompt injection check
        for pattern in self._injection_patterns:
            if pattern.search(query_clean):
                return self._fail(
                    "prompt_injection",
                    "Security alert: your query contains patterns that are not permitted.",
                    t0,
                )

        # Domain relevance check
        relevance = self._check_domain_relevance(query_clean)
        if not relevance["is_relevant"]:
            return self._fail(
                "off_topic",
                (
                    "Your question appears to be outside the scope of the MSMARCO-XI "
                    "knowledge base. Please ask questions about general knowledge, "
                    "Indian history, culture, science, technology, or similar topics."
                ),
                t0,
                extra={"max_anchor_similarity": relevance["max_similarity"]},
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "passed": True,
            "reason": "clean_input",
            "message": "Query passed all pre-execution safety and domain guardrails.",
            "latency_ms": elapsed_ms,
            "domain_similarity": relevance["max_similarity"],
        }

    def _check_domain_relevance(self, query: str) -> Dict[str, Any]:
        """
        Check if the query is semantically related to the domain.

        Uses embedding cosine similarity against pre-embedded domain anchors.
        Falls back to keyword heuristics if no embedder is available.
        """
        # Keyword-based fallback (when no embedder available)
        if self._embedder is None or self._anchor_embeddings is None:
            return self._keyword_relevance_check(query)

        try:
            q_emb = self._embedder([query])[0]  # shape (dim,)
            # Cosine sim (anchors are already normalised, q_emb may need normalising)
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
            sims = np.dot(self._anchor_embeddings, q_emb)  # (N_anchors,)
            max_sim = float(np.max(sims))
            return {
                "is_relevant": max_sim >= _DOMAIN_THRESHOLD,
                "max_similarity": round(max_sim, 4),
                "method": "embedding",
            }
        except Exception as exc:
            logger.warning(f"GuardrailEngine: embedding-based relevance check failed — {exc}")
            return self._keyword_relevance_check(query)

    def _keyword_relevance_check(self, query: str) -> Dict[str, Any]:
        """
        Simple keyword heuristic fallback when embedder is unavailable.
        Blocks only the most obvious off-topic requests.
        """
        query_lower = query.lower()
        OFF_TOPIC_TRIGGERS = [
            r"buy\s+(crypto|bitcoin|ethereum|nft|stock)",
            r"(dirty|sexual|adult|nsfw)\s+(joke|content|video)",
            r"write\s+(a\s+)?(virus|malware|exploit|ransomware|keylogger)",
            r"hack\s+(into|the)\s+",
            r"how\s+to\s+(steal|commit|kill)",
        ]
        for pattern in OFF_TOPIC_TRIGGERS:
            if re.search(pattern, query_lower):
                return {"is_relevant": False, "max_similarity": 0.0, "method": "keyword"}
        return {"is_relevant": True, "max_similarity": 1.0, "method": "keyword"}

    # ── Layer 2: Retrieval Confidence ─────────────────────────────────────────

    def validate_retrieved_context(
        self,
        results: List[Dict[str, Any]],
        top_score: float,
        threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Refuse to generate if the best-matching retrieved passage score is
        below the confidence threshold.

        This prevents the LLM from hallucinating when the corpus genuinely
        doesn't contain a relevant answer.
        """
        t0 = time.perf_counter()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

        if not results:
            return {
                "should_refuse": True,
                "refusal_reason": "no_results",
                "refusal_message": (
                    "I couldn't find any relevant information in the knowledge base "
                    "to answer your question."
                ),
                "latency_ms": elapsed_ms,
                "top_score": top_score,
            }

        if top_score < threshold:
            return {
                "should_refuse": True,
                "refusal_reason": "low_retrieval_confidence",
                "refusal_message": (
                    f"I found results in the knowledge base, but none were sufficiently "
                    f"relevant to your question (best match: {top_score:.2f} < "
                    f"threshold {threshold:.2f}). I cannot give a reliable answer."
                ),
                "latency_ms": elapsed_ms,
                "top_score": top_score,
            }

        return {
            "should_refuse": False,
            "refusal_reason": None,
            "refusal_message": None,
            "latency_ms": elapsed_ms,
            "top_score": top_score,
        }

    # ── Layer 3: Post-Generation Groundedness ─────────────────────────────────

    def verify_groundedness(
        self,
        generated_answer: str,
        retrieved_passages: List[str],
        similarity_threshold: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Sentence-level claim verification.

        Algorithm:
          1. Split the answer into individual claim sentences.
          2. Split each retrieved passage into sentences.
          3. For each claim sentence, compute embedding cosine similarity
             against all passage sentences.
          4. A claim is "grounded" if its max similarity to any passage
             sentence exceeds `similarity_threshold`.
          5. groundedness_score = grounded_claims / total_claims

        This is far superior to bag-of-words overlap because it:
          - Handles paraphrasing ("founded in 1969" ↔ "established in 1969")
          - Works across languages
          - Can detect claims that add information not in context
        """
        t0 = time.perf_counter()

        if not generated_answer or not retrieved_passages:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "groundedness_score": 0.0,
                "is_grounded": False,
                "flagged": True,
                "grounded_claims": 0,
                "total_claims": 0,
                "claim_details": [],
                "latency_ms": elapsed_ms,
            }

        # Check for explicit refusal phrases — always grounded
        answer_lower = generated_answer.lower()
        explicit_refusals = [
            "cannot find a grounded answer",
            "not enough information",
            "context does not contain",
            "provided passages do not",
        ]
        if any(phrase in answer_lower for phrase in explicit_refusals):
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            return {
                "groundedness_score": 1.0,
                "is_grounded": True,
                "flagged": False,
                "grounded_claims": 1,
                "total_claims": 1,
                "claim_details": [{"claim": "explicit_refusal", "grounded": True, "max_sim": 1.0}],
                "latency_ms": elapsed_ms,
            }

        if self._embedder is None:
            # Fallback: word overlap groundedness
            return self._word_overlap_groundedness(generated_answer, retrieved_passages, t0)

        try:
            # Split answer into claim sentences
            claim_sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', generated_answer)
                               if s.strip() and len(s.split()) > 3]

            if not claim_sentences:
                # Single sentence or very short answer — treat whole answer as one claim
                claim_sentences = [generated_answer.strip()]

            # Split passages into sentences for fine-grained matching
            passage_sentences = []
            for passage in retrieved_passages:
                sents = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', passage) if s.strip()]
                passage_sentences.extend(sents)
                # Also add the full passage as a candidate for matching
                if passage.strip():
                    passage_sentences.append(passage.strip())

            if not passage_sentences:
                passage_sentences = retrieved_passages

            # Embed everything in two batches
            all_texts = claim_sentences + passage_sentences
            all_embeddings = self._embedder(all_texts)
            # Normalise
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            all_embeddings = all_embeddings / norms

            claim_embs = all_embeddings[:len(claim_sentences)]
            passage_embs = all_embeddings[len(claim_sentences):]

            # For each claim, find max cosine similarity to any passage sentence
            sim_matrix = np.dot(claim_embs, passage_embs.T)  # (n_claims, n_passages)

            claim_details = []
            grounded_count = 0

            for i, claim in enumerate(claim_sentences):
                max_sim = float(sim_matrix[i].max()) if sim_matrix.shape[1] > 0 else 0.0
                is_grounded = max_sim >= similarity_threshold
                if is_grounded:
                    grounded_count += 1
                claim_details.append({
                    "claim": claim[:100],
                    "grounded": is_grounded,
                    "max_sim": round(max_sim, 4),
                })

            score = round(grounded_count / len(claim_sentences), 4) if claim_sentences else 0.0
            is_grounded = score >= 0.60  # ≥60% of claims must be grounded

        except Exception as exc:
            logger.warning(f"GuardrailEngine: embedding-based groundedness failed — {exc}. "
                           "Falling back to word overlap.")
            return self._word_overlap_groundedness(generated_answer, retrieved_passages, t0)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "groundedness_score": score,
            "is_grounded": is_grounded,
            "flagged": not is_grounded,
            "grounded_claims": grounded_count,
            "total_claims": len(claim_sentences),
            "claim_details": claim_details,
            "warning": (
                None if is_grounded else
                f"Potential hallucination: only {grounded_count}/{len(claim_sentences)} "
                f"claims are grounded in retrieved context."
            ),
            "latency_ms": elapsed_ms,
        }

    def _word_overlap_groundedness(
        self,
        answer: str,
        passages: List[str],
        t0: float,
    ) -> Dict[str, Any]:
        """
        Fallback word-overlap groundedness when embedder is unavailable.
        Better than the original: uses content words only (length > 3),
        removes numbers that could be hallucinated.
        """
        combined = " ".join(passages).lower()
        context_words = set(w for w in re.findall(r"\b\w{4,}\b", combined)
                            if not w.isdigit())

        answer_words = [w for w in re.findall(r"\b\w{4,}\b", answer.lower())
                        if not w.isdigit()]

        if not answer_words:
            score = 1.0
        else:
            grounded = sum(1 for w in answer_words if w in context_words)
            score = round(grounded / len(answer_words), 4)

        is_grounded = score >= 0.50
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

        return {
            "groundedness_score": score,
            "is_grounded": is_grounded,
            "flagged": not is_grounded,
            "grounded_claims": int(score * len(answer_words)) if answer_words else 0,
            "total_claims": len(answer_words),
            "claim_details": [],
            "warning": (
                None if is_grounded else
                "Potential hallucination detected (word overlap heuristic)."
            ),
            "latency_ms": elapsed_ms,
            "method": "word_overlap_fallback",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fail(reason: str, message: str, t0: float, extra: dict = None) -> Dict[str, Any]:
        result = {
            "passed": False,
            "reason": reason,
            "message": message,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
        if extra:
            result.update(extra)
        return result
