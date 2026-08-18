"""
Generation Prompts for Voice-Enabled RAG System
================================================
Strict, context-grounded prompts for LLM answer synthesis.

Design principles:
  1. Answers must be derivable from the supplied context ONLY.
  2. Voice-optimized: no markdown, no bullet points, no headers.
  3. Concise: 1–3 sentences for most factual questions.
  4. Multilingual-aware: respond in the query language when context supports it.
  5. Explicit refusal when context is insufficient (never hallucinate).
"""

from typing import List

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, voice-optimized question-answering assistant for the AI4Bharat MSMARCO-XI multilingual dataset.

STRICT RULES (violating any rule is a critical failure):
1. Answer ONLY using information explicitly present in the provided CONTEXT PASSAGES.
2. If the passages do not contain enough information to answer the question, respond exactly with: "I cannot find a grounded answer for this question in the available context."
3. Do NOT invent, extrapolate, or add any facts, dates, names, statistics, or claims not supported by the context.
4. Keep your answer to 1–3 clear sentences. This is a voice system — no bullet points, no numbered lists, no markdown formatting of any kind.
5. If the question is in Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Malayalam, Kannada, Punjabi, Odia, or another Indic language, AND the relevant context passage is in that language, answer in that same language.
6. If the context is in English but the question is in an Indic language, you may answer in English or translate the answer — but do not add information not present in the context.
7. Cite your source implicitly — naturally integrate the information from the context into your answer. Do not say "according to passage 1" or similar.
8. If multiple passages are provided, synthesize them coherently rather than repeating each one.

Remember: A grounded refusal is far better than a hallucinated answer."""


# ── Refusal Phrases ───────────────────────────────────────────────────────────

REFUSAL_PHRASES = [
    "I cannot find a grounded answer",
    "not enough information in the context",
    "the context does not contain",
    "the provided passages do not",
    "no relevant information found",
    "context insufficient",
]


# ── Message Builder ───────────────────────────────────────────────────────────

def build_user_message(query: str, passages: List[str]) -> str:
    """
    Construct the user turn for the LLM.

    Formats retrieved passages clearly with numbered labels so the model
    can reference them without confusion in a multi-passage scenario.

    Args:
        query:    The user's transcribed question.
        passages: List of retrieved passage texts (typically 3).

    Returns:
        A formatted user message string.
    """
    if not passages:
        context_block = "[No context passages retrieved.]"
    else:
        labeled = []
        for i, p in enumerate(passages, 1):
            # Truncate very long passages to stay within token budget
            truncated = p.strip()[:1200]
            labeled.append(f"[Passage {i}]\n{truncated}")
        context_block = "\n\n".join(labeled)

    return (
        f"CONTEXT PASSAGES:\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n\n"
        f"QUESTION: {query.strip()}\n\n"
        f"Answer (using only the passages above):"
    )


def build_groq_payload(query: str, passages: List[str], model: str = "llama-3.3-70b-versatile") -> dict:
    """
    Build a complete Groq-compatible chat completion request payload.

    Args:
        query:    User's question.
        passages: Retrieved context passages.
        model:    Groq model name.

    Returns:
        JSON-serializable dict for the /v1/chat/completions endpoint.
    """
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, passages)},
        ],
        "temperature": 0.1,       # Low temperature for factual grounding
        "max_tokens": 200,        # Voice answers should be concise
        "top_p": 0.95,
        "stream": False,          # Non-streaming for latency measurement accuracy
    }


def build_openai_payload(query: str, passages: List[str], model: str = "gpt-4o-mini") -> dict:
    """
    Build an OpenAI-compatible chat completion payload.
    Same structure as Groq since Groq uses the OpenAI API format.
    """
    return build_groq_payload(query, passages, model=model)
