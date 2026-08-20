"""
backend/generation package
==========================
LLM prompts, Extractive Synthesizer, and Cache Engines.
"""
from .prompts import SYSTEM_PROMPT, build_user_message, REFUSAL_PHRASES
from .extractive_synthesizer import ExtractiveSynthesizer
from .cache_engine import ExactCache, SemanticCache

__all__ = [
    "SYSTEM_PROMPT", "build_user_message", "REFUSAL_PHRASES",
    "ExtractiveSynthesizer", "ExactCache", "SemanticCache"
]
