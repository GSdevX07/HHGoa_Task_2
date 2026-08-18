"""
backend/generation package
==========================
LLM prompt templates and generation utilities.
"""
from .prompts import SYSTEM_PROMPT, build_user_message, REFUSAL_PHRASES

__all__ = ["SYSTEM_PROMPT", "build_user_message", "REFUSAL_PHRASES"]
