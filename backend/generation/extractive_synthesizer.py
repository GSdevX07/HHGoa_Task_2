"""
Non-LLM Extractive Answer Synthesizer
======================================
Produces accurate, concise, grounded answers in 1–3 ms without any cloud API network latency.
Supports English and all Indic languages (Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, etc.).
"""

import re
import time
import unicodedata
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("generation.synthesizer")

# Indic + English sentence boundary regex
SENTENCE_SPLIT_REGEX = re.compile(r"[\n\r]+|(?<=[.!?।॥])\s+")

STOPWORDS = {
    "what", "is", "the", "of", "and", "a", "to", "in", "for", "are", "on", "with",
    "as", "by", "at", "from", "how", "where", "who", "which", "why", "when",
    "का", "के", "की", "है", "हैं", "में", "से", "को", "पर", "यह", "और", "एक", "क्या",
    "ఉంది", "యొక్క", "మరియు", "అనేది", "ஆகும்", "மற்றும்", "என்பது"
}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences supporting Indic punctuation (danda ।) and Latin punctuation."""
    if not text:
        return []
    raw = SENTENCE_SPLIT_REGEX.split(text.strip())
    sentences = []
    for s in raw:
        cleaned = s.strip()
        if len(cleaned) > 10:
            sentences.append(cleaned)
    return sentences if sentences else [text.strip()]


def extract_query_keywords(query: str) -> set:
    """Extract content words from query for lexical matching."""
    cleaned = unicodedata.normalize("NFKC", query).lower()
    tokens = re.findall(r"[\w\u0900-\u0D7F]+", cleaned)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


class ExtractiveSynthesizer:
    """
    Extracts the most relevant, concise, and coherent answer from retrieved passages.
    Target latency: < 2 ms.
    """

    def synthesize(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_sentences: int = 2,
    ) -> Tuple[str, float]:
        """
        Synthesize answer from retrieved passages.
        Returns (answer_string, latency_ms).
        """
        t0 = time.perf_counter()
        if not retrieved_results:
            return "", round((time.perf_counter() - t0) * 1000, 3)

        query_keywords = extract_query_keywords(query)

        # Collect all candidate sentences from the top passages
        scored_sentences = []
        seen_sentences = set()

        for passage_rank, r in enumerate(retrieved_results[:3]):
            passage_text = r.get("parent_text") or r.get("text", "")
            base_score = r.get("reranker_score", r.get("rrf_score", r.get("dense_score", 0.5)))
            sentences = split_into_sentences(passage_text)

            for s_idx, sentence in enumerate(sentences):
                norm_s = sentence.lower()
                if norm_s in seen_sentences:
                    continue
                seen_sentences.add(norm_s)

                s_tokens = set(re.findall(r"[\w\u0900-\u0D7F]+", norm_s))
                
                # 1. Keyword overlap score
                overlap_count = sum(1 for kw in query_keywords if kw in s_tokens or any(kw in st for st in s_tokens))
                overlap_ratio = overlap_count / max(len(query_keywords), 1)

                # 2. Position bias (first sentence in passage often contains main definition)
                position_bonus = 0.35 if s_idx == 0 else (0.15 if s_idx == 1 else 0.0)

                # 3. Passage rank weighting
                rank_weight = 1.0 / (1.0 + 0.5 * passage_rank)

                # 4. Total relevance score
                total_score = (overlap_ratio * 2.0 + position_bonus + base_score * 0.5) * rank_weight

                scored_sentences.append({
                    "text": sentence,
                    "score": total_score,
                    "overlap": overlap_count,
                    "passage_rank": passage_rank,
                    "sentence_idx": s_idx,
                })

        if not scored_sentences:
            return "", round((time.perf_counter() - t0) * 1000, 3)

        # Sort by total score descending
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)

        # Pick top 1-2 coherent sentences
        selected = [scored_sentences[0]["text"]]
        if max_sentences > 1 and len(scored_sentences) > 1:
            best_first = scored_sentences[0]
            # Look for a complementary second sentence from the same top passage if high quality
            for cand in scored_sentences[1:]:
                # Check for low redundancy with first sentence
                words1 = set(selected[0].split())
                words2 = set(cand["text"].split())
                jaccard = len(words1 & words2) / max(len(words1 | words2), 1)
                if jaccard < 0.6 and cand["overlap"] >= 1:
                    selected.append(cand["text"])
                    break

        answer = " ".join(selected).strip()
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        return answer, elapsed_ms
