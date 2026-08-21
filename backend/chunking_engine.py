"""
Chunking Engine for MSMARCO-XI RAG System
==========================================
Four advanced chunking strategies, all using word-token counts
(not characters) for meaningful, consistent chunk sizing:

  1. Fixed-token overlapping window
  2. Semantic sentence-boundary chunking (Indic-aware)
  3. Metadata-aware hierarchical chunking
  4. Parent-child / multi-vector chunking

Key design decisions:
  - Chunk size measured in WORDS (≈ tokens), not characters
  - Overlap measured as a fraction of chunk size
  - Indic sentence boundaries (। ॥) handled natively
  - Parent-child: child chunks are embedded for retrieval precision;
    parent text is stored for LLM context richness
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("chunking_engine")


# ── Sentence Splitter ─────────────────────────────────────────────────────────

# Matches end-of-sentence positions for:
#   .  !  ?          — English / Latin scripts
#   ।  ॥             — Devanagari (Hindi, Marathi, Sanskrit)
#   ।  (same Unicode U+0964 / U+0965) used in Bengali, Gujarati, etc.
_SENTENCE_END = re.compile(r'(?<=[.!?।॥])\s+')


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences at natural boundaries."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    """Count whitespace-delimited tokens (words) in text."""
    return len(text.split())


class ChunkingEngine:
    """
    Multi-strategy document chunker with word-token-based sizing.

    All strategies accept `chunk_size` (words) and `chunk_overlap`
    (words, or computed as 10–20% of chunk_size when not supplied).
    """

    def __init__(self):
        # Default overlap = 15% of chunk_size; callers can override
        self._default_overlap_ratio = 0.15

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "semantic",
        chunk_size: int = 256,          # words
        chunk_overlap: Optional[int] = None,  # words; defaults to 15%
    ) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents using the specified strategy.

        Args:
            documents:     List of document dicts (from corpus.jsonl or built-ins).
            strategy:      One of: 'fixed', 'semantic', 'metadata_aware', 'parent_child'.
            chunk_size:    Target chunk size in words. Default: 256.
            chunk_overlap: Overlap in words. Default: 15% of chunk_size.

        Returns:
            List of chunk dicts, each with keys:
              chunk_id, text, parent_text, metadata
        """
        if chunk_overlap is None:
            chunk_overlap = max(1, int(chunk_size * self._default_overlap_ratio))

        t0 = time.perf_counter()
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("id") or doc.get("passage_id") or doc.get("doc_id") or "doc_unknown"
            p_native = (doc.get("passage", "") or doc.get("text", "")).strip()
            p_en = doc.get("passage_en", "").strip()
            if p_native and p_en and p_en != p_native:
                text = f"{p_native}\n\n{p_en}"
            else:
                text = p_native or p_en

            if not text:
                continue

            lang_code = doc.get("language") or doc.get("source_lang") or "en"
            lang_map = {"hi": "Hindi", "en": "English", "mr": "Marathi", "te": "Telugu", "ta": "Tamil", "bn": "Bengali"}
            lang_name = doc.get("lang_name") or lang_map.get(lang_code, lang_code.upper())

            metadata = {
                "doc_id": doc_id,
                "language": lang_code,
                "lang_name": lang_name,
                "query_context": doc.get("query", "") or doc.get("title", ""),
                "query_en": doc.get("query_en", ""),
                "answers": doc.get("answers", []),
                "source": "ai4bharat/MSMARCO-XI",
                "chunking_strategy": strategy,
                "chunk_size_words": chunk_size,
                "chunk_overlap_words": chunk_overlap,
            }

            if strategy == "fixed":
                doc_chunks = self._fixed_token_chunking(text, metadata, chunk_size, chunk_overlap)
            elif strategy == "semantic":
                doc_chunks = self._semantic_boundary_chunking(text, metadata, chunk_size, chunk_overlap)
            elif strategy == "metadata_aware":
                doc_chunks = self._metadata_aware_chunking(text, metadata, chunk_size, chunk_overlap)
            elif strategy == "parent_child":
                doc_chunks = self._parent_child_chunking(text, metadata, chunk_size)
            else:
                logger.warning(f"Unknown strategy '{strategy}', falling back to 'semantic'.")
                doc_chunks = self._semantic_boundary_chunking(text, metadata, chunk_size, chunk_overlap)

            chunks.extend(doc_chunks)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        logger.info(
            f"ChunkingEngine [{strategy}, sz={chunk_size}w, ov={chunk_overlap}w]: "
            f"{len(documents)} docs → {len(chunks)} chunks in {elapsed_ms}ms"
        )
        return chunks

    def compare_strategies(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 256,
    ) -> Dict[str, Any]:
        """
        Run all four strategies on `documents` and return comparison stats.
        Used by the /api/chunking/compare endpoint.
        """
        strategies = ["fixed", "semantic", "metadata_aware", "parent_child"]
        results = {}

        for strategy in strategies:
            t0 = time.perf_counter()
            chunks = self.chunk_documents(documents, strategy=strategy, chunk_size=chunk_size)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

            if chunks:
                word_counts = [_word_count(c["text"]) for c in chunks]
                avg_words = round(sum(word_counts) / len(word_counts), 1)
                char_counts = [len(c["text"]) for c in chunks]
                avg_chars = round(sum(char_counts) / len(char_counts), 1)
            else:
                word_counts = [0]
                avg_words = 0
                char_counts = [0]
                avg_chars = 0

            results[strategy] = {
                "strategy": strategy,
                "total_chunks": len(chunks),
                "avg_words_per_chunk": avg_words,
                "min_words": min(word_counts),
                "max_words": max(word_counts),
                "avg_chars_per_chunk": avg_chars,
                "processing_ms": elapsed_ms,
                "sample_chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "word_count": _word_count(c["text"]),
                        "text_snippet": c["text"][:150] + ("..." if len(c["text"]) > 150 else ""),
                    }
                    for c in chunks[:3]
                ],
            }

        return results

    # ── Strategy 1: Fixed-Token Overlapping Window ────────────────────────────

    def _fixed_token_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping windows of exactly `chunk_size` words.

        This is the baseline strategy — fast and predictable, but agnostic
        to sentence and paragraph boundaries.
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        chunk_idx = 0
        step = max(1, chunk_size - chunk_overlap)

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"{metadata['doc_id']}_fixed_{chunk_idx:04d}",
                "text": chunk_text,
                "parent_text": text,
                "metadata": {
                    **metadata,
                    "chunk_idx": chunk_idx,
                    "word_start": start,
                    "word_end": end,
                    "word_count": len(chunk_words),
                },
            })

            if end >= len(words):
                break
            start += step
            chunk_idx += 1

        return chunks

    # ── Strategy 2: Semantic Sentence-Boundary Chunking ───────────────────────

    def _semantic_boundary_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        target_words: int,
        overlap_words: int,
    ) -> List[Dict[str, Any]]:
        """
        Split text at natural sentence boundaries (English + Indic punctuation).

        Accumulates sentences until the target word count is reached,
        then starts a new chunk. Overlap is implemented by carrying forward
        the last `overlap_words` words of the previous chunk as a prefix.

        This preserves semantic coherence — no mid-sentence cuts.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return self._fixed_token_chunking(text, metadata, target_words, overlap_words)

        chunks = []
        current_sentences: List[str] = []
        current_words = 0
        chunk_idx = 0
        # Words carried over from the previous chunk for overlap
        overlap_prefix: List[str] = []

        def _flush(sentences_list: List[str], prefix: List[str]) -> str:
            all_words = prefix + " ".join(sentences_list).split()
            return " ".join(all_words)

        for sentence in sentences:
            s_words = _word_count(sentence)

            # If adding this sentence would exceed the target, emit current chunk
            if current_words + s_words > target_words and current_sentences:
                chunk_text = _flush(current_sentences, overlap_prefix)
                chunks.append({
                    "chunk_id": f"{metadata['doc_id']}_semantic_{chunk_idx:04d}",
                    "text": chunk_text,
                    "parent_text": text,
                    "metadata": {
                        **metadata,
                        "chunk_idx": chunk_idx,
                        "sentence_count": len(current_sentences),
                        "word_count": _word_count(chunk_text),
                    },
                })
                chunk_idx += 1

                # Build overlap prefix from end of this chunk
                all_chunk_words = chunk_text.split()
                overlap_prefix = all_chunk_words[-overlap_words:] if overlap_words > 0 else []
                current_sentences = []
                current_words = 0

            current_sentences.append(sentence)
            current_words += s_words

        # Flush remaining sentences
        if current_sentences:
            chunk_text = _flush(current_sentences, overlap_prefix)
            chunks.append({
                "chunk_id": f"{metadata['doc_id']}_semantic_{chunk_idx:04d}",
                "text": chunk_text,
                "parent_text": text,
                "metadata": {
                    **metadata,
                    "chunk_idx": chunk_idx,
                    "sentence_count": len(current_sentences),
                    "word_count": _word_count(chunk_text),
                },
            })

        return chunks

    # ── Strategy 3: Metadata-Aware Chunking ───────────────────────────────────

    def _metadata_aware_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        target_words: int,
        overlap_words: int,
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking with a metadata context prefix injected into the
        embedded text.

        The prefix encodes document-level signals (language, corpus source,
        query context) into the embedding, improving retrieval precision for
        multilingual and cross-lingual queries.

        The LLM always receives the raw chunk text WITHOUT the prefix
        (stored in 'raw_text') to avoid polluting the generation context.
        """
        # Build prefix (kept short to avoid dominating the embedding)
        lang = metadata.get("lang_name", "")
        doc_id = metadata.get("doc_id", "")
        query_ctx = (metadata.get("query_context") or metadata.get("query_en") or "")[:80]
        prefix = f"[{lang} | MSMARCO-XI | {doc_id} | {query_ctx}] "

        # Get base semantic chunks
        base_chunks = self._semantic_boundary_chunking(text, metadata, target_words, overlap_words)

        enriched = []
        for idx, chunk in enumerate(base_chunks):
            enriched_text = prefix + chunk["text"]
            enriched.append({
                "chunk_id": f"{metadata['doc_id']}_metaaware_{idx:04d}",
                "text": enriched_text,          # Used for embedding
                "raw_text": chunk["text"],       # Used for LLM context
                "parent_text": text,
                "metadata": {
                    **chunk["metadata"],
                    "chunking_strategy": "metadata_aware",
                    "has_metadata_prefix": True,
                    "prefix": prefix,
                },
            })

        return enriched

    # ── Strategy 4: Parent-Child / Multi-Vector Chunking ─────────────────────

    def _parent_child_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        parent_size_words: int,
    ) -> List[Dict[str, Any]]:
        """
        Two-level chunking:
          - Parent chunks: ~parent_size_words, semantic boundaries
            → stored as context for the LLM
          - Child chunks: ~1/3 of parent size, fixed-token
            → embedded and retrieved with high precision

        This gives the best of both worlds:
          - High recall: small child chunks → precise embedding matches
          - Rich context: parent passage → better LLM answers
          - Low hallucination: LLM sees full sentence context, not a fragment

        The reranker scores using parent_text (see reranker.py) for even
        richer signal, then returns child chunk for citation accuracy.
        """
        child_size = max(32, parent_size_words // 3)
        child_overlap = max(1, child_size // 5)

        # Step 1: build parent chunks using semantic boundaries
        parent_chunks = self._semantic_boundary_chunking(
            text, metadata, parent_size_words, max(1, parent_size_words // 10)
        )

        all_child_chunks = []

        for p_idx, parent in enumerate(parent_chunks):
            parent_text = parent["text"]
            parent_word_count = _word_count(parent_text)

            # Step 2: split parent into child chunks (fixed-token within parent)
            parent_words = parent_text.split()
            step = max(1, child_size - child_overlap)
            c_idx = 0
            start = 0

            while start < len(parent_words):
                end = min(start + child_size, len(parent_words))
                child_text = " ".join(parent_words[start:end])

                all_child_chunks.append({
                    "chunk_id": f"{metadata['doc_id']}_parchild_{p_idx:03d}_{c_idx:04d}",
                    "text": child_text,              # Embedded for retrieval
                    "parent_text": parent_text,      # Given to LLM for answering
                    "metadata": {
                        **metadata,
                        "chunking_strategy": "parent_child",
                        "parent_idx": p_idx,
                        "child_idx": c_idx,
                        "parent_word_count": parent_word_count,
                        "child_word_count": len(child_text.split()),
                        "is_child": True,
                    },
                })

                if end >= len(parent_words):
                    break
                start += step
                c_idx += 1

        return all_child_chunks
