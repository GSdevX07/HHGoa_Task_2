"""
Vast Chunking Engine for MSMARCO-XI RAG System
Implements multiple advanced chunking strategies:
1. Fixed-Size Overlapping Window Chunking
2. Semantic Boundary Chunking (Sentence / Danda aware)
3. Metadata-Aware Hierarchical Chunking
4. Parent-Child / Multi-Vector Chunking
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("chunking_engine")

class ChunkingEngine:
    def __init__(self):
        # Regex for sentence boundaries supporting English (.!?) and Indic languages (।, ॥)
        self.sentence_regex = re.compile(r'(?<=[.!?।॥])\s+')

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        strategy: str = "semantic",
        chunk_size: int = 200,
        chunk_overlap: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for document chunking.
        Strategies: 'fixed', 'semantic', 'metadata_aware', 'parent_child'
        """
        start_time = time.perf_counter()
        chunks = []

        for doc in documents:
            doc_id = doc.get("id", "doc_unknown")
            text = doc.get("passage", "") or doc.get("passage_en", "")
            metadata = {
                "doc_id": doc_id,
                "language": doc.get("language", "en"),
                "lang_name": doc.get("lang_name", "English"),
                "query_context": doc.get("query", ""),
                "answers": doc.get("answers", []),
                "source": "ai4bharat/MSMARCO-XI"
            }

            if not text.strip():
                continue

            if strategy == "fixed":
                doc_chunks = self._fixed_size_chunking(text, metadata, chunk_size, chunk_overlap)
            elif strategy == "semantic":
                doc_chunks = self._semantic_boundary_chunking(text, metadata, chunk_size)
            elif strategy == "metadata_aware":
                doc_chunks = self._metadata_aware_chunking(text, metadata, chunk_size)
            elif strategy == "parent_child":
                doc_chunks = self._parent_child_chunking(text, metadata, chunk_size)
            else:
                doc_chunks = self._semantic_boundary_chunking(text, metadata, chunk_size)

            chunks.extend(doc_chunks)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 3)
        logger.info(f"Strategy '{strategy}' generated {len(chunks)} chunks from {len(documents)} docs in {elapsed_ms}ms")
        return chunks

    def _fixed_size_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Splits text into fixed character blocks with specified overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]
            
            chunks.append({
                "chunk_id": f"{metadata['doc_id']}_fixed_{chunk_idx}",
                "text": chunk_text,
                "metadata": {**metadata, "strategy": "fixed", "chunk_idx": chunk_idx, "char_start": start, "char_end": end},
                "parent_text": text
            })
            
            if end >= text_len:
                break
            start += (chunk_size - chunk_overlap)
            chunk_idx += 1

        return chunks

    def _semantic_boundary_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """Splits text at natural sentence boundaries (supporting Indic punctuation '।' and '॥')."""
        sentences = self.sentence_regex.split(text)
        chunks = []
        current_chunk_sentences = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)
            if current_len + sentence_len > target_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id": f"{metadata['doc_id']}_semantic_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {**metadata, "strategy": "semantic", "chunk_idx": chunk_idx, "sentence_count": len(current_chunk_sentences)},
                    "parent_text": text
                })
                chunk_idx += 1
                current_chunk_sentences = []
                current_len = 0

            current_chunk_sentences.append(sentence)
            current_len += sentence_len + 1

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": f"{metadata['doc_id']}_semantic_{chunk_idx}",
                "text": chunk_text,
                "metadata": {**metadata, "strategy": "semantic", "chunk_idx": chunk_idx, "sentence_count": len(current_chunk_sentences)},
                "parent_text": text
            })

        return chunks

    def _metadata_aware_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        target_size: int
    ) -> List[Dict[str, Any]]:
        """Prepends document metadata context header into chunk embedding text to enrich semantic search."""
        base_chunks = self._semantic_boundary_chunking(text, metadata, target_size)
        header_prefix = f"[{metadata['lang_name']} Corpus | ID: {metadata['doc_id']} | Query: {metadata['query_context']}] "

        enriched_chunks = []
        for idx, chk in enumerate(base_chunks):
            enriched_text = header_prefix + chk["text"]
            enriched_chunks.append({
                "chunk_id": f"{metadata['doc_id']}_meta_{idx}",
                "text": enriched_text,
                "raw_text": chk["text"],
                "metadata": {**metadata, "strategy": "metadata_aware", "has_header": True},
                "parent_text": text
            })

        return enriched_chunks

    def _parent_child_chunking(
        self,
        text: str,
        metadata: Dict[str, Any],
        parent_size: int
    ) -> List[Dict[str, Any]]:
        """Creates small fine-grained child chunks for high precision retrieval linked to parent passage."""
        child_size = max(60, parent_size // 3)
        child_chunks_raw = self._fixed_size_chunking(text, metadata, child_size, chunk_overlap=15)

        chunks = []
        for idx, chk in enumerate(child_chunks_raw):
            chunks.append({
                "chunk_id": f"{metadata['doc_id']}_parent_child_{idx}",
                "text": chk["text"], # Fine-grained chunk for vector embedding lookup
                "metadata": {
                    **metadata,
                    "strategy": "parent_child",
                    "child_idx": idx,
                    "is_child": True
                },
                "parent_text": text # Full parent context returned for LLM answer generation
            })

        return chunks

    def compare_strategies(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs and evaluates all 4 chunking strategies on the corpus for analytics and visual benchmarking."""
        strategies = ["fixed", "semantic", "metadata_aware", "parent_child"]
        results = {}

        for strat in strategies:
            st = time.perf_counter()
            chunks = self.chunk_documents(documents, strategy=strat)
            elapsed_ms = round((time.perf_counter() - st) * 1000, 3)

            lengths = [len(c["text"]) for c in chunks] if chunks else [0]
            avg_len = round(sum(lengths) / len(lengths), 1) if lengths else 0
            
            results[strat] = {
                "strategy_name": strat,
                "total_chunks": len(chunks),
                "avg_chunk_length": avg_len,
                "min_chunk_length": min(lengths),
                "max_chunk_length": max(lengths),
                "processing_time_ms": elapsed_ms,
                "sample_chunks": [
                    {"chunk_id": c["chunk_id"], "text_snippet": c["text"][:120] + "..."}
                    for c in chunks[:3]
                ]
            }

        return results
