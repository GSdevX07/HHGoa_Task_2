"""
Download and Preprocess AI4Bharat MSMARCO-XI Dataset
=====================================================
Run this ONCE before starting the server.

Usage:
    python scripts/download_dataset.py [--langs hi,en,ta,te,bn] [--limit 500]

Output:
    data/corpus.jsonl   — one JSON document per line
"""

import os
import sys
import json
import hashlib
import argparse
import logging
import re
import unicodedata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_dataset")

# Resolve paths relative to repo root regardless of where script is run from
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

LANG_NAMES = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ml": "Malayalam",
    "kn": "Kannada",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
}

# ── Text Normalization ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize Unicode, collapse whitespace, strip HTML artifacts."""
    if not text:
        return ""
    # Normalize Unicode to NFC (composed form — important for Indic scripts)
    text = unicodedata.normalize("NFC", text)
    # Strip simple HTML tags if any
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple whitespace / newlines
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def passage_hash(text: str) -> str:
    """SHA-256 hash of normalized passage for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_passage_text(item: dict) -> str:
    """Extract passage text from various MSMARCO-XI field layouts."""
    # Field layout varies slightly between HuggingFace splits
    passage = item.get("passage", "") or item.get("passage_text", "")
    if not passage:
        passages_field = item.get("passages", "")
        if isinstance(passages_field, str):
            passage = passages_field
        elif isinstance(passages_field, list) and passages_field:
            first = passages_field[0]
            if isinstance(first, dict):
                passage = first.get("passage_text", "") or first.get("text", "")
            elif isinstance(first, str):
                passage = first
    return passage


def extract_answers(item: dict) -> list:
    """Extract answer list, handling multiple field layouts."""
    answers = item.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    elif isinstance(answers, dict):
        answers = list(answers.values())
    return [str(a) for a in answers if a]


# ── Main Download Logic ────────────────────────────────────────────────────────

def download_language(lang_code: str, limit: int) -> list:
    """Download up to `limit` documents for one language split."""
    logger.info(f"Downloading lang={lang_code}, limit={limit} ...")
    items = []

    try:
        from datasets import load_dataset
        # Streaming avoids downloading the entire split before iterating
        # Use 'default' config and filter by language, and remove deprecated trust_remote_code
        ds = load_dataset(
            "ai4bharat/MSMARCO-XI",
            "default",
            split="train",
            streaming=True,
        )
        count = 0
        for raw in ds:
            # Check if this row matches the target language
            row_lang = raw.get("language", "")
            if row_lang and row_lang != lang_code:
                continue

            passage = normalize_text(extract_passage_text(raw))
            query = normalize_text(raw.get("query", ""))
            if not passage or not query:
                continue

            items.append({
                "id": f"msmarco_hf_{lang_code}_{count:05d}",
                "language": lang_code,
                "lang_name": LANG_NAMES.get(lang_code, lang_code.upper()),
                "query": query,
                "query_en": normalize_text(raw.get("query_en", "")),
                "answers": extract_answers(raw),
                "passage": passage,
                "passage_en": normalize_text(raw.get("passage_en", "")),
            })
            count += 1
            if count >= limit:
                break

        logger.info(f"  [{lang_code}] Downloaded {len(items)} documents.")
    except Exception as e:
        logger.warning(f"  [{lang_code}] HuggingFace download failed: {e}")
        logger.warning(f"  [{lang_code}] This language will be skipped. "
                       "Ensure HuggingFace datasets library is installed and network is available.")

    return items


def deduplicate(docs: list) -> list:
    """Remove documents with duplicate passage text."""
    seen_hashes = set()
    unique = []
    for doc in docs:
        h = passage_hash(doc["passage"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(doc)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess MSMARCO-XI corpus.")
    parser.add_argument(
        "--langs",
        default="hi,en,ta,te,bn",
        help="Comma-separated language codes (default: hi,en,ta,te,bn)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max documents per language (default: 500)"
    )
    args = parser.parse_args()

    lang_codes = [l.strip() for l in args.langs.split(",") if l.strip()]
    os.makedirs(DATA_DIR, exist_ok=True)

    all_docs = []
    for lang in lang_codes:
        docs = download_language(lang, args.limit)
        all_docs.extend(docs)

    before_dedup = len(all_docs)
    all_docs = deduplicate(all_docs)
    logger.info(
        f"Total: {before_dedup} docs → {len(all_docs)} after deduplication "
        f"({before_dedup - len(all_docs)} duplicates removed)."
    )

    if not all_docs:
        logger.error(
            "No documents downloaded. Check your internet connection, "
            "HuggingFace access, and that 'datasets' is installed: pip install datasets"
        )
        sys.exit(1)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_docs)} documents to {CORPUS_PATH}")
    logger.info("Next step: run  python scripts/build_index.py")


if __name__ == "__main__":
    main()
