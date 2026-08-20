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

def _extract_passage_msmarco_xi(raw: dict) -> str:
    """
    Extract best passage from MSMARCO-XI nested passages struct.
    Schema: passages = { English_passages: list[str], is_selected: list[int] }
    """
    passages_field = raw.get("passages", None)
    if isinstance(passages_field, dict):
        eng = passages_field.get("English_passages", []) or []
        sel = passages_field.get("is_selected", []) or []
        for i, s in enumerate(sel):
            if s == 1 and i < len(eng) and eng[i]:
                return normalize_text(eng[i])
        if eng and eng[0]:
            return normalize_text(eng[0])
    for field in ("passage", "passage_text", "passage_en"):
        v = raw.get(field, "")
        if v:
            return normalize_text(v)
    return ""


def download_language(lang_code: str, limit: int) -> list:
    """
    Download up to `limit` docs for one language using the HuggingFace
    Datasets Server REST API — avoids streaming 16 GB of parquet shards.

    API docs: https://huggingface.co/docs/dataset-viewer/en/rows
    Filters by source_lang to get only the target language rows.
    Falls back to single-shard streaming if the API is unavailable.
    """
    logger.info(f"Downloading lang={lang_code}, limit={limit} via Datasets Server API ...")
    items = []

    # ── Method 1: Datasets Server API (fast — no large downloads) ────────────
    try:
        import requests
        base_url = "https://datasets-server.huggingface.co/rows"
        offset = 0
        page_size = 100  # max per request

        while len(items) < limit:
            fetch = min(page_size, limit - len(items))
            params = {
                "dataset": "ai4bharat/MSMARCO-XI",
                "config": "default",
                "split": "validation",
                "offset": offset,
                "length": fetch,
                "filter": f'source_lang="{lang_code}"',
            }
            resp = requests.get(base_url, params=params, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("rows", [])
                if not rows:
                    break  # no more rows

                for row_wrapper in rows:
                    raw = row_wrapper.get("row", row_wrapper)
                    passage = _extract_passage_msmarco_xi(raw)
                    if not passage:
                        continue
                    query_native = normalize_text(raw.get("query", ""))
                    query_en     = normalize_text(raw.get("Eng_Query", ""))
                    if not query_native and not query_en:
                        continue
                    answer    = normalize_text(raw.get("Answer", "") or "")
                    answer_en = normalize_text(raw.get("Eng_Answer", "") or "")
                    answers   = [a for a in [answer, answer_en] if a]
                    items.append({
                        "id":        f"msmarco_xi_{lang_code}_{len(items):05d}",
                        "language":  lang_code,
                        "lang_name": LANG_NAMES.get(lang_code, lang_code.upper()),
                        "query":     query_native or query_en,
                        "query_en":  query_en,
                        "answers":   answers,
                        "passage":   passage,
                    })

                offset += len(rows)
                logger.info(f"  [{lang_code}] fetched {len(items)}/{limit} via API ...")

                if len(rows) < fetch:
                    break  # last page

            elif resp.status_code == 400:
                # Filter param may not be supported — fall through to streaming
                logger.warning(f"  [{lang_code}] API filter not supported (400), trying streaming ...")
                break
            else:
                logger.warning(f"  [{lang_code}] API returned {resp.status_code}, trying streaming ...")
                break

        if items:
            logger.info(f"  [{lang_code}] Downloaded {len(items)} docs via API.")
            return items

    except Exception as e:
        logger.warning(f"  [{lang_code}] API method failed: {e}. Falling back to single-shard streaming.")

    # ── Method 2: Stream only the first parquet shard (fallback) ────────────
    # Much faster than streaming the full split — shard 0000 is ~1.2 GB
    # but we stop as soon as we have `limit` matching rows.
    try:
        from datasets import load_dataset
        logger.info(f"  [{lang_code}] Streaming shard 0000.parquet ...")
        shard_url = (
            "hf://datasets/ai4bharat/MSMARCO-XI@refs/convert/parquet"
            "/default/validation/0000.parquet"
        )
        ds = load_dataset("parquet", data_files={"validation": shard_url},
                          split="validation", streaming=True)
        count = 0
        for raw in ds:
            row_lang = raw.get("source_lang", raw.get("language", ""))
            if row_lang and row_lang != lang_code:
                continue
            passage = _extract_passage_msmarco_xi(raw)
            if not passage:
                continue
            query_native = normalize_text(raw.get("query", ""))
            query_en     = normalize_text(raw.get("Eng_Query", ""))
            if not query_native and not query_en:
                continue
            answer    = normalize_text(raw.get("Answer", "") or "")
            answer_en = normalize_text(raw.get("Eng_Answer", "") or "")
            answers   = [a for a in [answer, answer_en] if a]
            items.append({
                "id":        f"msmarco_xi_{lang_code}_{count:05d}",
                "language":  lang_code,
                "lang_name": LANG_NAMES.get(lang_code, lang_code.upper()),
                "query":     query_native or query_en,
                "query_en":  query_en,
                "answers":   answers,
                "passage":   passage,
            })
            count += 1
            if count >= limit:
                break
        logger.info(f"  [{lang_code}] Downloaded {len(items)} docs via shard streaming.")

    except Exception as e:
        logger.warning(f"  [{lang_code}] Streaming fallback failed: {e}")
        logger.warning(f"  [{lang_code}] Skipping. Try: pip install -U datasets pyarrow requests")

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
