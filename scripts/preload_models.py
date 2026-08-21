"""
Pre-download Hugging Face Model Cache
=======================================
Run during Docker build phase to ensure embedding model weights are pre-baked
into the container image, eliminating model downloads during cold-start.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preload_models")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

def main():
    logger.info(f"Pre-downloading model '{MODEL_NAME}' into local cache...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        dim = model.get_sentence_embedding_dimension()
        logger.info(f"Successfully cached '{MODEL_NAME}' (dimension={dim}).")
    except Exception as exc:
        logger.error(f"Failed to pre-download model '{MODEL_NAME}': {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
