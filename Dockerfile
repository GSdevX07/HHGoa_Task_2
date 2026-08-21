# ==============================================================================
# HHGoa Task 2 — Production CPU-First Docker Deployment
# Target: ₹0 hosting, sub-200ms warm latency, zero per-request model downloads
# ==============================================================================

FROM python:3.11-slim as runtime

# Prevent Python from writing bytecode and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    RERANKER_ENABLED=false \
    ANSWER_MODE=extractive \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    PORT=8000 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy runtime requirements first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download and cache embedding model during image build phase
COPY scripts/preload_models.py /app/scripts/preload_models.py
RUN python /app/scripts/preload_models.py

# Copy application files, pre-built indexes, and frontend static assets
COPY backend/ /app/backend/
COPY data/ /app/data/
COPY indexes/ /app/indexes/
COPY frontend/ /app/frontend/
COPY scripts/ /app/scripts/

# Security: Create non-root system user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Production server entrypoint
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
