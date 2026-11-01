# FiQA production RAG API — runtime image (Qdrant index + Ollama are external/mounted).
# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenMP runtime used by PyTorch / sentence-transformers wheels on Debian slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY deploy ./deploy
COPY src ./src

RUN pip install ".[api,embeddings]"

# Prefetch frozen production embedder (BGE-base) into the image layer.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN mkdir -p /app/.cache/huggingface \
    && PYTHONPATH=/app/src python /app/src/scripts/prefetch_embedding_model.py

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /data/qdrant \
    && chown -R app:app /app /data/qdrant

USER app

# Qdrant index is external (local mount or Qdrant Cloud). Embedding weights are baked at build time.
ENV QDRANT_PATH=/data/qdrant \
    BM25_INDEX_DIR=deploy/index_metadata \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health', timeout=3)"

CMD ["python", "src/scripts/serve_api.py", "--host", "0.0.0.0"]
