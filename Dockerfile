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
COPY src ./src

RUN pip install ".[api,embeddings]"

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /data/qdrant /data/hf_cache \
    && chown -R app:app /app /data/qdrant /data/hf_cache

USER app

# Mount host or named volumes at runtime; do not bake indexes into the image.
ENV QDRANT_PATH=/data/qdrant \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache \
    HUGGINGFACE_HUB_CACHE=/data/hf_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "src/scripts/serve_api.py", "--host", "0.0.0.0", "--port", "8000"]
