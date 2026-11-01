#!/usr/bin/env bash
# Smoke-test a built API image on CI (Docker on linux; no Docker Desktop required locally).
set -euo pipefail

IMAGE="${1:?usage: ci_container_smoke.sh <image>}"
PORT="${SMOKE_PORT:-18080}"
CONTAINER_NAME="${SMOKE_CONTAINER_NAME:-fiqa-api-smoke}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:8000" \
  -e QDRANT_PATH=/data/qdrant \
  -e HF_HOME=/tmp/hf_cache \
  -e TRANSFORMERS_CACHE=/tmp/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/tmp/hf_cache \
  -v fiqa-smoke-qdrant:/data/qdrant \
  "${IMAGE}"

echo "Waiting for ${HEALTH_URL} (embedding model load may take several minutes)..."
for _ in $(seq 1 90); do
  if curl -sf "${HEALTH_URL}" >/dev/null; then
    echo "OK /health"
    curl -sf "${HEALTH_URL}"
    echo
    ready_code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/ready" || true)"
    echo "INFO /ready -> HTTP ${ready_code} (503 expected without mounted production index + Ollama)"
    exit 0
  fi
  sleep 5
done

echo "FAIL: /health did not become ready in time" >&2
docker logs "${CONTAINER_NAME}" >&2 || true
exit 1
