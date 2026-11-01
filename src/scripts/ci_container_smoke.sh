#!/usr/bin/env bash
# Smoke-test a built API image on CI (Docker on linux; no Docker Desktop required locally).
set -euo pipefail

IMAGE="${1:?usage: ci_container_smoke.sh <image>}"
HOST_PORT="${SMOKE_PORT:-18080}"
CONTAINER_PORT="${SMOKE_CONTAINER_PORT:-8080}"
CONTAINER_NAME="${SMOKE_CONTAINER_NAME:-fiqa-api-smoke}"
HEALTH_URL="http://127.0.0.1:${HOST_PORT}/health"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run -d --name "${CONTAINER_NAME}" -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e PORT="${CONTAINER_PORT}" \
  -e QDRANT_PATH=/data/qdrant \
  -v fiqa-smoke-qdrant:/data/qdrant \
  "${IMAGE}"

echo "Waiting for ${HEALTH_URL} (embedding model load may take several minutes)..."
for attempt in $(seq 1 90); do
  if ! docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q true; then
    echo "FAIL: container exited before /health was ready" >&2
    docker logs "${CONTAINER_NAME}" >&2 || true
    exit 1
  fi
  if curl -sf "${HEALTH_URL}" >/dev/null; then
    echo "OK /health"
    curl -sf "${HEALTH_URL}"
    echo
    ready_code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${HOST_PORT}/ready" || true)"
    echo "INFO /ready -> HTTP ${ready_code} (503 expected without mounted production index + Ollama)"
    exit 0
  fi
  if [ "$((attempt % 6))" -eq 0 ]; then
    echo "Still waiting (${attempt}/90)..."
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 15 || true
  fi
  sleep 5
done

echo "FAIL: /health did not become ready in time" >&2
docker logs "${CONTAINER_NAME}" >&2 || true
exit 1
