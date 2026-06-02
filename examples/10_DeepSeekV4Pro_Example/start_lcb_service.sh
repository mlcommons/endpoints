#!/usr/bin/env bash
# Build and run the LiveCodeBench evaluation container (repo-standard workflow).
set -euo pipefail

ENDPOINTS_DIR="${ENDPOINTS_DIR:-/home/karverma/endpoints}"
IMAGE="${LCB_IMAGE:-lcb-service:latest}"
CONTAINER_NAME="${LCB_CONTAINER_NAME:-lcb-service}"
PORT="${LCB_PORT:-13835}"

cd "${ENDPOINTS_DIR}"

if curl --output /dev/null --silent --fail "http://127.0.0.1:${PORT}/info" 2>/dev/null; then
  echo "lcb-service already responding on port ${PORT}"
  exit 0
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required to build lcb-service (dataset download at build time)."
    exit 1
  fi
  echo "Building ${IMAGE} (requires 'docker login dhi.io')..."
  docker build \
    -f src/inference_endpoint/evaluation/livecodebench/lcb_serve.dockerfile \
    --secret id=HF_TOKEN,env=HF_TOKEN \
    -t "${IMAGE%%:*}" \
    src/inference_endpoint/evaluation/livecodebench
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting ${CONTAINER_NAME} on port ${PORT}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --rm \
  -p "127.0.0.1:${PORT}:13835" \
  "${IMAGE}"

for _ in $(seq 1 120); do
  if curl --output /dev/null --silent --fail "http://127.0.0.1:${PORT}/info"; then
    echo "lcb-service is ready at http://127.0.0.1:${PORT}/info"
    exit 0
  fi
  sleep 5
done

echo "ERROR: lcb-service did not become ready. Check: docker logs ${CONTAINER_NAME}"
exit 1
