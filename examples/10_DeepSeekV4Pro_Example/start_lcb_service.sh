#!/usr/bin/env bash
# Build and run the LiveCodeBench evaluation container (repo-standard workflow).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

IMAGE="${LCB_IMAGE:-lcb-service:latest}"
CONTAINER_NAME="${LCB_CONTAINER_NAME:-lcb-service}"
PORT="${LCB_PORT:-13835}"

cd "${ENDPOINTS_DIR}"
ensure_docker_log_dir "lcb"

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
    src/inference_endpoint/evaluation/livecodebench \
    2>&1 | tee "${LOG_DIR}/lcb_build.log"
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting ${CONTAINER_NAME} on port ${PORT}..."
echo "LCB logs: ${LOG_DIR} (container /workspace, ${DOCKER_LOG_STORAGE_GB}G storage-opt when supported)"
# shellcheck disable=SC2207
RUN_STORAGE_OPTS=($(docker_storage_args))
docker run -d \
  "${RUN_STORAGE_OPTS[@]}" \
  --name "${CONTAINER_NAME}" \
  --rm \
  -p "127.0.0.1:${PORT}:13835" \
  -v "${LOG_DIR}:/workspace:rw" \
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
