#!/usr/bin/env bash
# Run DeepSeek-V4-Pro benchmark the same way as examples/04_GPTOSS120B_Example:
#   1. SGLang server on :30000
#   2. lcb-service container on :13835 (for LiveCodeBench scoring)
#   3. inference-endpoint benchmark from-config
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"
CONFIG="${CONFIG:-${ENDPOINTS_DIR}/examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_example.yaml}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
LCB_PORT="${LCB_PORT:-13835}"
TIMEOUT="${TIMEOUT:-60}"
MODE="${MODE:-}"  # empty = default (perf for online); set to 'both' for perf + accuracy collection

cd "${ENDPOINTS_DIR}"

if [[ -z "${TOKENIZER_MODEL_PATH:-}" ]]; then
  export TOKENIZER_MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro}"
fi

echo "=== Pre-flight checks ==="

SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:${SGLANG_PORT}}"
WAIT_FOR_SGLANG_S="${WAIT_FOR_SGLANG_S:-0}"

if ! wait_openai_compatible_server "${SGLANG_BASE_URL}" "${WAIT_FOR_SGLANG_S}"; then
  echo "ERROR: Inference server not reachable at ${SGLANG_BASE_URL} (tried GET /health and GET /v1/models)." >&2
  echo "  Start the SGLang server first (see examples/10_DeepSeekV4Pro_Example/README.md)." >&2
  echo "  Smoke test: uv run python -m inference_endpoint.testing.echo_server --host 127.0.0.1 --port ${SGLANG_PORT}" >&2
  echo "  If the server is slow to bind: export WAIT_FOR_SGLANG_S=120" >&2
  exit 1
fi

if ! curl --output /dev/null --silent --fail "http://127.0.0.1:${LCB_PORT}/info"; then
  echo "ERROR: lcb-service is not running on port ${LCB_PORT}."
  echo ""
  echo "Build and start it per:"
  echo "  src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container"
  echo ""
  echo "Quick start (after 'docker login dhi.io'):"
  echo "  docker build -f src/inference_endpoint/evaluation/livecodebench/lcb_serve.dockerfile \\"
  echo "    --secret id=HF_TOKEN,env=HF_TOKEN -t lcb-service \\"
  echo "    src/inference_endpoint/evaluation/livecodebench"
  echo "  docker run --name lcb-service --rm -p 127.0.0.1:${LCB_PORT}:13835 lcb-service:latest"
  exit 1
fi
echo "lcb-service OK on port ${LCB_PORT}"

echo ""
echo "=== Running benchmark ==="
CMD=(uv run inference-endpoint benchmark from-config -c "${CONFIG}" --timeout "${TIMEOUT}")
if [[ -n "${MODE}" ]]; then
  CMD+=(--mode "${MODE}")
fi
echo "${CMD[*]}"
"${CMD[@]}"
