#!/usr/bin/env bash
# Run DeepSeek-V4-Pro benchmark against SGLang:
#   1. SGLang server on :30000 (see start_sglang_server.sh)
#   2. lcb-service container on :13835 (for LiveCodeBench scoring)
#   3. inference-endpoint benchmark from-config
set -euo pipefail

ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG="${CONFIG:-${ENDPOINTS_DIR}/examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_example.yaml}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
LCB_PORT="${LCB_PORT:-13835}"
TIMEOUT="${TIMEOUT:-60}"
MODE="${MODE:-}"

cd "${ENDPOINTS_DIR}"

if [[ -z "${TOKENIZER_MODEL_PATH:-}" ]]; then
  export TOKENIZER_MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro}"
fi

echo "=== Pre-flight checks ==="

if ! curl --output /dev/null --silent --fail "http://127.0.0.1:${SGLANG_PORT}/health"; then
  echo "ERROR: SGLang is not healthy on port ${SGLANG_PORT}."
  echo "Start the server first (see README.md — SGLang section)."
  exit 1
fi
echo "SGLang OK on port ${SGLANG_PORT}"

if ! curl --output /dev/null --silent --fail "http://127.0.0.1:${LCB_PORT}/info"; then
  echo "ERROR: lcb-service is not running on port ${LCB_PORT}."
  echo ""
  echo "Quick start (after 'docker login dhi.io'):"
  echo "  ${ENDPOINTS_DIR}/examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh"
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
