#!/usr/bin/env bash
# Run DeepSeek-V4-Pro benchmark the same way as examples/04_GPTOSS120B_Example:
#   1. vLLM server on :8000
#   2. lcb-service container on :13835 (for LiveCodeBench scoring)
#   3. inference-endpoint benchmark from-config
set -euo pipefail

ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONFIG="${CONFIG:-${ENDPOINTS_DIR}/examples/10_DeepSeekV4Pro_Example/vllm_deepseek_v4_pro_example.yaml}"
VLLM_PORT="${VLLM_PORT:-8000}"
LCB_PORT="${LCB_PORT:-13835}"
TIMEOUT="${TIMEOUT:-60}"
MODE="${MODE:-}"  # empty = default (perf for online); set to 'both' for perf + accuracy collection

cd "${ENDPOINTS_DIR}"

if [[ -z "${TOKENIZER_MODEL_PATH:-}" ]]; then
  export TOKENIZER_MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro}"
fi

echo "=== Pre-flight checks ==="

if ! curl --output /dev/null --silent --fail "http://127.0.0.1:${VLLM_PORT}/health"; then
  echo "ERROR: vLLM is not healthy on port ${VLLM_PORT}."
  echo "Start the server first (see README.md)."
  exit 1
fi
echo "vLLM OK on port ${VLLM_PORT}"

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
