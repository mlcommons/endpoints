#!/usr/bin/env bash
# Accuracy-only DeepSeek-V4-Pro benchmark against vLLM (AIME25 + GPQA + LCB).
# Workflow mirrors run_sglang_accuracy_benchmark.sh + from-config accuracy YAML.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

CONFIG="${CONFIG:-${SCRIPT_DIR}/vllm_deepseek_v4_pro_accuracy.yaml}"
VLLM_PORT="${VLLM_PORT:-8000}"
LCB_PORT="${LCB_PORT:-13835}"
TIMEOUT="${TIMEOUT:-3600}"
USE_PYTHON_SCRIPT="${USE_PYTHON_SCRIPT:-false}"
# Accuracy phases drain with no timeout by default (settings.drain.accuracy_timeout_s).
export ALLOW_LCB_LOCAL_EVAL="${ALLOW_LCB_LOCAL_EVAL:-true}"

cd "${ENDPOINTS_DIR}"

if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME:-${HOME}/.cache/huggingface}/token" ]]; then
  HF_TOKEN="$(cat "${HF_HOME:-${HOME}/.cache/huggingface}/token")"
  export HF_TOKEN
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required (GPQA is a gated HuggingFace dataset)."
  echo "  export HF_TOKEN=<your HuggingFace token>"
  exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

if [[ -z "${TOKENIZER_MODEL_PATH:-}" ]]; then
  export TOKENIZER_MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro}"
fi

ensure_docker_log_dir "accuracy"

export LCB_DATASETS_DIR="${LCB_DATASETS_DIR:-${ENDPOINTS_DIR}/dataset_cache/livecodebench/release_v6}"

echo "=== Pre-flight checks ==="

if ! curl --output /dev/null --silent --fail "http://127.0.0.1:${VLLM_PORT}/health"; then
  echo "ERROR: vLLM is not healthy on port ${VLLM_PORT}."
  echo "Start the server first (see README.md)."
  exit 1
fi
echo "vLLM OK on port ${VLLM_PORT}"

_allow_lcb_local=false
case "${ALLOW_LCB_LOCAL_EVAL:-}" in
  true | 1 | yes | TRUE | YES) _allow_lcb_local=true ;;
esac

if [[ "${_allow_lcb_local}" == "true" ]]; then
  export ALLOW_LCB_LOCAL_EVAL=true
  echo "ALLOW_LCB_LOCAL_EVAL=true — LiveCodeBench will use subprocess scoring (no lcb-service)"
  if [[ ! -d "${LCB_DATASETS_DIR}/test_cases" ]]; then
    echo "LiveCodeBench test_cases missing — regenerating dataset cache (required for local scoring)..."
    uv run python -c "
from pathlib import Path
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench

LiveCodeBench.generate(
    Path('${ENDPOINTS_DIR}/dataset_cache'),
    variant='release_v6',
    force=True,
    save_test_cases=True,
)
"
  fi
elif ! curl --output /dev/null --silent --fail "http://127.0.0.1:${LCB_PORT}/info"; then
  echo "ERROR: lcb-service is not running on port ${LCB_PORT}."
  echo "Either start it: ${SCRIPT_DIR}/start_lcb_service.sh (requires 'docker login dhi.io')"
  echo "Or run without the container: export ALLOW_LCB_LOCAL_EVAL=true"
  exit 1
else
  echo "lcb-service OK on port ${LCB_PORT}"
fi

echo "Log directory (host): ${LOG_DIR}"
echo "Accuracy phase drain: unlimited (settings.drain.accuracy_timeout_s default)"
echo ""

rm -rf "${ENDPOINTS_DIR}/results/vllm_deepseek_v4_pro_accuracy"
echo "Cleared prior results: results/vllm_deepseek_v4_pro_accuracy/"
echo ""

if [[ "${USE_PYTHON_SCRIPT}" == "true" ]]; then
  echo "=== Running accuracy suite (Python script) ==="
  BENCHMARK_LOG="${LOG_DIR}/accuracy_benchmark.log"
  uv run python "${SCRIPT_DIR}/run_accuracy.py" \
    --endpoint-url "http://127.0.0.1:${VLLM_PORT}" \
    --report-dir results/vllm_deepseek_v4_pro_accuracy \
    --max-duration "${TIMEOUT}" \
    2>&1 | tee "${BENCHMARK_LOG}"
else
  echo "=== Running accuracy benchmark (from-config) ==="
  BENCHMARK_LOG="${LOG_DIR}/accuracy_from_config.log"
  CMD=(uv run inference-endpoint benchmark from-config -c "${CONFIG}" --timeout "${TIMEOUT}")
  echo "${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "${BENCHMARK_LOG}"
fi

echo ""
echo "Benchmark log: ${BENCHMARK_LOG}"
