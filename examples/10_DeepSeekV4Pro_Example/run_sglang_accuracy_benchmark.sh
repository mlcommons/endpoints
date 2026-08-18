#!/usr/bin/env bash
# Pass@1 accuracy suite for DeepSeek-V4-Pro (AIME25 → GPQA → LiveCodeBench).
# Matches the validated recipe: target_concurrency=64, max_new_tokens=256000.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

SGLANG_PORT="${SGLANG_PORT:-30000}"
LCB_PORT="${LCB_PORT:-13835}"
TIMEOUT="${TIMEOUT:-1209600}"
export DOCKER_LOG_STORAGE_GB="${DOCKER_LOG_STORAGE_GB:-64}"
export ALLOW_LCB_LOCAL_EVAL="${ALLOW_LCB_LOCAL_EVAL:-true}"

AIME_CFG="${AIME_CFG:-${SCRIPT_DIR}/sglang_deepseek_v4_pro_aime_pass1.yaml}"
GPQA_CFG="${GPQA_CFG:-${SCRIPT_DIR}/sglang_deepseek_v4_pro_gpqa_pass1.yaml}"
LCB_CFG="${LCB_CFG:-${SCRIPT_DIR}/sglang_deepseek_v4_pro_lcb_pass1.yaml}"

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
  export TOKENIZER_MODEL_PATH="${MODEL_PATH:-/data/workloads-inference/hf_hub_cache/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1}"
fi

ensure_docker_log_dir "accuracy_pass1"
export LCB_DATASETS_DIR="${LCB_DATASETS_DIR:-${ENDPOINTS_DIR}/dataset_cache/livecodebench/release_v6}"

echo "=== Pre-flight checks ==="

SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:${SGLANG_PORT}}"
WAIT_FOR_SGLANG_S="${WAIT_FOR_SGLANG_S:-0}"

if ! wait_openai_compatible_server "${SGLANG_BASE_URL}" "${WAIT_FOR_SGLANG_S}"; then
  echo "ERROR: Inference server not reachable at ${SGLANG_BASE_URL}." >&2
  echo "  Start SGLang: ${SCRIPT_DIR}/start_sglang_server.sh" >&2
  echo "  If the server is slow to bind: export WAIT_FOR_SGLANG_S=120" >&2
  exit 1
fi

_allow_lcb_local=false
case "${ALLOW_LCB_LOCAL_EVAL:-}" in
  true | 1 | yes | TRUE | YES) _allow_lcb_local=true ;;
esac

if [[ "${_allow_lcb_local}" == "true" ]]; then
  export ALLOW_LCB_LOCAL_EVAL=true
  echo "ALLOW_LCB_LOCAL_EVAL=true — LiveCodeBench uses local subprocess scoring"
  if [[ ! -d "${LCB_DATASETS_DIR}/test_cases" ]]; then
    echo "LiveCodeBench test_cases missing — regenerating dataset cache..."
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
echo ""

run_one() {
  local label=$1
  local config=$2
  local logf="${LOG_DIR}/${label}_pass1.log"
  echo "=== Running ${label} pass@1 ==="
  echo "Config: ${config}"
  local cmd=(
    uv run inference-endpoint benchmark from-config
    -c "${config}"
    --timeout "${TIMEOUT}"
    --mode both
  )
  echo "${cmd[*]}"
  set +e
  "${cmd[@]}" 2>&1 | tee "${logf}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${label}_EXIT=${rc} (log: ${logf})"
  return "${rc}"
}

set +e
run_one AIME "${AIME_CFG}"
aime_rc=$?
run_one GPQA "${GPQA_CFG}"
gpqa_rc=$?
run_one LCB "${LCB_CFG}"
lcb_rc=$?
set -e

echo ""
echo "=== Suite summary ==="
echo "AIME_EXIT=${aime_rc} GPQA_EXIT=${gpqa_rc} LCB_EXIT=${lcb_rc}"
echo "Reports:"
echo "  results/sglang_deepseek_v4_pro_aime_pass1/accuracy/accuracy_results.json"
echo "  results/sglang_deepseek_v4_pro_gpqa_pass1/accuracy/accuracy_results.json"
echo "  results/sglang_deepseek_v4_pro_lcb_pass1/accuracy/accuracy_results.json"

exit $(( aime_rc != 0 || gpqa_rc != 0 || lcb_rc != 0 ? 1 : 0 ))
