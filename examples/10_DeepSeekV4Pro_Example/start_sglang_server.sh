#!/usr/bin/env bash
# Launch SGLang for DeepSeek-V4-Pro on ROCm (MI35x / DSv4 image).
# Mirrors the FP4-experts env block from SGLang amd/deepseek_v4 run_dsv4.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINTS_DIR="${ENDPOINTS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

MODEL="${MODEL:-${MODEL_PATH:-/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro}}"
MODEL_REPO="${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Pro}"
# HTTP listen port for our scripts/YAML. Do NOT export SGLANG_PORT to the SGLang
# process: upstream uses that env var for internal ZMQ ports (get_open_port), which
# collides with --port and breaks uvicorn bind.
PORT="${HTTP_PORT:-${SGLANG_PORT:-30000}}"
TP="${TP:-8}"
CONC="${CONC:-512}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
# Must be >= max_new_tokens (320k) + input so 320k-token generations aren't
# capped by context length. Model supports up to 1M positions.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
# DSv4 tokenizer has no chat_template in tokenizer_config.json; SGLang needs --chat-template
# for /v1/chat/completions. Default to the v3.2 tool template shipped in the DSv4 image.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja}"
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  CHAT_TEMPLATE=""
fi
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706}"
RUN_MODE="${RUN_MODE:-host}"  # host | docker

patch_model_config() {
  local model_ref="$1"
  python3 <<PYEOF
import json
from huggingface_hub import hf_hub_download

repo_id = "${model_ref}"
path = hf_hub_download(repo_id=repo_id, filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF
}

export_sglang_env() {
  # Triton/AITER fallback env block for the stock lmsysorg sglang-rocm image,
  # which does not ship deep_gemm/tilelang kernels. Routes the DSv4 attention
  # backend around deep_gemm (unified_kv_triton flashmla, triton fused compress,
  # torch paged MQA logits, aiter indexer instead of tilelang).
  export SGLANG_DEFAULT_THINKING=1
  export SGLANG_DSV4_REASONING_EFFORT=max
  export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
  export SGLANG_USE_AITER=1
  export SGLANG_USE_ROCM700A=0
  export SGLANG_DP_USE_GATHERV=1
  export SGLANG_OPT_USE_FUSED_COMPRESS=true
  export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  export SGLANG_OPT_FP8_WO_A_GEMM=false
  export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
  export SGLANG_OPT_USE_TOPK_V2=false
  export SGLANG_OPT_USE_AITER_INDEXER=true
  export SGLANG_OPT_USE_TILELANG_INDEXER=false
  export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
  export SGLANG_OPT_USE_TILELANG_MHC_POST=false
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
  export AITER_BF16_FP8_MOE_BOUND=0
  export SGLANG_EAGER_INPUT_NO_COPY=true
  export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
  export SGLANG_ROCM_USE_MULTI_STREAM=false
}

launch_sglang_server() {
  local model_path="$1"
  local eval_context_args=()
  if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    eval_context_args=(--context-length "${EVAL_MAX_MODEL_LEN:-${MAX_MODEL_LEN}}")
  fi

  local parallel_args=(--tensor-parallel-size "${TP}")
  if [[ "${DP_ATTENTION}" == "true" ]]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5
    parallel_args+=(--dp "${TP}" --enable-dp-attention --enable-prefill-delayer --enable-two-batch-overlap)
  fi
  if [[ "${EP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--ep-size "${EP_SIZE}")
  fi

  local chat_template_args=()
  if [[ -n "${CHAT_TEMPLATE}" ]]; then
    chat_template_args=(--chat-template "${CHAT_TEMPLATE}")
  fi

  # SGLANG_PORT must stay unset: see PORT comment above.
  env -u SGLANG_PORT python3 -m sglang.launch_server \
    --model-path "${model_path}" \
    --host=0.0.0.0 \
    --port "${PORT}" \
    "${parallel_args[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --cuda-graph-max-bs "${CONC}" \
    --max-running-requests "${CONC}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --swa-full-tokens-ratio 0.15 \
    --page-size 256 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length "${MAX_MODEL_LEN}" \
    --chunked-prefill-size 8192 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    "${chat_template_args[@]}" \
    --watchdog-timeout 1800 \
    "${eval_context_args[@]}"
}

if [[ ! -d "${MODEL}" && "${MODEL}" == *"/"* ]]; then
  echo "NOTE: local model path ${MODEL} not found; patching HF cache for ${MODEL_REPO}"
  patch_model_config "${MODEL_REPO}"
  MODEL="${MODEL_REPO}"
elif [[ -f "${MODEL}/config.json" ]]; then
  python3 <<PYEOF
import json
from pathlib import Path

path = Path("${MODEL}") / "config.json"
config = json.loads(path.read_text())
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
else:
    print(f"No patch needed: model_type is {config.get('model_type')!r}")
PYEOF
else
  patch_model_config "${MODEL_REPO}"
fi

export_sglang_env

ensure_docker_log_dir "sglang"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/server.log}"

if [[ "${RUN_MODE}" == "docker" && ! -f /.dockerenv ]]; then
  if [[ ! -d "${MODEL}" ]]; then
    echo "ERROR: RUN_MODE=docker requires a local model directory at MODEL=${MODEL}"
    exit 1
  fi
  # Writable layer budget for server logs under /workspace (opt-in --storage-opt).
  DOCKER_LOG_STORAGE_GB="${DOCKER_LOG_STORAGE_GB:-64}"
  export DOCKER_LOG_STORAGE_GB
  HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  # shellcheck disable=SC2207
  STORAGE_OPTS=($(docker_storage_args))
  echo "Docker log mount: ${LOG_DIR} -> /workspace"
  DOCKER_RUN_ARGS=(--rm -it)
  if [[ -n "${DOCKER_NAME:-}" ]]; then
    docker rm -f "${DOCKER_NAME}" 2>/dev/null || true
    DOCKER_RUN_ARGS=(--name "${DOCKER_NAME}" -d)
  fi
  docker run "${DOCKER_RUN_ARGS[@]}" \
    "${STORAGE_OPTS[@]}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    -p "127.0.0.1:${PORT}:${PORT}" \
    -v "${MODEL}:${MODEL}:ro" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${LOG_DIR}:/workspace:rw" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e MODEL="${MODEL}" \
    -e MODEL_REPO="${MODEL_REPO}" \
    -e HTTP_PORT="${PORT}" \
    -e TP="${TP}" \
    -e CONC="${CONC}" \
    -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    -e DP_ATTENTION="${DP_ATTENTION}" \
    -e EP_SIZE="${EP_SIZE}" \
    -e CHAT_TEMPLATE="${CHAT_TEMPLATE}" \
    -e EVAL_ONLY="${EVAL_ONLY:-false}" \
    -e EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-}" \
    -e RUN_MODE=host \
    -e SERVER_LOG=/workspace/server.log \
    -e LOG_DIR=/workspace \
    -v "${SCRIPT_DIR}/start_sglang_server.sh:/start_sglang_server.sh:ro" \
    -v "${SCRIPT_DIR}/docker_common.sh:/docker_common.sh:ro" \
    "${SGLANG_IMAGE}" \
    bash -c 'source /docker_common.sh && bash /start_sglang_server.sh'
  exit 0
fi

echo "Starting SGLang on port ${PORT} with model ${MODEL} (TP=${TP}, CONC=${CONC})"
echo "Server log: ${SERVER_LOG}"
launch_sglang_server "${MODEL}" 2>&1 | tee -a "${SERVER_LOG}"
