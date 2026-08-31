#!/usr/bin/env bash
# Launch SGLang for DeepSeek-V4-Pro on ROCm (MI35x).
# Mirrors InferenceX benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh
# against the DSv4 `_prs` image (baked open PRs; see verify_baked_patches).
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
ISL="${ISL:-8192}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
# Must be >= max_new_tokens (320k) + input so 320k-token generations aren't
# capped by context length. Model supports up to 1M positions.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-327680}"
DP_ATTENTION="${DP_ATTENTION:-false}"
EP_SIZE="${EP_SIZE:-1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/chat_templates/deepseek_v4_thinking.jinja}"
SGLANG_IMAGE="${SGLANG_IMAGE:-rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs}"
RUN_MODE="${RUN_MODE:-host}"  # host | docker
VERIFY_BAKED_PATCHES="${VERIFY_BAKED_PATCHES:-true}"

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

# Five sglang PRs and one aiter PR that DSv4 wants are baked into the `_prs`
# image (built from the upstream PR diffs). Verify instead of trusting the tag:
# benchmarking unpatched sources would silently report the wrong configuration.
resolve_pkg_dir() { # python package name -> directory of the installed package
  # Drop the cwd entry from sys.path first: the image WORKDIR may hold a
  # `sglang` source tree that otherwise shadows the installed package as a
  # namespace package whose __file__ is None.
  python3 -c "
import sys
sys.path = [p for p in sys.path if p not in ('', '.')]
import importlib.util
spec = importlib.util.find_spec('$1')
print(next(iter(spec.submodule_search_locations)))
"
}

require_baked_patch() { # PR label, file, fixed-string pattern, expected count
  local got
  got=$(grep -cF "$3" "$2" 2>/dev/null || true)
  if [[ "${got}" == "$4" ]]; then
    return 0
  fi
  echo "FATAL: $2 matched '$3' ${got:-0} time(s), expected $4. This image is" \
    "missing $1; refusing to benchmark unpatched sources. Use" \
    "${SGLANG_IMAGE} (or set VERIFY_BAKED_PATCHES=false)." >&2
  exit 1
}

verify_baked_patches() {
  local sglang_root aiter_root
  sglang_root=$(resolve_pkg_dir sglang)
  aiter_root=$(resolve_pkg_dir aiter)

  require_baked_patch "sgl-project/sglang#32340" \
    "${sglang_root}/kernels/ops/moe/fused_moe_triton_kernels.py" \
    "BLOCK_K=triton.next_power_of_2(k)" 2
  require_baked_patch "sgl-project/sglang#32577" \
    "${sglang_root}/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py" \
    "try_aiter_fused_mhc_post_pre" 2
  require_baked_patch "sgl-project/sglang#33165" \
    "${sglang_root}/srt/layers/quantization/fp8_utils.py" \
    "bpreshuffle_fp8_scale_nocopy" 3
  require_baked_patch "sgl-project/sglang#33166" \
    "${sglang_root}/srt/models/deepseek_common/attention_forward_methods/forward_mla.py" \
    "bpreshuffle_fp8_scale_nocopy_tuple" 3
  # v0.5.16_prs still ships the pre-rename flag; newer images use
  # SGLANG_OPT_USE_AITER_BATCHED_GEMM. Accept either marker.
  local environ_py="${sglang_root}/srt/environ.py"
  local batched_got
  batched_got=$(grep -cE 'SGLANG_OPT_(USE_AITER_BATCHED_GEMM|WO_A_AITER_BATCHED_GEMM)' \
    "${environ_py}" 2>/dev/null || true)
  if [[ "${batched_got}" -lt 1 ]]; then
    echo "FATAL: ${environ_py} is missing sgl-project/sglang#33313" \
      "(SGLANG_OPT_*AITER_BATCHED_GEMM). Refusing to continue." >&2
    exit 1
  fi
  require_baked_patch "ROCm/aiter#4506" \
    "${aiter_root}/ops/triton/quant/fused_fp8_quant.py" \
    "out1_bs = out1_bs.transpose(0, 1)" 2
  echo "Verified sglang #32340/#32577/#33165/#33166/#33313 and aiter #4506 are baked in."
}

export_sglang_env() {
  export SGLANG_DEFAULT_THINKING=1
  export SGLANG_DSV4_REASONING_EFFORT=max
  export SGLANG_USE_ROCM700A=0
  export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  # sglang defaults SGLANG_USE_AITER to False; leaving it unset disables the
  # aiter MoE/quant path and strands the baked PRs above.
  export SGLANG_USE_AITER=1
  export AITER_BF16_FP8_MOE_BOUND=0
  # sglang#33313: export both names so v0.5.16_prs (WO_A_*) and newer
  # (_USE_AITER_*) images both pick up the opt-in.
  export SGLANG_OPT_WO_A_AITER_BATCHED_GEMM=1
  export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
}

launch_sglang_server() {
  local model_path="$1"
  local eval_context_args=()
  if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    eval_context_args=(--context-length "${EVAL_MAX_MODEL_LEN:-${MAX_MODEL_LEN}}")
  fi

  local parallel_args=(--tensor-parallel-size "${TP}")
  local chunked_prefill_size="${ISL}"
  local moe_fusion_args
  # Shared-experts fusion and the DP shared-expert optimisations are mutually
  # exclusive: fusion folds the shared expert into the MoE and strands
  # SGLANG_SHARED_EXPERT_TP1 / SGLANG_DP_SHARED_EXPERT_LOCAL. Enable fusion only
  # on the pure-TP (dp-attn:false) path; under DP attention disable it.
  if [[ "${DP_ATTENTION}" == "true" ]]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5
    chunked_prefill_size=$((ISL * TP))
    parallel_args+=(
      --dp "${TP}"
      --enable-dp-attention
      --enable-prefill-delayer
      --enable-two-batch-overlap
    )
    moe_fusion_args=(--disable-shared-experts-fusion)
  else
    moe_fusion_args=(--enforce-shared-experts-fusion)
  fi
  if [[ "${EP_SIZE:-1}" -gt 1 ]]; then
    parallel_args+=(--ep-size "${EP_SIZE}")
  fi

  local chat_template_args=()
  if [[ -n "${CHAT_TEMPLATE}" && -f "${CHAT_TEMPLATE}" ]]; then
    chat_template_args=(--chat-template "${CHAT_TEMPLATE}")
  elif [[ -n "${CHAT_TEMPLATE}" ]]; then
    echo "WARNING: CHAT_TEMPLATE=${CHAT_TEMPLATE} not found; launching without --chat-template" >&2
  fi

  local serve_cmd=(python3 -m sglang.launch_server)
  if command -v sglang >/dev/null 2>&1; then
    serve_cmd=(sglang serve)
  fi

  local model_override_args=()
  if [[ -n "${JSON_MODEL_OVERRIDE_ARGS:-}" ]]; then
    model_override_args=(--json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}")
  fi

  # SGLANG_PORT must stay unset: see PORT comment above.
  env -u SGLANG_PORT "${serve_cmd[@]}" \
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
    --chunked-prefill-size "${chunked_prefill_size}" \
    "${moe_fusion_args[@]}" \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    "${chat_template_args[@]}" \
    "${model_override_args[@]}" \
    --watchdog-timeout 1800 \
    "${eval_context_args[@]}"
}

if [[ ! -d "${MODEL}" && "${MODEL}" == *"/"* ]]; then
  echo "NOTE: local model path ${MODEL} not found; patching HF cache for ${MODEL_REPO}"
  patch_model_config "${MODEL_REPO}"
  MODEL="${MODEL_REPO}"
elif [[ -f "${MODEL}/config.json" ]]; then
  # HF cache snapshots are often root-owned/read-only. Prefer an in-place
  # config patch when writable; otherwise pass a runtime override.
  unset JSON_MODEL_OVERRIDE_ARGS || true
  set +e
  python3 - "${MODEL}" <<'PYEOF'
import json, sys
from pathlib import Path

path = Path(sys.argv[1]) / "config.json"
config = json.loads(path.read_text())
model_type = config.get("model_type")
if model_type != "deepseek_v4":
    print(f"No patch needed: model_type is {model_type!r}")
    raise SystemExit(0)
try:
    config["model_type"] = "deepseek_v3"
    path.write_text(json.dumps(config, indent=2) + "\n")
except OSError as exc:
    print(f"WARNING: cannot patch {path} ({exc}); using json-model-override-args")
    raise SystemExit(2)
print(f"Patched {path}: model_type deepseek_v4 -> deepseek_v3")
PYEOF
  patch_rc=$?
  set -e
  if [[ "${patch_rc}" -eq 2 ]]; then
    export JSON_MODEL_OVERRIDE_ARGS='{"model_type":"deepseek_v3"}'
  elif [[ "${patch_rc}" -ne 0 ]]; then
    exit "${patch_rc}"
  fi
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
  echo "Docker image: ${SGLANG_IMAGE}"
  DOCKER_RUN_ARGS=(--rm -it)
  if [[ -n "${DOCKER_NAME:-}" ]]; then
    docker rm -f "${DOCKER_NAME}" 2>/dev/null || true
    DOCKER_RUN_ARGS=(--name "${DOCKER_NAME}" -d)
  fi

  CHAT_TEMPLATE_DOCKER="/chat_templates/deepseek_v4_thinking.jinja"
  # Mount the model path and, when present, its HF cache root so relative
  # snapshot->blob symlinks resolve inside the container.
  MODEL_MOUNTS=(-v "${MODEL}:${MODEL}:ro")
  if [[ -n "${HF_CACHE_ROOT:-}" && -d "${HF_CACHE_ROOT}" ]]; then
    MODEL_MOUNTS+=(-v "${HF_CACHE_ROOT}:${HF_CACHE_ROOT}:ro")
  elif [[ "${MODEL}" == *"/snapshots/"* ]]; then
    # .../models--ORG--NAME/snapshots/HASH -> mount models--ORG--NAME
    _model_repo_dir="$(dirname "$(dirname "${MODEL}")")"
    if [[ -d "${_model_repo_dir}" ]]; then
      MODEL_MOUNTS+=(-v "${_model_repo_dir}:${_model_repo_dir}:ro")
    fi
  fi
  docker run "${DOCKER_RUN_ARGS[@]}" \
    "${STORAGE_OPTS[@]}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    -p "127.0.0.1:${PORT}:${PORT}" \
    "${MODEL_MOUNTS[@]}" \
    -v "${HF_HOME}:/root/.cache/huggingface" \
    -v "${LOG_DIR}:/workspace:rw" \
    -v "${SCRIPT_DIR}/chat_templates:/chat_templates:ro" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e MODEL="${MODEL}" \
    -e MODEL_REPO="${MODEL_REPO}" \
    -e HTTP_PORT="${PORT}" \
    -e TP="${TP}" \
    -e CONC="${CONC}" \
    -e ISL="${ISL}" \
    -e MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    -e DP_ATTENTION="${DP_ATTENTION}" \
    -e EP_SIZE="${EP_SIZE}" \
    -e CHAT_TEMPLATE="${CHAT_TEMPLATE_DOCKER}" \
    -e EVAL_ONLY="${EVAL_ONLY:-false}" \
    -e EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-}" \
    -e VERIFY_BAKED_PATCHES="${VERIFY_BAKED_PATCHES}" \
    -e JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS:-}" \
    -e SGLANG_IMAGE="${SGLANG_IMAGE}" \
    -e RUN_MODE=host \
    -e SERVER_LOG=/workspace/server.log \
    -e LOG_DIR=/workspace \
    -v "${SCRIPT_DIR}/start_sglang_server.sh:/start_sglang_server.sh:ro" \
    -v "${SCRIPT_DIR}/docker_common.sh:/docker_common.sh:ro" \
    "${SGLANG_IMAGE}" \
    bash -c 'source /docker_common.sh && bash /start_sglang_server.sh'
  exit 0
fi

if [[ "${VERIFY_BAKED_PATCHES}" == "true" ]]; then
  verify_baked_patches
fi

echo "Starting SGLang on port ${PORT} with model ${MODEL} (TP=${TP}, CONC=${CONC}, ISL=${ISL}, DP_ATTENTION=${DP_ATTENTION})"
echo "Server log: ${SERVER_LOG}"
launch_sglang_server "${MODEL}" 2>&1 | tee -a "${SERVER_LOG}"
