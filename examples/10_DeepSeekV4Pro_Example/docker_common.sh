# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared Docker helpers for DeepSeek-V4-Pro example scripts.

# Writable log directory on the host (mounted into containers at /workspace).
# Accuracy + server logs can grow large; default leaves headroom on the host.
DOCKER_LOG_STORAGE_GB="${DOCKER_LOG_STORAGE_GB:-16}"

ensure_docker_log_dir() {
  local subdir="${1:-misc}"
  LOG_DIR="${LOG_DIR:-${ENDPOINTS_DIR:-.}/results/docker_logs/${subdir}}"
  mkdir -p "${LOG_DIR}"
  export LOG_DIR
}

# Extra docker run args for a larger container writable layer (opt-in only).
# Logs are written to the mounted host LOG_DIR; most hosts use overlay2 without xfs
# pquota and reject --storage-opt.
docker_storage_args() {
  if [[ "${DOCKER_USE_LOG_STORAGE_OPT:-false}" == "true" ]]; then
    # shellcheck disable=SC2207
    echo --storage-opt "size=${DOCKER_LOG_STORAGE_GB}G"
  fi
}

# Wait for an OpenAI-compatible or SGLang HTTP server (example script preflight).
# Tries GET /health (SGLang native, vLLM), then GET /v1/models (OpenAI compatibility).
# Args: base_url [max_wait_seconds]
wait_openai_compatible_server() {
  local base="${1%/}"
  local max_wait="${2:-0}"
  local start
  start=$(date +%s)
  while true; do
    for path in /health /v1/models; do
      if curl --output /dev/null --silent --fail --max-time 5 "${base}${path}"; then
        echo "Inference server ready (${base}${path})"
        return 0
      fi
    done
    if (( max_wait <= 0 )); then
      return 1
    fi
    if (( $(date +%s) - start >= max_wait )); then
      return 1
    fi
    sleep 2
  done
}
