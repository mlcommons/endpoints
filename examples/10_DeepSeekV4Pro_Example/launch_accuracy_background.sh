#!/usr/bin/env bash
# Start run_sglang_accuracy_benchmark.sh under nohup with a stable log path.
# (Avoids `VAR=... && nohup ... &` where bash backgrounds the whole `&&` chain so
# follow-up commands in the parent shell lose VAR.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=docker_common.sh
source "${SCRIPT_DIR}/docker_common.sh"

ensure_docker_log_dir "accuracy"
LOGF="${LOG_DIR}/nohup_accuracy_$(date +%Y%m%d_%H%M%S).log"
export WAIT_FOR_SGLANG_S="${WAIT_FOR_SGLANG_S:-120}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

nohup "${SCRIPT_DIR}/run_sglang_accuracy_benchmark.sh" >"${LOGF}" 2>&1 &
echo "Started accuracy benchmark PID=$!"
echo "Wrapper log: ${LOGF}"
echo "Tee log (inside run): ${LOG_DIR}/accuracy_from_config.log"
