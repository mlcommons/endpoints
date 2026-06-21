#!/usr/bin/env bash
# Monitor the running SGLang DeepSeek-V4-Pro accuracy benchmark.
# Emits sentinel lines (ALERT_HANG / RUN_FAILED / RUN_FINISHED_OK) so an external
# watcher can be notified, and logs periodic status to a monitor log.
set -uo pipefail

PROC_PAT="${PROC_PAT:-from-config.*sglang_deepseek_v4_pro_accuracy}"
WRAPPER_LOG="${WRAPPER_LOG:?set WRAPPER_LOG to the nohup wrapper log path}"
REPORT_DIR="${REPORT_DIR:-results/sglang_deepseek_v4_pro_accuracy}"
INTERVAL_S="${INTERVAL_S:-300}"
# Consecutive idle (no new completes + GPU idle) checks before declaring a hang.
HANG_STRIKES="${HANG_STRIKES:-3}"
GPU_IDLE_PCT="${GPU_IDLE_PCT:-5}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

live_events_file() {
  # Derive the event logger's tmpfs events.jsonl from its --log-dir arg.
  local logdir
  logdir=$(pgrep -af "services.event_logger" \
    | grep -oE -- "--log-dir [^ ]+" | head -1 | awk '{print $2}')
  [[ -n "${logdir}" ]] && echo "${logdir}/events.jsonl"
}

gpu_busy_pct() {
  rocm-smi --showuse 2>/dev/null \
    | grep -oE "GPU use \(%\): [0-9]+" | grep -oE "[0-9]+$" \
    | sort -rn | head -1
}

completed_count() {
  local f="$1" c
  if [[ -f "${f}" ]]; then
    # grep -c always prints a count (0 when no match); capture it so a non-zero
    # exit status on 0 matches does not also trigger a fallback echo.
    c=$(grep -c '"event_type":"sample.complete"' "${f}" 2>/dev/null)
    echo "${c:-0}"
  else
    echo 0
  fi
}

prev_completed=-1
strikes=0

echo "[$(ts)] monitor start: pat='${PROC_PAT}' interval=${INTERVAL_S}s wrapper=${WRAPPER_LOG}"

while true; do
  if ! pgrep -f "${PROC_PAT}" >/dev/null 2>&1; then
    # Process gone — classify success vs failure from the wrapper log.
    if grep -q "Score for livecodebench" "${WRAPPER_LOG}" 2>/dev/null \
       || grep -q "Saved: ${REPORT_DIR}/results.json" "${WRAPPER_LOG}" 2>/dev/null; then
      echo "[$(ts)] RUN_FINISHED_OK"
    else
      echo "[$(ts)] RUN_FAILED (process exited without final score)"
      echo "---- wrapper tail ----"
      tr '\r' '\n' < "${WRAPPER_LOG}" 2>/dev/null | tail -25
    fi
    exit 0
  fi

  ev=$(live_events_file)
  done_n=$(completed_count "${ev:-/nonexistent}")
  gpu=$(gpu_busy_pct); gpu="${gpu:-0}"
  echo "[$(ts)] alive completed=${done_n} gpu_busy=${gpu}% events=${ev:-none}"

  # Hang heuristic: no new completions AND GPUs idle for HANG_STRIKES intervals.
  if [[ "${done_n}" == "${prev_completed}" && "${gpu}" -lt "${GPU_IDLE_PCT}" ]]; then
    strikes=$((strikes + 1))
    echo "[$(ts)] no-progress strike ${strikes}/${HANG_STRIKES} (completed unchanged at ${done_n}, gpu ${gpu}%)"
    if [[ "${strikes}" -ge "${HANG_STRIKES}" ]]; then
      echo "[$(ts)] ALERT_HANG completed stuck at ${done_n}, gpu ${gpu}% for ${strikes} checks"
      echo "---- wrapper tail ----"
      tr '\r' '\n' < "${WRAPPER_LOG}" 2>/dev/null | tail -15
    fi
  else
    strikes=0
  fi
  prev_completed="${done_n}"
  sleep "${INTERVAL_S}"
done
