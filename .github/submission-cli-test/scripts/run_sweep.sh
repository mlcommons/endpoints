#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 MLCommons
# SPDX-License-Identifier: Apache-2.0
#
# Run the sample pareto sweep against the dataset replay server and dry-run every
# resulting run folder through `endpoints-submission-cli runs create`.
#
# This is the whole gate in one script so it can be run identically on a laptop
# and inside GitHub Actions — the workflow calls this rather than duplicating
# the steps in YAML.
#
#   .github/submission-cli-test/scripts/run_sweep.sh
#
# Environment:
#   RUNNER      command prefix for the endpoints CLI  (default: "uv run")
#   SUBMIT_CLI  the submission CLI entry point        (default: endpoints-submission-cli)
#   OUT_DIR     where run folders are written         (default: results/pareto_ci)

set -euo pipefail

C_MIN=1
C_MAX=256
POINTS=(1 4 7 16 32 128 256)

RUNNER="${RUNNER:-uv run}"
SUBMIT_CLI="${SUBMIT_CLI:-endpoints-submission-cli}"
OUT_DIR="${OUT_DIR:-results/pareto_ci}"
# The repo's committed smoke dataset — the same one the README's Quick Start
# uses. Its `text_input` / `ref_output` columns are exactly what the replay
# server serves back, so no dataset has to be generated for this sweep. Keep it in
# step with the `path:` in .github/submission-cli-test/points_config/point_c*.yaml.
DATASET="${PARETO_CI_DATASET:-tests/assets/datasets/dummy_1k.jsonl}"
PORT="${REPLAY_PORT:-8765}"

log() { printf '\n=== %s ===\n' "$*"; }

# --------------------------------------------------------------------------
# 1. Dataset replay server.
# --------------------------------------------------------------------------
# POINTS satisfies the rules §5.3 region coverage for the declared envelope
# C_min=1 / C_max=256. Boundaries derived from §5.5 (log2-space thirds, banker's
# rounding), fixed for this envelope:
#   Ultra Low 1-32 (fixed, §5.4) | Low 2-7 | Medium 8-41 | High 42-256
# Changing C_max or adding a point means re-deriving these by hand — nothing
# checks region coverage automatically. Each point's config is validated against
# this array by scripts/check_payload.py once the run folder exists.
log "Starting dataset replay server on port $PORT"
$RUNNER python -m inference_endpoint.testing.dataset_replay_server \
  --port "$PORT" --dataset "$DATASET" \
  --ttft-ms 40 --tpot-ms 6 --slots 64 > dataset_replay_server.log 2>&1 &
SERVER_PID=$!

# Always take the server down, including on a failed benchmark, so a local run
# does not leave a port bound behind it.
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Poll rather than sleeping a fixed span: startup is fast locally and slower on
# a cold CI runner.
for _ in $(seq 1 40); do
  if curl -sf -m 2 -o /dev/null \
      -H 'Content-Type: application/json' \
      -d '{"model":"sim-model","messages":[{"role":"user","content":"ping"}]}' \
      "http://127.0.0.1:$PORT/v1/chat/completions"; then
    break
  fi
  sleep 0.5
done
log "Dataset replay server ready"

# --------------------------------------------------------------------------
# 2. One benchmark per measurement point.
# --------------------------------------------------------------------------
for c in "${POINTS[@]}"; do
  log "Benchmarking concurrency $c"
  $RUNNER inference-endpoint benchmark from-config \
    -c ".github/submission-cli-test/points_config/point_c${c}.yaml" --mode both
done

# --------------------------------------------------------------------------
# 3. Author system_desc.json, then dry-run each run folder through the
#    submission CLI. `runs create --dry-run` prints the payload and exits
#    before touching the API, so no PRISM token is needed.
# --------------------------------------------------------------------------
# Payloads go beside the run folders, not inside them: a real `runs create`
# archives the whole run folder, and the dry-run's own output is not part of
# the submission.
payload_dir="${OUT_DIR}/payloads"
mkdir -p "$payload_dir"

for c in "${POINTS[@]}"; do
  run_dir="${OUT_DIR}/point_c${c}"
  payload="${payload_dir}/point_c${c}.json"

  log "Preparing run folder $run_dir"
  $RUNNER python .github/submission-cli-test/scripts/make_system_desc.py \
    --run-dir "$run_dir" --c-max "$C_MAX" --system-name "sim_ci"

  log "runs create --dry-run for concurrency $c"
  $SUBMIT_CLI runs create --path "$run_dir" --dry-run > "$payload"
  $RUNNER python .github/submission-cli-test/scripts/check_payload.py \
    --payload "$payload" --concurrency "$c"
done

log "Sweep complete: ${#POINTS[@]} points measured and dry-run validated"
