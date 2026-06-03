#!/usr/bin/env bash
# launch_and_run.sh - End-to-end MLPerf accuracy run for DeepSeek-R1 FP4 on a
# single GB200x4 node: launch trtllm-serve in a Pyxis/Enroot container, wait
# for health, probe, then run the offline accuracy benchmark + scorer.
#
# Two ways to run (see README.md for the full runbook):
#
#   (A) Single-arch cluster (client can run on the compute node):
#         sbatch examples/10_DeepSeekR1_Example/launch_and_run.sh
#       ...or inside an existing allocation:
#         salloc -N1 --exclusive -p <your-partition> -A <your-slurm-account> -t 5:00:00
#         bash examples/10_DeepSeekR1_Example/launch_and_run.sh
#
#   (B) Heterogeneous cluster - x86 login + aarch64 GB200 compute: the benchmark
#       client can't run on the ARM compute node,
#       so run the server here in SERVER_ONLY mode and drive the client from the
#       login node with run_client.sh:
#         SERVER_ONLY=1 sbatch examples/10_DeepSeekR1_Example/launch_and_run.sh
#         # wait for logs/dsr1_server_ready, then on the login node:
#         bash examples/10_DeepSeekR1_Example/run_client.sh
#
# Overridable via environment variables (defaults below).

# A GB200x4 node is 2 Grace + 4 Blackwell (144 CPUs). If GPUs are not exposed as
# a SLURM gres on your cluster, allocate the whole node exclusively - all 4 GPUs
# come with it. Set the account/partition for your cluster (or override on the
# sbatch CLI with -A/-p).
#SBATCH --job-name=dsr1-fp4-acc
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=05:00:00
#SBATCH --output=dsr1-fp4-acc-%j.out

set -euo pipefail

# enroot/pyxis import the image into $TMPDIR on the compute node. A stray
# TMPDIR pointing at a login-node-only path makes the import fail with
# "mktemp: failed to create directory". Force a path that exists on the node.
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"

# Ensure a user-local uv (the benchmark client runs via `uv run`) is on PATH.
export PATH="${HOME}/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Under sbatch the script is copied to /var/spool/slurm, so BASH_SOURCE is not
# the repo path - anchor on $SLURM_SUBMIT_DIR (the dir sbatch was invoked from,
# i.e. the repo root). Fall back to BASH_SOURCE for interactive `bash` runs.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
SCRIPT_DIR="${REPO_ROOT}/examples/10_DeepSeekR1_Example"

# NOTE on checkpoint choice: the `deepseek_r1-torch-fp4-v2` (FP4-WO) checkpoint
# is NOT loadable by stock trtllm-serve (1.2.0rc6 / 1.3.0rc14) - its
# hf_quant_config excludes the *post-fusion* name `self_attn.fused_a` while the
# loader checks the *pre-fusion* q_a_proj/kv_a_proj names, so it allocates an
# FP4-packed buffer (3584) for the BF16 weight (7168) and aborts. The sibling
# `deepseek_r1-torch-fp4` excludes the individual names and loads cleanly, so it
# is the default here. Point MODEL_DIR at the -v2 checkpoint only with a
# trtllm build that handles it (the model card's reference is TRT-LLM main on 8xB200).
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your DeepSeek-R1 FP4 (ModelOpt) checkpoint directory}"
# trtllm-serve registers the model under the checkpoint dir's basename; the
# /v1/completions `model` field (model_params.name in the YAML) must match it.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_DIR}")}"
SOURCE_PKL="${SOURCE_PKL:?Set SOURCE_PKL to the MLPerf DeepSeek-R1 source dataset .pkl}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc14}"
SERVE_CONFIG="${SERVE_CONFIG:-${SCRIPT_DIR}/trtllm_serve_config.yaml}"
BENCH_CONFIG="${BENCH_CONFIG:-${SCRIPT_DIR}/offline_deepseek_r1_accuracy.yaml}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"                   # tensor-parallel ranks = total GPUs
NNODES="${NNODES:-${SLURM_NNODES:-1}}"    # nodes in the allocation (set --nodes on sbatch)
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"       # GB200x4
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"  # seconds to wait for the server to come up

# Mount the model, the dataset source, and the repo into the container.
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${MODEL_DIR}:${MODEL_DIR},$(dirname "${SOURCE_PKL}"):$(dirname "${SOURCE_PKL}"),${REPO_ROOT}:${REPO_ROOT}}"

# The benchmark client runs on this same node, so it reaches the server over
# localhost. The YAML config's endpoint must match (http://localhost:${PORT}).
ENDPOINT="http://localhost:${PORT}"

echo "=========================================="
echo " DeepSeek-R1 FP4 accuracy run"
echo "   model:     ${MODEL_DIR}"
echo "   image:     ${CONTAINER_IMAGE}"
echo "   endpoint:  ${ENDPOINT}"
echo "   repo:      ${REPO_ROOT}"
echo "=========================================="

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 0. Prepare the benchmark dataset (pkl -> parquet) if missing.
# ---------------------------------------------------------------------------
PARQUET="${SCRIPT_DIR}/data/deepseek_r1_eval.parquet"
if [ ! -f "${PARQUET}" ]; then
    echo "==> Preparing dataset parquet"
    python "${SCRIPT_DIR}/prepare_dataset.py" --source "${SOURCE_PKL}" --output "${PARQUET}"
fi

# ---------------------------------------------------------------------------
# 1. Launch trtllm-serve in the container (background srun step).
# ---------------------------------------------------------------------------
echo "==> Launching trtllm-serve (TP=${TP_SIZE} over ${NNODES} node(s) via trtllm-llmapi-launch)"
# TP>1 needs the TP ranks pre-launched: srun --ntasks=TP --ntasks-per-node=GPUS
# with --mpi=pmix, wrapped by trtllm-llmapi-launch (rank 0 serves, the rest are
# workers). Launching trtllm-serve directly instead fails with MPI_ERR_SPAWN -
# dynamic MPI process spawn does not work under srun/PMIx. For multi-node TP
# (e.g. TP=8 over 2 nodes), rank 0 (and the HTTP server) lands on the first
# allocated node, which is this batch node - so the localhost health check holds.
srun --overlap --nodes="${NNODES}" --ntasks="${TP_SIZE}" --ntasks-per-node="${GPUS_PER_NODE}" --mpi=pmix \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --no-container-remap-root \
    bash -lc "
        export TRTLLM_DISABLE_NIXL=1
        export UCX_RNDV_SCHEME=get_zcopy
        trtllm-llmapi-launch trtllm-serve '${MODEL_DIR}' \
            --host 0.0.0.0 --port ${PORT} \
            --extra_llm_api_options '${SERVE_CONFIG}'
    " &
SERVER_STEP_PID=$!

cleanup() {
    echo "==> Shutting down server step"
    kill "${SERVER_STEP_PID}" 2>/dev/null || true
    wait "${SERVER_STEP_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Wait for /health.
# ---------------------------------------------------------------------------
echo "==> Waiting for ${ENDPOINT}/health (up to ${HEALTH_TIMEOUT}s)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
until curl -sf "${ENDPOINT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_STEP_PID}" 2>/dev/null; then
        echo "ERROR: server step exited before becoming healthy" >&2
        exit 1
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        echo "ERROR: server did not become healthy within ${HEALTH_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 10
done
echo "==> Server healthy"

# ---------------------------------------------------------------------------
# SERVER_ONLY mode: hold the allocation with the server up and let an external
# client drive the benchmark. Needed on heterogeneous clusters (e.g. x86 login
# + aarch64 GB200 compute) where the benchmark client can't run on the compute
# node. Writes "<host>:<port>" to $READY_FILE, then blocks until $STOP_FILE
# appears or the server dies.
# ---------------------------------------------------------------------------
if [ "${SERVER_ONLY:-0}" = "1" ]; then
    READY_FILE="${READY_FILE:-${REPO_ROOT}/logs/dsr1_server_ready}"
    STOP_FILE="${STOP_FILE:-${REPO_ROOT}/logs/dsr1_server_stop}"
    mkdir -p "$(dirname "${READY_FILE}")"
    rm -f "${STOP_FILE}"
    echo "$(hostname):${PORT}" > "${READY_FILE}"
    echo "==> SERVER_ONLY: ready at $(hostname):${PORT}; waiting for ${STOP_FILE}"
    while [ ! -f "${STOP_FILE}" ]; do
        kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "server step died"; break; }
        sleep 10
    done
    echo "==> SERVER_ONLY: stop requested; shutting down."
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Probe, then run the accuracy benchmark (client runs in the repo uv env).
# ---------------------------------------------------------------------------
echo "==> Probing endpoint (liveness check via chat completions)"
uv run inference-endpoint probe \
    --endpoints "${ENDPOINT}" \
    --model "${SERVED_MODEL_NAME}" \
    --api-type openai \
    --requests 4

echo "==> Running accuracy benchmark"
# Endpoint comes from the YAML (http://localhost:${PORT}); from-config has no
# --endpoints override. Edit the YAML if you change PORT.
uv run inference-endpoint benchmark from-config \
    --config "${BENCH_CONFIG}" \
    --mode acc

echo "==> Done. Results under logs/deepseek_r1_fp4_accuracy/ (results.json)."
echo "    Golden FP32 exact_match=81.3582 (pass >= 80.5246);"
echo "    tokens_per_sample golden=3886.2274 (pass band 90-110%)."
