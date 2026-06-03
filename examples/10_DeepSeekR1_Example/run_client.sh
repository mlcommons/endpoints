#!/usr/bin/env bash
# run_client.sh - drive the DeepSeek-R1 accuracy benchmark from the LOGIN node
# against a trtllm-serve started by `SERVER_ONLY=1 ... launch_and_run.sh`.
#
# Why a separate client step: on heterogeneous clusters (x86 login + aarch64
# GB200 compute) the benchmark client can't run on the ARM compute node, but the
# login node can reach the compute node's HTTP port. So the server holds its
# allocation (SERVER_ONLY) and this script runs the client here.
#
# Usage:
#   bash examples/10_DeepSeekR1_Example/run_client.sh [HOST[:PORT]]
# If HOST is omitted, it is read from logs/dsr1_server_ready (written by the
# SERVER_ONLY job). Choose the dataset via BENCH_CONFIG (default: full 4388).
#   BENCH_CONFIG=.../offline_deepseek_r1_accuracy_subset.yaml bash run_client.sh
# Set RELEASE_SERVER=1 to stop the server job (touch the stop file) when done.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
# The deepseek_r1 scorer shells out to the accuracy subproject (set up once with
# `cd accuracy && uv sync && bash setup_eval.sh`).
export DEEPSEEK_EVAL_PROJECT_PATH="${SCRIPT_DIR}/accuracy"
# The benchmark config references ${MODEL_DIR} (tokenizer for tokens_per_sample);
# it is resolved when the YAML is loaded, so it must be set in this environment.
: "${MODEL_DIR:?Set MODEL_DIR to your DeepSeek-R1 FP4 checkpoint directory}"

HOSTPORT="${1:-$(cat "${REPO_ROOT}/logs/dsr1_server_ready" 2>/dev/null || true)}"
if [ -z "${HOSTPORT}" ]; then
    echo "ERROR: no server host. Pass HOST[:PORT] or start the SERVER_ONLY job first." >&2
    exit 1
fi
case "${HOSTPORT}" in
    *:*) ENDPOINT="http://${HOSTPORT}" ;;
    *)   ENDPOINT="http://${HOSTPORT}:8000" ;;
esac

BENCH_CONFIG="${BENCH_CONFIG:-${SCRIPT_DIR}/offline_deepseek_r1_accuracy.yaml}"

# from-config takes the endpoint from the YAML, so render a run copy with this
# server's endpoint substituted in.
RUN_YAML="$(mktemp /tmp/dsr1_run_XXXX.yaml)"
sed -E "s#(- \").*(:8000\")#\1${ENDPOINT%:8000}\2#; s#https?://[^\"]+:8000#${ENDPOINT}#g" \
    "${BENCH_CONFIG}" > "${RUN_YAML}"

echo "==> endpoint: ${ENDPOINT}"
echo "==> config:   ${BENCH_CONFIG}"
curl -sf -m 10 "${ENDPOINT}/health" >/dev/null && echo "==> server health OK" || {
    echo "ERROR: ${ENDPOINT}/health unreachable" >&2; exit 1; }

cd "${REPO_ROOT}"
uv run inference-endpoint benchmark from-config --config "${RUN_YAML}" --mode acc

REPORT_DIR="$(grep -E '^report_dir:' "${BENCH_CONFIG}" | awk '{print $2}' | tr -d '"')"
echo "==> Done. Aggregate score:"
python3 -c "import json,sys; d=json.load(open('${REPORT_DIR}/results.json')); print(json.dumps(d.get('accuracy_scores'), indent=2))" 2>/dev/null || true
echo "    Per-subset: ${REPORT_DIR}/deepseek_eval/*_results.json"
echo "    Golden FP32 exact_match=81.3582 (pass >= 80.5246); tokens_per_sample golden=3886.2274 (90-110%)."

if [ "${RELEASE_SERVER:-0}" = "1" ]; then
    touch "${REPO_ROOT}/logs/dsr1_server_stop"
    echo "==> Signalled the SERVER_ONLY job to stop."
fi
