#!/usr/bin/env bash
# Launch DeepSeek-V4-Pro with vLLM on 8×B200 / 8×B300.
#
# Key flags vs. a standard vLLM launch:
#   --data-parallel-size 8        Expert parallelism across 8 GPUs (no TP)
#   --enable-expert-parallel      Required for MoE data-parallel dispatch
#   --kv-cache-dtype fp8          DeepSeek V4's hybrid KV cache (c4a/c128a)
#   --block-size 256              Matches the 256-native-token logical block size
#   --attention_config.use_fp4_indexer_cache=True  FP4 indexer for 2x KV savings
#   --tokenizer-mode deepseek_v4  Custom tokenizer for V4 chat template
#   --reasoning-parser deepseek_v4  Strips <think>…</think> into reasoning_content
#   --compilation-config …        FULL_AND_PIECEWISE cudagraph + all custom fusions
#
# Startup time note:
#   Model weight loading (64 shards) + TileLang kernel compilation takes ~22 min
#   on 8×B200. The default VLLM_ENGINE_READY_TIMEOUT_S=600 (10 min) is too short
#   and will crash the API server with a TimeoutError even though the workers are
#   fine. Always set VLLM_ENGINE_READY_TIMEOUT_S=3600 for this model.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the directory containing the DeepSeek-V4-Pro weights}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH=${MODEL_PATH} does not exist."
  echo "Set MODEL_PATH to the directory containing the DeepSeek-V4-Pro weights."
  exit 1
fi

echo "Launching DeepSeek-V4-Pro on port ${PORT} (model: ${MODEL_PATH})"
echo "Startup takes ~22 minutes for weight loading + TileLang kernel compilation."
echo ""

CONTAINER_ID=$(docker run -d \
  --gpus all \
  --shm-size 32g \
  --net host \
  --ipc host \
  -v "${MODEL_PATH}:/model" \
  -v "${HF_HOME:-${HOME}/.cache/huggingface}:/root/.cache/huggingface" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model /model \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --disable-log-stats \
  --disable-uvicorn-access-log \
  --port "${PORT}")

echo "Container started: ${CONTAINER_ID:0:12}"
echo ""
echo "Polling http://localhost:${PORT}/health ..."

TIMEOUT=2400
START=$(date +%s)
while true; do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    ELAPSED=$(( $(date +%s) - START ))
    echo "Server healthy after ${ELAPSED}s. Ready to benchmark."
    break
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [[ ${ELAPSED} -ge ${TIMEOUT} ]]; then
    echo "ERROR: server not healthy after ${TIMEOUT}s"
    docker logs "${CONTAINER_ID}" | tail -40
    exit 1
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_ID}" 2>/dev/null)" != "true" ]]; then
    echo "ERROR: container exited unexpectedly"
    docker logs "${CONTAINER_ID}" | tail -40
    exit 1
  fi
  echo "  Waiting... (${ELAPSED}s)"
  sleep 15
done

echo ""
echo "Container ID : ${CONTAINER_ID:0:12}"
echo "Stop with   : docker stop ${CONTAINER_ID:0:12}"
