#!/usr/bin/env bash
# setup_and_test.sh — End-to-end runbook for the WAN 2.2 video-generation example.
#
# Steps:
#   1. Download the WAN 2.2 weights from HuggingFace.
#   2. Launch trtllm-serve in a separate shell.
#   3. Run the offline benchmark from this script.
#
# Prerequisites: Python 3.12, a GPU host with trtllm-serve installed,
# and HuggingFace credentials (`huggingface-cli login`) — the WAN 2.2
# weights are gated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_REPO="Wan-AI/Wan2.2-T2V-A14B"   # https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
MODEL_DIR="${MODEL_DIR:-${HOME}/models/wan2.2-t2v-a14b}"

cd "${REPO_ROOT}"

# 1. Download model weights (~28 GB).
huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"

# 2. Launch the server in a separate shell, then re-run this script:
#
#      trtllm-serve "${MODEL_DIR}" --host 0.0.0.0 --port 8000 \
#          --backend pytorch --task text_to_video
#
# 3. Run the offline benchmark.
inference-endpoint benchmark from-config \
    --config "${SCRIPT_DIR}/offline_wan22.yaml"
