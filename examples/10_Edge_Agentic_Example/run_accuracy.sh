#!/usr/bin/env bash
# Reproduce BFCL v4 edge-agentic accuracy reference results (~2.5 h on an edge device).
#
# Usage:
#   1. Edit MODEL and ENDPOINT below to match your server.
#   2. bash run_accuracy.sh
#
# Results are written to:
#   results/bfcl_v4_single_turn_accuracy/   (single-turn)
#   results/bfcl_v4_multi_turn/             (multi-turn)

set -euo pipefail

MODEL="${MODEL:-Qwen3.6-27B-Q4_K_M}"
ENDPOINT="${ENDPOINT:-http://localhost:8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== BFCL v4 edge-agentic accuracy run ==="
echo "  Model:    $MODEL"
echo "  Endpoint: $ENDPOINT"
echo ""

# Single-turn: non_live (20%), live (10%), hallucination (5%) — ~82 min
echo "--- Single-turn (~82 min) ---"
inference-endpoint benchmark from-config \
    --config offline_bfcl_v4_single_turn.yaml \
    --accuracy-only \
    --model-params.name "$MODEL" \
    --endpoint-config.endpoints "$ENDPOINT"

# Multi-turn: 3% sample across all four subsets — ~64 min
echo "--- Multi-turn (~64 min) ---"
python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --sample-pct 3 \
    --temperature 0 \
    --seed 42 \
    --max-steps-per-turn 25 \
    --report-dir results/bfcl_v4_multi_turn/

echo ""
echo "=== Done. Results in results/ ==="
