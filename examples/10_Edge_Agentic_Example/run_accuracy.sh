#!/usr/bin/env bash
# Reproduce the BFCL v4 edge-agentic accuracy reference result (~3 h on an edge
# device). The finalized accuracy benchmark is single-turn only, sampled to ~995
# samples (see offline_bfcl_v4_single_turn.yaml).
#
# Usage:
#   1. Edit MODEL and ENDPOINT below to match your server.
#   2. bash run_accuracy.sh
#
# Results are written to:
#   results/bfcl_v4_single_turn_accuracy/   (single-turn)
#
# Multi-turn is no longer part of the accuracy gate; see README Step 3 for the
# optional exploratory multi-turn run.

set -euo pipefail

MODEL="${MODEL:-Qwen3.6-27B-Q4_K_M}"
ENDPOINT="${ENDPOINT:-http://localhost:8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== BFCL v4 edge-agentic accuracy run ==="
echo "  Model:    $MODEL"
echo "  Endpoint: $ENDPOINT"
echo ""

# from-config reads model name and endpoint from the YAML (it has no
# model/endpoint override flags). Render a temp config with MODEL/ENDPOINT
# substituted in so the env vars above take effect without editing the
# committed YAML; the trailing "# set to your ..." comments anchor the edit.
ST_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$ST_CONFIG"' EXIT
sed -E \
    -e "s|^( *name: ).*(# set to your served model name\.)|\1\"${MODEL}\" \2|" \
    -e "s|^( *- ).*(# set to your endpoint URL\.)|\1\"${ENDPOINT}\" \2|" \
    offline_bfcl_v4_single_turn.yaml > "$ST_CONFIG"

# Single-turn: non_live (62%), live (10%), hallucination (10%) — ~995 samples, ~3 h
echo "--- Single-turn (~995 samples, ~3 h) ---"
inference-endpoint benchmark from-config \
    --config "$ST_CONFIG" \
    --accuracy-only

echo ""
echo "=== Done. Results in results/bfcl_v4_single_turn_accuracy/ ==="
