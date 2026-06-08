#!/usr/bin/env bash
# Deprecated: use run_vllm_accuracy_benchmark.sh for accuracy-only runs.
echo "NOTE: This script is deprecated. Use run_vllm_accuracy_benchmark.sh instead." >&2
exec "$(dirname "$0")/run_vllm_accuracy_benchmark.sh" "$@"
