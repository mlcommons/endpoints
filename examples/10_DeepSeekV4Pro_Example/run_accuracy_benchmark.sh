#!/usr/bin/env bash
# Deprecated: use run_benchmark.sh (repo-standard from-config workflow).
echo "NOTE: This script is deprecated. Use run_benchmark.sh instead." >&2
exec "$(dirname "$0")/run_benchmark.sh" "$@"
