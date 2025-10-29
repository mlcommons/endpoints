#!/usr/bin/env bash
# Setup script for PyTorch CPU-only installation for POC benchmarking
set -euo pipefail

echo "Setting up PyTorch CPU-only installation for POC benchmarking..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
