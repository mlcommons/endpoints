#!/usr/bin/env bash
# setup_and_test.sh — Set up the WAN2.2 environment on Lyris and run unit tests.
#
# Usage:
#   bash setup_and_test.sh [--skip-setup]
#
#   --skip-setup   Skip venv creation and pip install (use existing venv).
#
# Prerequisites on Lyris:
#   - Python 3.12 available as python3.12
#   - Access to the repo root (this script lives in examples/09_Wan22_VideoGen_Example/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

SKIP_SETUP=false
for arg in "$@"; do
    [[ "$arg" == "--skip-setup" ]] && SKIP_SETUP=true
done

cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------
if [[ "$SKIP_SETUP" == false ]]; then
    echo "==> Creating virtual environment at ${VENV_DIR}"
    python3.12 -m venv "${VENV_DIR}"

    echo "==> Installing package with [wan22,test] extras"
    "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
    "${VENV_DIR}/bin/pip" install -e ".[wan22,test]" --quiet
else
    echo "==> Skipping setup (--skip-setup)"
fi

PYTHON="${VENV_DIR}/bin/python"
PYTEST="${VENV_DIR}/bin/pytest"

if [[ ! -x "$PYTEST" ]]; then
    echo "ERROR: pytest not found at ${PYTEST}. Run without --skip-setup first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Generate JSONL dataset from prompts.txt (idempotent)
# ---------------------------------------------------------------------------
PROMPTS_TXT="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone/preprocessed_data/wan22-a14b/prompts.txt"
PROMPTS_JSONL="${REPO_ROOT}/datasets/wan22_prompts.jsonl"

if [[ -f "$PROMPTS_TXT" && ! -f "$PROMPTS_JSONL" ]]; then
    echo "==> Converting prompts.txt -> wan22_prompts.jsonl"
    mkdir -p "$(dirname "$PROMPTS_JSONL")"
    PROMPTS_TXT="$PROMPTS_TXT" PROMPTS_JSONL="$PROMPTS_JSONL" "$PYTHON" - <<'EOF'
import json, pathlib, os
src = pathlib.Path(os.environ["PROMPTS_TXT"])
dst = pathlib.Path(os.environ["PROMPTS_JSONL"])
lines = [l.strip() for l in src.read_text().splitlines() if l.strip()]
with dst.open("w") as f:
    for i, p in enumerate(lines):
        f.write(json.dumps({"prompt": p, "sample_id": str(i), "sample_index": i,
                            "negative_prompt": "", "mode": "perf"}) + "\n")
print(f"Written {len(lines)} prompts to {dst}")
EOF
elif [[ -f "$PROMPTS_JSONL" ]]; then
    echo "==> wan22_prompts.jsonl already exists, skipping conversion"
else
    echo "WARNING: prompts.txt not found at ${PROMPTS_TXT}, skipping JSONL generation"
fi

# ---------------------------------------------------------------------------
# 3. Run WAN2.2 unit tests
# ---------------------------------------------------------------------------
echo ""
echo "==> Running WAN2.2 unit tests"
"$PYTEST" tests/unit/wan22/ \
    -v \
    --tb=short \
    --no-cov \
    -q

echo ""
echo "==> Running WAN2.2 integration tests"
"$PYTEST" tests/integration/wan22/ \
    -v \
    --tb=short \
    --no-cov \
    -q || { code=$?; [ $code -eq 5 ] && echo "No integration tests collected (skipped)" || exit $code; }

echo ""
echo "All tests passed."
