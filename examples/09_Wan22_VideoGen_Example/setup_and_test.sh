#!/usr/bin/env bash
# setup_and_test.sh — Set up the WAN2.2 environment and run unit tests.
#
# Usage:
#   bash setup_and_test.sh [--skip-setup]
#
#   --skip-setup   Skip venv creation and pip install (use existing venv).
#
# Prerequisites:
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

    echo "==> Installing package with [videogen,test] extras"
    "${VENV_DIR}/bin/pip" install --upgrade pip --quiet
    "${VENV_DIR}/bin/pip" install -e ".[videogen,test]" --quiet
else
    echo "==> Skipping setup (--skip-setup)"
fi

PYTEST="${VENV_DIR}/bin/pytest"

if [[ ! -x "$PYTEST" ]]; then
    echo "ERROR: pytest not found at ${PYTEST}. Run without --skip-setup first."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Locate bundled prompts dataset
# ---------------------------------------------------------------------------
PROMPTS_JSONL="${SCRIPT_DIR}/wan22_prompts.jsonl"
echo "==> Using bundled prompts dataset: ${PROMPTS_JSONL}"

# ---------------------------------------------------------------------------
# 3. Run WAN2.2 unit tests
# ---------------------------------------------------------------------------
echo ""
echo "==> Running WAN2.2 unit tests"
"$PYTEST" tests/unit/videogen/ \
    -v \
    --tb=short \
    --no-cov \
    -q

echo ""
echo "==> Running WAN2.2 integration tests"
"$PYTEST" tests/integration/videogen/ \
    -v \
    --tb=short \
    --no-cov \
    -q || { code=$?; [ $code -eq 5 ] && echo "No integration tests collected (skipped)" || exit $code; }

echo ""
echo "All tests passed."
