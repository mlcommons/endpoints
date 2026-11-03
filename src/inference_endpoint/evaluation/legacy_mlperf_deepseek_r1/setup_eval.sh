#!/usr/bin/env bash
# setup_eval.sh - lay out the MLCommons DeepSeek-R1 evaluator under ./mlperf_eval/.
#
# eval_accuracy.py is vendored (committed) at mlperf_eval/eval_accuracy.py - it
# is a single file and avoids a network fetch at setup time. This script only
# clones its two submodules (prm800k, LiveCodeBench) at pinned commits, laid
# out exactly as eval_accuracy.py expects (it resolves submodules relative to
# its own __file__):
#
#   mlperf_eval/
#     eval_accuracy.py          # vendored from mlcommons/inference @ e59ce58
#     submodules/prm800k/
#     submodules/LiveCodeBench/
#
# Run once on the accuracy host after `uv sync`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${SCRIPT_DIR}/mlperf_eval"
SUBMODULES_DIR="${EVAL_DIR}/submodules"

# Pinned upstream commits (resolved 2026-06-01). Bump deliberately.
# eval_accuracy.py is vendored from mlcommons/inference @ MLC_INFERENCE_COMMIT;
# re-vendor it (and bump this commit) when updating the evaluator.
MLC_INFERENCE_REPO="https://github.com/mlcommons/inference"
MLC_INFERENCE_COMMIT="e59ce582f544edcc1b3f69a6c6f3ebc66eecb3d7"
PRM800K_REPO="https://github.com/openai/prm800k"
PRM800K_COMMIT="7ecc794703b2877f63226f2477a49b34f9b25163"
LCB_REPO="https://github.com/LiveCodeBench/LiveCodeBench"
LCB_COMMIT="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

mkdir -p "${SUBMODULES_DIR}"

if [ ! -f "${EVAL_DIR}/eval_accuracy.py" ]; then
    echo "ERROR: vendored eval_accuracy.py missing at ${EVAL_DIR}." >&2
    echo "It is committed in the repo; check out the full tree." >&2
    exit 1
fi

clone_pinned() {
    local repo="$1" commit="$2" dest="$3"
    if [ -d "${dest}/.git" ]; then
        echo "==> ${dest} already cloned; fetching pinned commit"
        git -C "${dest}" fetch --depth 1 origin "${commit}"
    else
        echo "==> Cloning ${repo}"
        git clone --filter=blob:none "${repo}" "${dest}"
        git -C "${dest}" fetch --depth 1 origin "${commit}"
    fi
    git -C "${dest}" checkout --detach "${commit}"
}

clone_pinned "${PRM800K_REPO}" "${PRM800K_COMMIT}" "${SUBMODULES_DIR}/prm800k"
clone_pinned "${LCB_REPO}" "${LCB_COMMIT}" "${SUBMODULES_DIR}/LiveCodeBench"

echo "==> Installing LiveCodeBench (no deps; eval-path deps are pinned in pyproject.toml)"
uv pip install --no-deps -e "${SUBMODULES_DIR}/LiveCodeBench"

echo "==> Done. Evaluator ready at ${EVAL_DIR}"
