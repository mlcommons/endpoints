#!/usr/bin/env bash
# score_livecodebench.sh - score the LiveCodeBench subset of a completed
# DeepSeek-R1 accuracy run, on a compute node (SLURM).
#
# WHY a separate job: the `deepseek_r1` scorer's MLCommons evaluator grades the
# four text subsets (math500/aime/gpqa/mmlu_pro) fine, but its in-process LCB
# code executor (concurrent.futures) can't kill runaway model code, so the pool
# hangs. And the LCB dataset load needs ~21GB, which the login node's per-user
# cgroup OOM-kills. So LCB is scored here, on a clean aarch64 compute node, with
# the repo's hardened `lcb_serve` (kill-on-timeout sandboxing of each execution).
#
# Submit:
#   sbatch examples/10_DeepSeekR1_Example/score_livecodebench.sh
# Overridable: OUTPUTS_PARQUET (the scorer's saved outputs), LCB_VARIANT.
#
#SBATCH --job-name=dsr1-lcb-score
#SBATCH --account=<your-slurm-account>
#SBATCH --partition=<your-partition>
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/dsr1_lcb_%j.out

set -uo pipefail
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "${REPO_ROOT}"
echo "=== node $(hostname) arch $(uname -m) @ $(date -u +%H:%M:%S) ==="

EX="examples/10_DeepSeekR1_Example"
OUTPUTS_PARQUET="${OUTPUTS_PARQUET:-${REPO_ROOT}/logs/deepseek_r1_fp4_accuracy/deepseek_eval/deepseek_r1_accuracy_outputs.parquet}"
LCB_VARIANT="${LCB_VARIANT:-release_v6}"   # superset, so all question_ids resolve
LCBDIR="${REPO_ROOT}/src/inference_endpoint/evaluation/deepseek_r1/lcb_datasets"
LCBIN="${LCBDIR}/lcb_input.parquet"
RESULTS="${LCBDIR}/lcb_results.json"

# aarch64 uv + a minimal eval venv (datasets==3.6.0 + what lcb_serve imports).
ARCHUV_DIR="${HOME}/.local/uv-aarch64"
if ! "${ARCHUV_DIR}/uv" --version >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${ARCHUV_DIR}" INSTALLER_NO_MODIFY_PATH=1 sh
fi
UV="${ARCHUV_DIR}/uv"
EVALVENV="${REPO_ROOT}/.venv-lcb-aarch64"
PY="${EVALVENV}/bin/python"
"${UV}" venv "${EVALVENV}" --python 3.12
"${UV}" pip install --python "${PY}" -q "datasets==3.6.0" pandas tqdm numpy msgspec

echo "=== generate LCB ${LCB_VARIANT} test cases ==="
"${PY}" src/inference_endpoint/evaluation/livecodebench/generate.py \
  --datasets-dir "${LCBDIR}" --variant "${LCB_VARIANT}"

echo "=== extract code + question_id from saved outputs ==="
"${PY}" - "${OUTPUTS_PARQUET}" "${LCBIN}" <<'PY'
import pandas as pd, re, sys
df = pd.read_parquet(sys.argv[1])
lcb = df[df["dataset"] == "livecodebench"].copy()
extract = lambda t: (re.findall(r"```python(.*?)```", str(t), re.DOTALL) or [""])[-1].strip()
lcb["extracted_code"] = lcb["model_output"].map(extract)
lcb["question_id"] = lcb["ground_truth"].astype(str)
lcb[["question_id", "extracted_code"]].to_parquet(sys.argv[2], index=False)
print("lcb rows:", len(lcb))
PY

echo "=== hardened lcb_serve eval (kill-on-timeout) ==="
# lcb_serve writes progress/log lines to stdout alongside the final JSON, so
# capture everything to a .raw log and keep only the last JSON object as the
# results file (a wholesale redirect leaves non-JSON lines that break parsing).
PYTHONPATH="${REPO_ROOT}/src" "${PY}" -m inference_endpoint.evaluation.livecodebench.lcb_serve \
  "${LCBIN}" --datasets-dir "${LCBDIR}" --version-tag "${LCB_VARIANT}" > "${RESULTS}.raw" 2>&1
grep -E '^\s*\{.*\}\s*$' "${RESULTS}.raw" | tail -1 > "${RESULTS}"
echo "=== ${RESULTS} ==="; cat "${RESULTS}"; echo
if ! [ -s "${RESULTS}" ]; then
  echo "ERROR: lcb_serve produced no JSON result; tail of ${RESULTS}.raw:" >&2
  tail -n 20 "${RESULTS}.raw" >&2
  exit 1
fi
echo "Fold into the aggregate: full_exact_match = (text_subset_correct + LCB passed_samples) / 4388."
