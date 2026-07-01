# DeepSeek-R1 Accuracy Example

MLPerf DeepSeek-R1 combined-subset accuracy benchmarking against an
OpenAI-compatible endpoint (e.g. TensorRT-LLM served via `api_type: openai` /
`openai_completions`). The exact MLPerf prompt is issued via pre-tokenized
`input_tokens`, bypassing the server chat template.

## Dataset

A prepared, pre-tokenized copy of the dataset ships here (git-LFS):

```
data/deepseek_r1_eval.parquet   # input_tokens, ground_truth, dataset, question
```

Point the `legacy_mlperf_deepseek_r1` predefined dataset at it (the loader accepts a prepared
`.parquet` only; building from the raw MLPerf source is out of scope):

```bash
export LEGACY_MLPERF_DEEPSEEK_R1_DATASET=examples/07_DeepSeekR1_Example/data/deepseek_r1_eval.parquet
```

## Accuracy evaluator subproject

Isolated `uv` environment that wraps the official MLCommons DeepSeek-R1
accuracy evaluator. `inference_endpoint.evaluation.scoring.LegacyMLPerfDeepSeekR1Scorer`
shells out to `deepseek_eval_runner.py` (under
`src/inference_endpoint/evaluation/legacy_mlperf_deepseek_r1/`) via `uv run --project`, so the
parent benchmark process never imports the (old/heavy, conflicting) evaluator
deps.

## What it scores

The MLPerf DeepSeek-R1 accuracy dataset is an ensemble of five subsets, each
graded differently and then aggregated into a single `exact_match` (golden
FP32 = `81.3582`) plus `tokens_per_sample` (golden = `3886.2274`):

| subset          | parse                         | grade                     |
| --------------- | ----------------------------- | ------------------------- |
| `gpqa`          | last `ANSWER: X` (A-D)        | letter match              |
| `mmlu_pro`      | last `ANSWER: X` (A-J)        | letter match              |
| `math500`       | `\boxed{...}`                 | prm800k symbolic grader   |
| `aime*`         | `\boxed{int}` / `Answer: int` | integer match (0-999)     |
| `livecodebench` | ` ```python ... ``` `         | execute vs LCB test cases |

> **`livecodebench` is graded by a container, not here.** In the
> `LegacyMLPerfDeepSeekR1Scorer` flow this subproject only _tokenizes_ the livecodebench
> rows (keeping `tokens_per_sample` correct) and marks them `external`; the
> generated code is executed out-of-band by the `lcb-service` WebSocket
> container at `ws://localhost:13835/evaluate` (the scorer's default). See
> [`livecodebench/README.md`](../../src/inference_endpoint/evaluation/livecodebench/README.md)
> for launching the container. Grading LCB inside this subproject would require a
> 3.12-compatible `pyext` (intentionally omitted - see `pyproject.toml`) and is
> not the supported path.

## Preconditions

- **`uv`** on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- **Network egress** to GitHub + PyPI + HuggingFace Hub: `setup_eval.sh`
  fetches the evaluator and clones two submodules; the runner downloads the
  DeepSeek tokenizer and (for `livecodebench`) the LCB dataset on first use.
- The parent endpoints env is already synced (`uv sync --extra dev` from the
  repo root). The evaluator deps live in **this** subproject only.

## Setup (run once on the accuracy host)

```bash
cd src/inference_endpoint/evaluation/legacy_mlperf_deepseek_r1
uv sync                       # resolves the isolated eval env into .venv/
bash setup_eval.sh            # fetches eval_accuracy.py + prm800k + LiveCodeBench
```

`setup_eval.sh` populates `mlperf_eval/` (gitignored):

```
mlperf_eval/
  eval_accuracy.py            # mlcommons/inference @ e59ce58
  submodules/prm800k/         # openai/prm800k @ 7ecc794
  submodules/LiveCodeBench/   # LiveCodeBench @ 28fef95  (installed --no-deps)
```

Sanity-check the import:

```bash
uv run python -c "import sys; sys.path.insert(0,'mlperf_eval'); import eval_accuracy; print('ok')"
```

## Smoke test (no GPU, no server)

Score a tiny hand-made set of outputs covering several subsets:

```bash
uv run python - <<'PY'
import pandas as pd
pd.DataFrame([
    {"dataset":"math500","ground_truth":"8","question":"q",
     "model_output":r"The answer is \boxed{8}."},
    {"dataset":"aime1983","ground_truth":"42","question":"q",
     "model_output":r"Hence \boxed{42}."},
    {"dataset":"gpqa","ground_truth":"B","question":"q",
     "model_output":"After reasoning, ANSWER: B"},
]).to_parquet("/tmp/ds_smoke.parquet", index=False)
PY

uv run python deepseek_eval_runner.py \
  --input /tmp/ds_smoke.parquet \
  --output /tmp/ds_smoke_results.json \
  --tokenizer deepseek-ai/DeepSeek-R1
cat /tmp/ds_smoke_results.json
```

Expect `exact_match: 100.0` (these three are all correct), `per_dataset` with
three `status: ok` entries, and `complete: true`. (The `livecodebench` subset
is exercised end-to-end only when its rows are present and the LCB dataset is
reachable.)

## How the scorer calls this

`LegacyMLPerfDeepSeekR1Scorer.score()` writes
`<report_dir>/deepseek_eval/<name>_outputs.parquet`, then runs:

```bash
uv run --project <this dir> python deepseek_eval_runner.py \
  --input  <report_dir>/deepseek_eval/<name>_outputs.parquet \
  --output <report_dir>/deepseek_eval/<name>_results.json \
  --tokenizer <model path or HF id>
```

stdout/stderr are captured to `<report_dir>/deepseek_eval_subprocess.log`.
