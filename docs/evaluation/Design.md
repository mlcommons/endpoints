# Evaluation — Design Spec

> Scores model responses against ground-truth answers after an accuracy benchmark run; extracts answers from raw response text and supports LiveCodeBench code-execution evaluation via an external sandboxed server.

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · **evaluation** · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`evaluation/` scores model responses against ground-truth answers for accuracy benchmarks.
It is invoked after a benchmark run that collected responses (i.e. `--mode acc` or `--mode both`).
Today that orchestration happens in `commands/benchmark/execute.py`, not in `metrics/`.

## Responsibilities

- Extract model answers from raw response text
- Score extracted answers against ground truth
- Support LiveCodeBench code execution evaluation (requires external server)

## Component Map

```
QueryResult.response_output (raw response text)
        |
        v
extractor.py  -->  extracted answer string
        |
        v
scoring.py    -->  correct / incorrect (per sample)
        |
        v
accuracy summary written into benchmark results
```

## Files

| File             | Purpose                                                           |
| ---------------- | ----------------------------------------------------------------- |
| `extractor.py`   | Extracts model answer from raw text (regex, boxed-answer parsing) |
| `scoring.py`     | Compares extracted answer to ground truth label                   |
| `livecodebench/` | LiveCodeBench-specific code execution pipeline                    |

## LiveCodeBench

LiveCodeBench requires a sandboxed code execution server. The `livecodebench/` subdirectory
contains the server implementation and a Dockerfile. See
`src/inference_endpoint/evaluation/livecodebench/README.md` for setup
instructions.

Files:

- `_server.py` — FastAPI server that executes submitted code
- `lcb_serve.py` — Server management utilities
- `generate.py` — Response generation utilities
- `run_lcb_tests.py` — Test runner for LCB evaluation
- `lcb_serve.dockerfile` — Docker image for the execution server

## Scoring Methods

The scorer registry in `evaluation/scoring.py` currently includes:

| Method                | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `pass_at_1`           | Exact-match style scoring; also used by the LiveCodeBench path |
| `string_match`        | Whitespace-trimmed string equality                             |
| `rouge`               | ROUGE-based text generation scoring                            |
| `code_bench_scorer`   | LiveCodeBench code-execution scoring                           |
| `shopify_category_f1` | Shopify category F1 evaluation                                 |

The scoring configuration used by benchmark execution is specified per accuracy dataset under
`datasets[].accuracy_config`, including `accuracy_config.eval_method`,
`accuracy_config.extractor`, and optional `accuracy_config.ground_truth`.

## Design Decisions

**Extraction is separate from scoring**

Model responses for tasks like GPQA often embed the answer in verbose reasoning text. Extraction
(finding the answer in the text) and scoring (comparing the answer) are separate concerns.
Different datasets may share a scoring method but require different extraction logic.

**LiveCodeBench requires an external service**

Code execution cannot be done safely in-process. The evaluation server runs in a Docker container
with resource limits. This is a deliberate architecture choice — not a shortcut — and is
documented prominently in the dataset README.

## Integration Points

| Component                          | Role                                                   |
| ---------------------------------- | ------------------------------------------------------ |
| `commands/benchmark/execute.py`    | Builds scorer/extractor configs and runs scoring       |
| `dataset_manager/predefined/`      | Provides ground truth labels alongside prompts         |
| `evaluation/livecodebench/`        | Provides external execution path for LiveCodeBench     |
| `results.json` / benchmark reports | Receives computed accuracy summary during finalization |
