# BFCL v4 Accuracy Benchmarking

## What is this?

This example runs [Berkeley Function Calling Leaderboard (BFCL) v4](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v4.html)
accuracy evaluation using the **MLPerf Inference Endpoints** benchmarking tool
([mlcommons/endpoints](https://github.com/mlcommons/endpoints)).

BFCL v4 tests whether a language model can correctly call functions / tools —
covering single-turn requests (one prompt → one structured tool call) and
agentic multi-turn conversations (parse call → execute locally → feed result
back → repeat).

The sampling rates here are tuned so the full four-category run finishes on an
edge device in **under 3 hours**.

---

## What is the Endpoints repo?

`mlcommons/endpoints` is a high-performance benchmarking tool for LLM inference
endpoints. It sends prompts to any OpenAI-compatible HTTP server, records
latency and accuracy metrics, and produces structured reports. You do not need
to know the internals — for this example you only use the `inference-endpoint`
CLI that comes with it.

---

## Step 0 — What you need before starting

| Requirement | Notes |
| --- | --- |
| Python 3.12+ | Earlier versions not supported |
| Git | To clone the repo |
| A running model server | Any OpenAI-compatible endpoint. Validated with `Qwen3.6-27B-Q4_K_M` via llama.cpp (see below) |
| ~16 GB RAM (GPU or CPU) | For the 27B Q4 model |
| ~3 hours wall-clock | For the sampled four-category run |

### Starting a model server (llama.cpp example)

If you already have an OpenAI-compatible server running, skip this section.
The commands below use llama.cpp's Docker image as a quick-start:

```bash
# Pull the model (adjust path/model name for your setup)
# Example: Qwen3-27B-Q4_K_M in GGUF format from HuggingFace
docker run --rm -it \
  -p 8080:8080 \
  -v /path/to/your/models:/models \
  ghcr.io/ggerganov/llama.cpp:server \
  -m /models/Qwen3-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 \
  -ngl 99          # layers to offload to GPU; set 0 for CPU-only
```

Verify it is up:

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

---

## Step 1 — Install from the PRs

This feature spans two pull requests. Both must be present:

| PR | Branch | What it does |
| --- | --- | --- |
| [PR #1](https://github.com/Palanivelg/endpoints/pull/1) | `chore/relax-numpy-pin` | Relaxes the `numpy` pin so `bfcl-eval` can install alongside the project |
| [PR #2](https://github.com/Palanivelg/endpoints/pull/2) | `feat/bfcl-v4-combined` | Full BFCL v4 single-turn + multi-turn accuracy integration |

**PR #1 is a prerequisite for PR #2.** Without it, `pip install -e ".[bfcl]"`
fails with a numpy version conflict.

```bash
# 1. Clone the fork that contains both PRs
git clone https://github.com/Palanivelg/endpoints.git
cd endpoints

# 2. Check out the BFCL v4 branch (PR #2; already includes the PR #1 change)
git checkout feat/bfcl-v4-combined

# 3. Install the package with the [bfcl] scoring extra
pip install -e ".[bfcl]"

# 4. Confirm the install resolved without conflict
pip show bfcl-eval numpy | grep -E "^(Name|Version)"
# Expected: bfcl-eval present, numpy 1.26.x

# 5. Confirm the CLI is available
inference-endpoint --help
```

---

## Step 2 — Run single-turn evaluation

Single-turn covers three BFCL v4 categories: `non_live`, `live`, and
`hallucination`. Each sample is one prompt → one structured tool-call response.
The YAML config in this directory has the sampling rates pre-configured.

```bash
# Run from the examples/10_BFCLv4_Example/ directory
cd examples/10_BFCLv4_Example/

inference-endpoint benchmark from-config \
  --config offline_bfcl_v4_single_turn.yaml \
  --accuracy-only
```

`--accuracy-only` skips the performance (throughput) phase and forces a single
worker and single connection for deterministic per-sample ordering.

Before running, open `offline_bfcl_v4_single_turn.yaml` and set
`model_params.name` to match the model name your server reports (e.g.
`Qwen3.6-27B-Q4_K_M`). The `endpoint_config.endpoints` list defaults to
`http://localhost:8080`.

**Sampling rates** (validated for <3 h on an edge device):

| Category | Sample rate | Notes |
| --- | --- | --- |
| non_live | 20% | ~230 samples |
| live | 10% (tiny subsets → 100%) | ~171 samples |
| hallucination | 5% | ~56 samples |

Results are written to `results/bfcl_v4_single_turn_accuracy/`.

---

## Step 3 — Run multi-turn evaluation

Multi-turn is an agentic loop: the model calls a function, the runner executes
it locally, feeds the result back, and the loop continues until all turns in
the test case are complete. It cannot use the same YAML pipeline as single-turn
and is driven by its own CLI.

```bash
python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \
  --endpoint http://localhost:8080 \
  --model Qwen3.6-27B-Q4_K_M \
  --sample-pct 3 \
  --temperature 0 \
  --report-dir results/bfcl_v4_multi_turn/
```

`--sample-pct 3` takes ~3% of each subset (~24 entries total). Omit it to run
all ~200 entries in `multi_turn_base` (expect ~80 min on an edge device).

Results are written to `results/bfcl_v4_multi_turn/`.

---

## Step 4 — Verify the results

```bash
# Single-turn overall accuracy
python3 -c "
import json, pathlib
s = json.loads(pathlib.Path('results/bfcl_v4_single_turn_accuracy/accuracy_scores.json').read_text())
print(s)
"

# Multi-turn overall accuracy
python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('results/bfcl_v4_multi_turn/results.json').read_text())
print('Overall MT accuracy:', r['accuracy_scores']['bfcl_v4::multi_turn']['score']['overall_accuracy'], '%')
"
```

---

## Reproducible runs with `--seed`

Pass `--seed <N>` to fix the server-side RNG. The same seed + same model +
deterministic server will produce identical outputs across runs.

```bash
# Single-turn with seed
inference-endpoint benchmark from-config \
  --config offline_bfcl_v4_single_turn.yaml \
  --accuracy-only \
  --seed 42

# Multi-turn with seed
python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \
  --endpoint http://localhost:8080 \
  --model Qwen3.6-27B-Q4_K_M \
  --sample-pct 3 \
  --temperature 0 \
  --seed 42 \
  --report-dir results/bfcl_v4_multi_turn/
```

`seed` is sent as the `seed` field in the `/v1/chat/completions` request body.
Servers that do not support it ignore it silently.

---

## Reference results (Thor edge device)

Validated on Thor (`Qwen3.6-27B-Q4_K_M`, `temperature=0`). Two independent
seed runs were executed to confirm determinism:

| Category | Run 1 | Run 2 | Match? |
| --- | --- | --- | --- |
| Single-turn `non_live` (AST, 456 samples) | 86.98% | 86.98% | ✓ |
| Single-turn `live` | 84.12% | 84.12% | ✓ |
| Single-turn `hallucination` | 94.32% | 94.32% | ✓ |
| **Single-turn overall** | **87.50%** | **87.50%** | ✓ |
| Multi-turn `multi_turn_base` (200/200 entries) | 70.00% (140/200) | — | exact parity with evalscope |

Wall-clock: ~82 min single-turn (~10.8 s/sample), ~80 min multi-turn for the
full `multi_turn_base`.

Excluded from aggregates: `live_relevance` (not part of any reported category)
and `memory` (not implemented on this branch).
