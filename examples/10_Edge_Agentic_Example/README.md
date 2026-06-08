# Edge-Agentic (BFCL v4) Accuracy Benchmarking

## Quick start

To reproduce all reference accuracy numbers (~2.5 h on an edge device), set
your model name and endpoint URL, then run:

```bash
MODEL=Qwen3.6-27B-Q4_K_M ENDPOINT=http://localhost:8080 bash run_accuracy.sh
```

`run_accuracy.sh` runs both single-turn and multi-turn phases end-to-end with
the exact validated parameters. See the steps below if you prefer to run each
phase individually or need to customise the configuration.

---

## What is this?

This example runs [Berkeley Function Calling Leaderboard (BFCL) v4](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v4.html)
accuracy evaluation using the **MLPerf Inference Endpoints** benchmarking tool
([mlcommons/endpoints](https://github.com/mlcommons/endpoints)).

BFCL v4 tests whether a language model can correctly call functions / tools —
covering single-turn requests (one prompt → one structured tool call) and
agentic multi-turn conversations (parse call → execute locally → feed result
back → repeat).

The sampling rates here are tuned so single-turn (3 categories) plus a sampled
multi-turn run finish on an edge device in **~2.5–3 hours**.

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
| ~24 GB memory (GPU/VRAM or unified) | The Q4 GGUF is ~16.8 GB on disk; the rest is KV cache at `--ctx-size 32768`. 16 GB is **not** enough. |
| ~2.5–3 hours wall-clock | Single-turn (3 categories) + sampled multi-turn |

### Starting a model server

If you already have an OpenAI-compatible server running, skip this section.

This example was validated on an **NVIDIA Jetson Thor** (aarch64, Blackwell GPU,
JetPack 7 / CUDA 13) using a **natively-built llama.cpp `llama-server`**.

> ⚠️ The prebuilt `ghcr.io/ggerganov/llama.cpp` Docker images do **not** work for
> GPU inference on Thor: the plain `:server` tag is CPU-only (so `-ngl` is a
> no-op and the 27B model runs entirely on CPU), and the CUDA tags are `x86_64`
> builds that do not target Thor's `sm_110` / aarch64-SBSA. On Thor, build
> llama.cpp from source.

**Build llama.cpp with CUDA on Thor (one time):**

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# CUDA toolkit ships with JetPack at /usr/local/cuda (CUDA 13 on R38)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110   # Thor = sm_110 (cc 11.0)
cmake --build build --config Release -j --target llama-server
```

**Start the server (matches the validated reproducibility runs):**

```bash
./build/bin/llama-server \
  --model /path/to/Qwen3.6-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  -np 1 \
  --reasoning off \
  --flash-attn on \
  --n-gpu-layers 99 \
  --seed 42
```

Verify it is up:

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

<details><summary>x86_64 / discrete-GPU users (Docker quick-start)</summary>

On a workstation or server with a CUDA dGPU you can use llama.cpp's Docker image
instead of building from source. Use a CUDA image tag and expose the GPU with
`--gpus all`:

```bash
docker run --rm -it --gpus all \
  -p 8080:8080 \
  -v /path/to/your/models:/models \
  ghcr.io/ggerganov/llama.cpp:server-cuda \
  -m /models/Qwen3.6-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  -ngl 99
```
</details>

---

## Step 1 — Install

```bash
# 1. Clone the repo
git clone https://github.com/mlcommons/endpoints.git
cd endpoints

# 2. Install the package with the [bfcl] scoring extra
pip install -e ".[bfcl]"

# 3. Confirm the install resolved without conflict
pip show bfcl-eval numpy | grep -E "^(Name|Version)"
# Expected: bfcl-eval present, numpy >= 1.26.4

# 4. Confirm the CLI is available
inference-endpoint --help
```

---

## Step 2 — Run single-turn evaluation

Single-turn covers three BFCL v4 categories: `non_live`, `live`, and
`hallucination`. Each sample is one prompt → one structured tool-call response.
The YAML config in this directory has the sampling rates pre-configured.

```bash
# Run from the examples/10_Edge_Agentic_Example/ directory
cd examples/10_Edge_Agentic_Example/

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

**Sampling rates** (validated for ~82 min single-turn on an edge device):

| Category | Sample rate | Notes |
| --- | --- | --- |
| non_live | 20% | ~230 samples |
| live | 10% (tiny subsets → 100%) | ~171 samples |
| hallucination | 5% | ~56 samples |

Total ≈ 456 single-turn samples. Results are written to
`results/bfcl_v4_single_turn_accuracy/`.

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
  --seed 42 \
  --max-steps-per-turn 25 \
  --report-dir results/bfcl_v4_multi_turn/
```

`--sample-pct 3` takes ~3% of each multi-turn subset (~24 entries total across
`multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, and
`multi_turn_long_context`). Omit it to run the full set (the ~200-entry
`multi_turn_base` plus the other subsets; expect ~80 min for `multi_turn_base`
alone on an edge device), or pass `--subsets multi_turn_base` to restrict to one
subset.

Results are written to `results/bfcl_v4_multi_turn/`.

---

## Step 4 — Verify the results

```bash
# Single-turn overall accuracy
python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('results/bfcl_v4_single_turn_accuracy/results.json').read_text())
print('Overall ST accuracy:',
      r['accuracy_scores']['bfcl_v4::function_calling']['score']['overall_accuracy'], '%')
"

# Multi-turn overall accuracy
python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('results/bfcl_v4_multi_turn/results.json').read_text())
print('Overall MT accuracy:',
      r['accuracy_scores']['bfcl_v4::multi_turn']['score']['overall_accuracy'], '%')
"
```

> Note: the single-turn pipeline writes `results.json` (accuracy nested under
> `accuracy_scores['bfcl_v4::function_calling']`). There is no separate
> `accuracy_scores.json`. A human-readable summary is also written to
> `results/bfcl_v4_single_turn_accuracy/report.txt`.

---

## Reproducible runs with `--seed`

Pass `--seed <N>` to fix the RNG used for sampling. The same seed + same model +
a deterministic server produce identical outputs across runs.

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

**Server-side determinism also matters.** The validated runs launched
`llama-server` with `--seed 42` and a single slot (`-np 1`) — see the server
command in Step 0. A multi-slot / dynamically-batched server can still produce
run-to-run variation even with a fixed client seed.

---

## Reference results (Thor edge device)

Validated on Thor (`Qwen3.6-27B-Q4_K_M`, `temperature=0`, `seed=42`).

### Determinism — two independent seed runs

The full sampled suite (single-turn + `--sample-pct 3` multi-turn) was run twice,
end to end, with a freshly restarted server each pass. Every score was identical.

| Metric | Run 1 | Run 2 | Match? |
| --- | --- | --- | --- |
| Single-turn `non_live` (AST, ~230 samples) | 86.98% | 86.98% | ✓ |
| Single-turn `live` (~171 samples) | 84.12% | 84.12% | ✓ |
| Single-turn `hallucination` (~56 samples) | 94.32% | 94.32% | ✓ |
| **Single-turn overall (456 samples)** | **87.50%** | **87.50%** | ✓ |
| Multi-turn `multi_turn_base` | 66.67% (4/6) | 66.67% (4/6) | ✓ |
| Multi-turn `multi_turn_miss_func` | 33.33% (2/6) | 33.33% (2/6) | ✓ |
| Multi-turn `multi_turn_miss_param` | 16.67% (1/6) | 16.67% (1/6) | ✓ |
| Multi-turn `multi_turn_long_context` | 66.67% (4/6) | 66.67% (4/6) | ✓ |
| **Multi-turn overall (24 sampled entries)** | **45.84%** | **45.84%** | ✓ |

Wall-clock per pass: ~82 min single-turn (~10.8 s/sample) + ~64 min multi-turn
(~159 s/entry) ≈ **2.4–2.5 h**. Run-to-run timing varied < 1.1%; accuracy did
not vary at all.

### Accuracy parity — full `multi_turn_base`

A separate single run of the full 200-entry `multi_turn_base` (no sampling)
scored **140/200 = 70.00%**, in exact parity with evalscope. This run takes
~80 min on an edge device and is not part of the determinism check above.

### Notes

Excluded from the reported aggregates: `live_relevance` (not part of any
reported category) and `memory` (not implemented on this branch).
