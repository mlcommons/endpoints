# Edge-Agentic (BFCL v4) Accuracy Benchmarking

## Quick start

To reproduce the reference accuracy number, edit
`model_params.name` and `endpoint_config.endpoints` in `online_edge_full_run.yaml`
to match your server, then run:

```bash
cd examples/11_Edge_Agentic_Example/
inference-endpoint benchmark from-config \
  --config online_edge_full_run.yaml \
  --accuracy-only
```

`--accuracy-only` runs the finalized single-turn accuracy benchmark (~995
samples) with the exact validated parameters and skips the performance phase. Drop
the flag to run performance + accuracy back-to-back (Step 5). See the steps below
for details.

---

## What is this?

This example runs [Berkeley Function Calling Leaderboard (BFCL) v4](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v4.html)
accuracy evaluation using the **MLPerf Inference Endpoints** benchmarking tool
([mlcommons/endpoints](https://github.com/mlcommons/endpoints)).

BFCL v4 tests whether a language model can correctly call functions / tools —
covering single-turn requests (one prompt → one structured tool call) and
agentic multi-turn conversations (parse call → execute locally → feed result
back → repeat).

The **finalized accuracy benchmark is single-turn only** (3 categories), with
per-category sampling tuned to draw **~995 samples** — a sample size large
enough for a stable point estimate — finishing on an edge device in **~3 hours**.
Multi-turn remains available as an optional exploratory run (Step 3) but is not
part of the accuracy gate.

---

## What is the Endpoints repo?

`mlcommons/endpoints` is a high-performance benchmarking tool for LLM inference
endpoints. It sends prompts to any OpenAI-compatible HTTP server, records
latency and accuracy metrics, and produces structured reports. You do not need
to know the internals — for this example you only use the `inference-endpoint`
CLI that comes with it.

---

## Step 0 — What you need before starting

| Requirement                         | Notes                                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Python 3.12+                        | Earlier versions not supported                                                                        |
| Git                                 | To clone the repo                                                                                     |
| A running model server              | Any OpenAI-compatible endpoint. Validated with `Qwen3.6-27B-Q4_K_M` via llama.cpp (see below)         |
| ~24 GB memory (GPU/VRAM or unified) | The Q4 GGUF is ~16.8 GB on disk; the rest is KV cache at `--ctx-size 32768`. 16 GB is **not** enough. |
| Time budget                         | Single-turn (3 categories), ~995 samples; several hours on a single-stream edge box                   |

### Obtaining the model

The reference runs use **Qwen3.6-27B** ([`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B),
Apache 2.0) quantized to **Q4_K_M GGUF** (~16.8 GB). Pull the GGUF from a public
Hugging Face quant repo — e.g. [`unsloth/Qwen3.6-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF):

```bash
pip install -U "huggingface_hub[cli]"
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir ./models
# -> ./models/Qwen3.6-27B-Q4_K_M.gguf
```

Alternatively, `llama-server` can fetch it directly with
`-hf unsloth/Qwen3.6-27B-GGUF:Q4_K_M` (requires a `LLAMA_CURL`-enabled build).

> Accuracy is reproducible from any correct Q4_K_M build of this model
> (deterministic decoding at `temperature 0` + fixed seed); exact GGUF byte size
> may differ slightly between quantizer versions. Use a model name / `--alias`
> of `Qwen3.6-27B-Q4_K_M` so it matches the config defaults below.

### Starting a model server

If you already have an OpenAI-compatible server running, skip this section.

This example was validated on an **NVIDIA Jetson AGX Thor** (aarch64, Blackwell GPU,
JetPack 7 / CUDA 13) using a **natively-built llama.cpp `llama-server`** at commit
**`cfff1fc`** — the reference results below were produced on this commit.

**Build llama.cpp with CUDA on Thor (one time):**

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# Pin to the validated reference commit (matches the Reference results below).
git checkout cfff1fc
# CUDA toolkit ships with JetPack at /usr/local/cuda (CUDA 13 on R38)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110   # Thor = sm_110 (cc 11.0)
cmake --build build --config Release -j --target llama-server
```

> **Other edge devices:** the steps above are the NVIDIA Jetson AGX Thor baseline; for other devices change accordingly. For example, for **DGX Spark (GB10)**
> set `-DCMAKE_CUDA_ARCHITECTURES=121` (`sm_121`, cc 12.1).

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
The single example config, `online_edge_full_run.yaml`, has the sampling rates
pre-configured; `--accuracy-only` runs just its accuracy phase.

```bash
# Run from the examples/11_Edge_Agentic_Example/ directory
cd examples/11_Edge_Agentic_Example/

inference-endpoint benchmark from-config \
  --config online_edge_full_run.yaml \
  --accuracy-only
```

`--accuracy-only` skips the performance (throughput) phase entirely and forces a
single worker and single connection for deterministic per-sample ordering. To run
accuracy **and** performance back-to-back, drop the flag (see **Step 5**).

Before running, open `online_edge_full_run.yaml` and set `model_params.name` to
match the model name your server reports (e.g. `Qwen3.6-27B-Q4_K_M`). The
`endpoint_config.endpoints` list defaults to `http://localhost:8080`.

**Sampling rates** (validated at ~995 samples):

| Category      | Sample rate               | Notes        |
| ------------- | ------------------------- | ------------ |
| non_live      | 62%                       | ~712 samples |
| live          | 10% (tiny subsets → 100%) | ~171 samples |
| hallucination | 10%                       | ~112 samples |

Subsets of size ≤ 25 are taken in full (`subset_floor: 25`) so their scores are
not reduced to one or two noisy samples. Total ≈ **995** single-turn samples.
Results are written to `results/edge_agentic_full_run/`.

---

## Step 3 — Run multi-turn evaluation (optional, not part of the accuracy gate)

> The finalized accuracy benchmark is **single-turn only** (Step 2). The
> multi-turn run below is **optional and exploratory** — it is not part of the
> reported accuracy gate. Its small sampled subsets are dominated by
> per-entry granularity noise, which is why single-turn (~995 samples) is the
> gateable metric.

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
r = json.loads(pathlib.Path('results/edge_agentic_full_run/results.json').read_text())
print('Overall ST accuracy:',
      r['accuracy_scores']['bfcl_v4::function_calling']['breakdown']['overall_accuracy'], '%')
"

# Multi-turn overall accuracy (only if you ran the optional Step 3)
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
> `results/edge_agentic_full_run/report.txt`.
>
> Result shape differs between the two pipelines: the **single-turn** entry
> stores `score` as a scalar fraction (0–1) with the per-subset dict under a
> separate `breakdown` key (hence `['breakdown']['overall_accuracy']` above),
> whereas the **multi-turn** CLI writes `score` as a dict (hence
> `['score']['overall_accuracy']` in the multi-turn snippet).

---

## Step 5 — Run the combined benchmark (performance + accuracy)

A single config, `online_edge_full_run.yaml`, runs the performance phase and the
accuracy phase back-to-back against the same server.

The **performance phase** replays recorded multi-turn agentic-coding trajectories
(SWE-bench-style) while an inline **"online checker"** scores the model's
generated tool calls against the recorded ones
(`accuracy_config.eval_method: agentic_inference_inline` on the performance
dataset); the dataset is both the performance workload and its own ground truth.
The **accuracy phase** is the BFCL v4 single-turn gate from Step 2.

```bash
# Run from the repo root (cd back if you followed Step 2), against the same
# server used in Step 0 (start it first). Unlike the accuracy-only runs above
# — whose BFCL dataset self-downloads — the combined run loads the performance
# dataset via the config's repo-root-relative path, so it must be launched from
# the repo root. Results land in ./results/edge_agentic_full_run/.
inference-endpoint benchmark from-config \
  --config examples/11_Edge_Agentic_Example/online_edge_full_run.yaml
```

Before running, open `online_edge_full_run.yaml` and set `model_params.name` to
your served model name. The config is single-stream (`target_concurrency: 1`,
matching `llama-server -np 1`), deterministic (`temperature 0`, `seed 42`),
reasoning **off**, runs **performance first, then accuracy**, and writes both
scores into one report directory (`results/edge_agentic_full_run/`).

| Phase       | Dataset                     | Conversations | Generated turns | Peak ISL       | Score             |
| ----------- | --------------------------- | ------------- | --------------- | -------------- | ----------------- |
| performance | `agentic_coding_2.5h.jsonl` | 20            | 1007            | ~23.5K (< 32K) | inline IoU 0.6335 |
| accuracy    | BFCL v4 single-turn (~995)  | —             | —               | < 32K          | overall 86.23%    |

The performance dataset is built so that **no conversation overflows the 32K
served context** — every turn completes and the run is _valid_ (0 dropped
turns). Serving optimizations (e.g. MTP speculative decoding) are expected to make
the performance phase substantially faster. Only raise `target_concurrency` when
pointing at a multi-slot endpoint.

> **Keep reasoning off.** On this tool-calling workload, enabling server-side
> reasoning gives no inline-accuracy benefit and costs ~60% more wall-clock (see
> the reference table below). Launch `llama-server` with `--reasoning off`.

### Verify the combined results

```bash
# Performance inline-checker score (a run is valid when no turns are missing)
python3 -c "
import json, pathlib
s = json.loads(pathlib.Path('results/edge_agentic_full_run/scores.json').read_text())
missing = s['turns']['missing']
print('Inline accuracy score:', s['score'], '| valid run:', missing == 0, '| missing turns:', missing)
"

# BFCL v4 single-turn accuracy (the gated metric)
python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('results/edge_agentic_full_run/results.json').read_text())
print('Overall ST accuracy:',
      r['accuracy_scores']['bfcl_v4::function_calling']['breakdown']['overall_accuracy'], '%')
"
```

Both scores land in `results/edge_agentic_full_run/`: the BFCL gate under
`results.json` (`accuracy_scores['bfcl_v4::function_calling']`) and the inline
performance checker in `scores.json`, alongside the performance metrics
(throughput, TTFT, TPOT, per-turn latency, ISL/OSL).

### Publish for the MLPerf submission checker

The MLPerf Inference submission checker (`tools/submission/submission_checker`
in `mlcommons/inference`, v5.0+) reads endpoints results directly from the
artifacts a run already writes — `result_summary.json`, `results.json`, and
`config.yaml` — so no separate "log" format is needed. `scripts/publish_submission.py`
copies that trio into the directory layout the checker walks and self-verifies
the fields it reads (primary-metric QPS, p99 latency, TTFT/TPOT p99, and the
accuracy score):

```bash
python scripts/publish_submission.py \
  --run results/edge_agentic_full_run \
  --output submission \
  --submitter <ORG> --system <SYSTEM_NAME> --benchmark qwen3.6-27b \
  --scaffold

# Then validate with the upstream checker:
python3 -m inference.tools.submission.submission_checker.main \
  --input submission --version v6.1 --submitter <ORG>
```

`--scaffold` also writes `measurements.json` and `systems/<system>.json`
templates (fill in the `TODO` fields before submitting). Use separate
`--performance-run` / `--accuracy-run` directories if you ran the two phases
separately.

### Reasoning ON vs OFF (Jetson Thor, Q4_K_M + llama.cpp)

Measured on **NVIDIA Jetson AGX Thor**, `Qwen3.6-27B-Q4_K_M` served single-slot
(`-np 1`, `--ctx-size 32768`, `--flash-attn on`, `-ngl 99`), driven single-stream
over `agentic_coding_2.5h.jsonl` (1007 generated turns, all completed):

| Dataset              | reasoning OFF                | reasoning ON         |
| -------------------- | ---------------------------- | -------------------- |
| `_2.5h` (1007 turns) | IoU **0.6335**, **2 h 37 m** | IoU 0.6374, 4 h 13 m |

Reasoning ON gives no meaningful accuracy change (IoU 0.6335 vs 0.6374, within
run-to-run noise) while costing ~60% more wall-clock. Based on this finding,
**this workload runs with `--reasoning off`** as the reference configuration.

---

## Reproducible runs with `--seed`

Fix the RNG used for sampling so the same seed + same model + a deterministic
server produce identical outputs across runs. How the seed is supplied depends
on the entrypoint: the `offline`/`online` CLI modes and the multi-turn CLI take
`--seed <N>`, but `from-config` does **not** — it reads `model_params.seed` from
the YAML (`online_edge_full_run.yaml` already sets `model_params.seed: 42`).

```bash
# Single-turn (from-config): seed comes from the YAML (model_params.seed: 42),
# not a CLI flag — `from-config` does not accept --seed.
inference-endpoint benchmark from-config \
  --config online_edge_full_run.yaml \
  --accuracy-only

# Multi-turn CLI: accepts --seed directly
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

## Reference results (single-turn, ~995 samples)

Validated with `Qwen3.6-27B-Q4_K_M`, `temperature=0`, `seed=42`, server launched
`--reasoning off --ctx-size 32768 -np 1` on llama.cpp `cfff1fc`. The single-turn
benchmark was run **twice, with a freshly restarted server each pass**.

| Device                  | Pass | Overall    | Normalized | non_live | live  | hallucination |
| ----------------------- | ---- | ---------- | ---------- | -------- | ----- | ------------- |
| Jetson Thor (reference) | run1 | **86.23%** | **87.96%** | 82.59    | 84.12 | 97.16         |
| Jetson Thor (reference) | run2 | **86.23%** | **87.96%** | 82.59    | 84.12 | 97.16         |

- **Run-to-run:** accuracy is **identical** across passes (deterministic
  decoding at `temperature 0` + fixed seeds).

### Accuracy gate

The pass/fail criterion is a **3% one-sided band** anchored on the Jetson Thor
`Qwen3.6-27B-Q4_K_M` reference above: a submission passes if its single-turn
score is **≥ 0.97 × reference**, with no upper bound (a higher score never
fails).

| Metric          | Reference | Pass threshold (0.97 ×) |
| --------------- | --------- | ----------------------- |
| Overall         | 86.23%    | **≥ 83.64%**            |
| Normalized (ST) | 87.96%    | **≥ 85.32%**            |

Accuracy is hardware-independent (deterministic at `temperature 0` + fixed seed),
so the same thresholds apply on any device. This gate is encoded in the ruleset
at `src/inference_endpoint/config/rulesets/mlcommons/models.py`
(`Qwen3_6_27B.accuracy_target_settings`).

### Optional: multi-turn parity

The optional multi-turn run (Step 3) is **not gated**. For reference, a single
run of the full 200-entry `multi_turn_base` (no sampling) scored
**140/200 = 70.00%**, in exact parity with evalscope (~80 min on an edge
device).

### Notes

Excluded from the reported aggregates: `live_relevance` (not part of any
reported category) and `memory` (not implemented on this branch).
