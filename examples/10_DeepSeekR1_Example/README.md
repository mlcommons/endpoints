# DeepSeek-R1 FP4 - MLPerf Accuracy on TensorRT-LLM (GB200x4)

End-to-end example: serve **DeepSeek-R1 (FP4, ModelOpt)** with **trtllm-serve**
on a single **GB200x4** node and score the **official MLCommons DeepSeek-R1
accuracy dataset** (4388 samples) with this repo's `inference-endpoint` tool.

Accuracy is the official MLCommons _combined-subset_ evaluation - `math500`,
`aime`, `gpqa`, `mmlu_pro`, `livecodebench`, each graded by its own parser and
aggregated into one `exact_match` plus `tokens_per_sample`, via the
`deepseek_r1` scorer (which shells out to the isolated `accuracy/` subproject -
see [`accuracy/RUNBOOK.md`](accuracy/RUNBOOK.md)).

| Metric              | Golden (FP32) | Pass criterion             |
| ------------------- | ------------- | -------------------------- |
| `exact_match`       | 81.3582       | >= 99% of golden (>=80.52) |
| `tokens_per_sample` | 3886.2274     | within 90-110%             |

## Files

| File                                       | Purpose                                                           |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `prepare_dataset.py`                       | pkl -> parquet (+ `--subset N` stratified slice, + tiny perf set) |
| `trtllm_serve_config.yaml`                 | `trtllm-serve --extra_llm_api_options` for 4 GPUs (TP=4, EP=4)    |
| `launch_and_run.sh`                        | SLURM launch: serve -> health -> (probe + run \| `SERVER_ONLY`)   |
| `run_client.sh`                            | Drive the benchmark from the login node (cross-arch clusters)     |
| `score_livecodebench.sh`                   | Score the LCB subset on a compute node (hardened sandbox)         |
| `offline_deepseek_r1_accuracy.yaml`        | Full 4388-sample accuracy config                                  |
| `offline_deepseek_r1_accuracy_subset.yaml` | ~385-sample representative config (quick estimate)                |
| `accuracy/`                                | Isolated `uv` subproject wrapping the MLCommons evaluator         |

## WARNING Read first - verified gotchas on a GB200 SLURM cluster

These are baked into the scripts/configs already; listed so you know _why_.

1. **Checkpoint:** the `deepseek_r1-torch-fp4-v2` (FP4-WO) checkpoint **does not
   load** in stock `trtllm-serve` (1.2.0rc6 / 1.3.0rc14): its `hf_quant_config`
   excludes the post-fusion `self_attn.fused_a` while the loader checks the
   pre-fusion `q_a_proj`/`kv_a_proj` names -> FP4-packed buffer (3584) vs BF16
   weight (7168) crash. The sibling **`deepseek_r1-torch-fp4`** loads cleanly and
   is the default. (Use `-v2` only with a trtllm build that handles it - its
   model card targets TRT-LLM `main` on 8xB200.)
2. **Image:** use **`nvcr.io#nvidia/tensorrt-llm/release:1.3.0rc14`** (has
   `trtllm-serve` + `trtllm-llmapi-launch`). `ai-dynamo:0.8.1.post2` hits the
   same loader issue.
3. **Multi-GPU launch:** `srun --ntasks=4 --ntasks-per-node=4 --mpi=pmix ... trtllm-llmapi-launch trtllm-serve ...`.
   Plain `trtllm-serve` under `srun` dies with `MPI_ERR_SPAWN`.
4. **Whole-node alloc:** if GPUs aren't a SLURM gres on your cluster, request
   `--exclusive` (one node = 1x GB200x4), not `--gres=gpu:4`. enroot also needs
   `TMPDIR=/tmp`.
5. **Served model id = checkpoint basename** (`deepseek_r1-torch-fp4`), so
   `model_params.name` must be that basename (not the full path).
6. **Cross-arch cluster** (x86 login + aarch64 GB200 compute): the benchmark
   client can't run on the compute node -> use `SERVER_ONLY=1` + `run_client.sh`
   (Workflow B below). The login node _can_ reach the compute node's port.
7. **Remote client:** keep `warmup_connections: 0` and
   `min_required_connections: 0` (the configs do) - auto pre-opens thousands of
   connections at init and blows the worker-init timeout.
8. **Long outputs:** the in-flight drain after a phase is bounded (240 s) - too
   short for reasoning, so the run would score only the fast samples. Set the
   accuracy-phase drain to wait indefinitely via `settings.drain.accuracy_timeout_s: null`
   (per-phase `DrainConfig`). This knob ships in the open PR
   `feat/make-configurable-drain-timeout`; until it merges, the drain is fixed
   at 240 s and a long full run must be split or driven against an already
   warmed-up batch.
9. **LiveCodeBench scoring is separate** (see below). The `deepseek_r1` scorer
   grades the four text subsets in-process, but LCB needs (a) a ~21 GB dataset
   load that the **login node's per-user cgroup OOM-kills**, and (b) a sandbox
   that **kills runaway generated code** (the MLCommons in-process executor
   hangs - `N of M futures unfinished`). Score LCB on a compute node with
   `score_livecodebench.sh`.

## Prerequisites

- Single GB200x4 node via SLURM (`-A <your-account> -p <your-partition>`).
- Pyxis/Enroot.
- Parent env synced: `uv sync --extra dev` from the repo root; `uv` on `PATH`.
- Accuracy subproject set up once (network needed):
  ```bash
  cd examples/10_DeepSeekR1_Example/accuracy && uv sync && bash setup_eval.sh && cd -
  ```

## Prepare the dataset (once)

```bash
# Full set + tiny perf set; add --subset 385 for the representative slice.
python examples/10_DeepSeekR1_Example/prepare_dataset.py --subset 385
# -> data/deepseek_r1_eval.parquet (4388), data/deepseek_r1_eval_subset.parquet (~385),
#    data/deepseek_r1_perf_tiny.parquet (4)
```

(Run in a working pandas/pyarrow env, e.g. the accuracy subproject's `.venv`.)

## Workflow A - single-arch cluster (client co-located on the compute node)

```bash
sbatch examples/10_DeepSeekR1_Example/launch_and_run.sh
# launches trtllm-serve, waits for /health, probes, runs the accuracy benchmark.
```

## Workflow B - heterogeneous cluster (x86 login + aarch64 GB200 compute) (PASS) verified

```bash
# MODEL_DIR points at the DeepSeek-R1 FP4 checkpoint dir; the server serves it and
# the client uses it as the tokenizer for tokens_per_sample (the configs read it
# via ${MODEL_DIR}). Export it for both steps.
export MODEL_DIR=/path/to/deepseek_r1-torch-fp4

# 1. Start the server only (holds the node, writes logs/dsr1_server_ready):
SERVER_ONLY=1 sbatch examples/10_DeepSeekR1_Example/launch_and_run.sh

# 2. Wait until logs/dsr1_server_ready appears (~8 min: image import + weight load).

# 3. Drive the client from the LOGIN node. Quick estimate (~385 samples).
#    For long runs set settings.drain.accuracy_timeout_s: null in the config
#    (needs the feat/make-configurable-drain-timeout PR; see gotcha #8).
BENCH_CONFIG=examples/10_DeepSeekR1_Example/offline_deepseek_r1_accuracy_subset.yaml \
RELEASE_SERVER=1 \
bash examples/10_DeepSeekR1_Example/run_client.sh
#   ...or the full 4388 set (multi-hour; needs ~9 h - likely > the 5 h batch limit
#   on a single node; use 8xB200 / 2 nodes for the official full run):
# BENCH_CONFIG=.../offline_deepseek_r1_accuracy.yaml bash run_client.sh
```

`run_client.sh` takes the server as its first argument (`<host>:<port>`, or a
bare host that defaults to `:8000`); with no argument it reads
`logs/dsr1_server_ready`. It substitutes that endpoint into a rendered copy of
the chosen config (the YAML is not modified), runs the benchmark, prints the
score, and (with `RELEASE_SERVER=1`) releases the node. It requires `MODEL_DIR`
to be exported.

## Workflow C - server already running (just point the client at it)

If a trtllm-serve (or any OpenAI-compatible `/v1/completions` endpoint serving
the model) is **already up** - started outside this example - skip
`launch_and_run.sh` entirely and run only the client. Two equivalent ways:

**(i) `run_client.sh` with an explicit host** (no YAML edit; it renders a copy
with the endpoint substituted):

```bash
export MODEL_DIR=/path/to/deepseek_r1-torch-fp4   # tokenizer for tokens_per_sample
BENCH_CONFIG=examples/10_DeepSeekR1_Example/offline_deepseek_r1_accuracy.yaml \
bash examples/10_DeepSeekR1_Example/run_client.sh my-server-host:8000
```

**(ii) Edit the client's config and run `from-config` yourself** - set
`endpoints:` (and confirm `model_params.name` matches the server's registered
model id) in the chosen YAML, then:

```bash
export MODEL_DIR=/path/to/deepseek_r1-torch-fp4
export DEEPSEEK_EVAL_PROJECT_PATH=examples/10_DeepSeekR1_Example/accuracy  # only if not running from the repo root
inference-endpoint benchmark from-config \
  --config examples/10_DeepSeekR1_Example/offline_deepseek_r1_accuracy.yaml --mode acc
```

Either way the accuracy score is written under `report_dir` (see Results). The
parent env must be synced (`uv sync --extra dev`) and the accuracy subproject set
up once (Prerequisites).

## Results

```bash
cat logs/deepseek_r1_fp4_accuracy_subset/results.json          # accuracy_scores block
cat logs/deepseek_r1_fp4_accuracy_subset/deepseek_eval/*_results.json  # per-subset + complete flag
```

`accuracy_scores["deepseek_r1_accuracy"].score` is the aggregate `exact_match`
(0-100) over the **four text subsets** (math500/aime/gpqa/mmlu_pro); the
`deepseek_r1` scorer reports `complete: false` and `livecodebench: failed`
because LCB code execution can't run in-process (see gotcha #9). Score LCB
separately and fold it in.

## Scoring LiveCodeBench (separate compute-node job)

```bash
# Run on a compute node (clean CPUs + 940GB RAM, hardened kill-on-timeout exec):
sbatch examples/10_DeepSeekR1_Example/score_livecodebench.sh
# -> examples/10_DeepSeekR1_Example/accuracy/lcb_datasets/lcb_results.json
#    {"total_samples": 349, "passed_samples": P, "pass_at_1": P/349}
```

It generates the LCB test cases (release_v6 superset -> all question_ids resolve;
`question_id` = the dataset's `ground_truth`), extracts the ` ```python `
block from each model output, and runs the repo's hardened `lcb_serve`
(`evaluation/livecodebench/`, kill-on-timeout per execution - the same engine
behind the optional Docker WebSocket service, run directly here since this
cluster has no Docker).

**Fold LCB into the official 5-subset number:**

```
full_exact_match = (sum of per-subset correct over math500/aime/gpqa/mmlu_pro
                    + LCB passed_samples) / 4388
```

where each text subset's correct count = `round(exact_match/100 * num_samples)`
from `deepseek_eval/deepseek_r1_accuracy_results.json`.

> Security note: `lcb_serve` executes model-generated code. The repo's
> `evaluation/livecodebench/lcb_serve.dockerfile` provides a hardened Docker
> sandbox; this script runs the same executor directly on an isolated,
> exclusive compute node - acceptable for a controlled in-house model, but use
> the container for untrusted endpoints.

## Throughput note

On a single GB200x4, DeepSeek-R1 reasoning runs at ~8 samples/min once the
server's batch is filled (TTFT ~0.6 s). The ~385-sample subset completes in
under an hour; the full 4388 is many hours (the model card's reference is
8xB200). Tune `trtllm_serve_config.yaml` (`max_batch_size`, `max_num_tokens`,
`kv_cache_config.free_gpu_memory_fraction`) for your throughput target.
