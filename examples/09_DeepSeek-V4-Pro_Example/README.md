# DeepSeek-V4-Pro Benchmark

End-to-end example for benchmarking `deepseek-ai/DeepSeek-V4-Pro` with vLLM on 8×B200 (or 8×B300), covering performance throughput and accuracy evaluation (AIME 2025, GPQA, and the full MLPerf Inference accuracy suite).

## Hardware

| Requirement  | Details                                                    |
| ------------ | ---------------------------------------------------------- |
| GPUs         | 8× NVIDIA B200 or B300                                     |
| System RAM   | ≥ 256 GB                                                   |
| Docker image | `vllm/vllm-openai:deepseekv4-cu130`                        |
| Startup time | ~22 minutes (weight loading + TileLang kernel compilation) |

The recipe is taken from the [vLLM DeepSeek V4 blog post](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-24-deepseek-v4.md).

## Environment Setup

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Pro   # local weight directory
export HF_HOME=~/.cache/huggingface
export HF_TOKEN=<your HuggingFace token>
```

## Launching the Server

```bash
bash examples/09_DeepSeek-V4-Pro_Example/launch_server.sh
```

The script mounts `$MODEL_PATH` into the container at `/model`, sets
`VLLM_ENGINE_READY_TIMEOUT_S=3600`, and polls `/health` until the server is ready.

### Why `VLLM_ENGINE_READY_TIMEOUT_S=3600` is required

The default value is 600 s (10 min). Loading DeepSeek-V4-Pro's 64 safetensor shards plus
compiling TileLang kernels (`mhc_pre_big_fuse_tilelang` etc.) across 8 DP workers takes
~22 min on 8×B200. With the default timeout the `ApiServer_0` process raises a `TimeoutError`
and exits — even though all 8 engine workers completed successfully — causing the container to
crash. Setting the timeout to 3600 s avoids this entirely.

### Key launch flags

| Flag                                                                                  | Purpose                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `--data-parallel-size 8`                                                              | Expert parallelism across 8 GPUs (no TP needed for MoE)       |
| `--enable-expert-parallel`                                                            | Required alongside `--data-parallel-size`                     |
| `--kv-cache-dtype fp8`                                                                | Matches DeepSeek V4's hybrid c4a / c128a KV cache design      |
| `--block-size 256`                                                                    | Unified 256-token logical block across all compression layers |
| `--attention_config.use_fp4_indexer_cache=True`                                       | FP4 indexer for ~2x additional KV savings                     |
| `--tokenizer-mode deepseek_v4`                                                        | Required for the V4 chat template                             |
| `--reasoning-parser deepseek_v4`                                                      | Strips `<think>…</think>` into `reasoning_content`            |
| `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` | Enables TileLang kernel fusions                               |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600`                                                    | Prevents premature `ApiServer_0` timeout during startup       |

## Performance Benchmark

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/09_DeepSeek-V4-Pro_Example/vllm_dsv4pro_perf.yaml
```

Config: [`vllm_dsv4pro_perf.yaml`](vllm_dsv4pro_perf.yaml)

- 2-minute minimum run at concurrency 32
- Metrics: throughput, latency, TTFT, TPOT

## Accuracy Benchmark (AIME 2025 + GPQA)

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/09_DeepSeek-V4-Pro_Example/vllm_dsv4pro_accuracy.yaml
```

Config: [`vllm_dsv4pro_accuracy.yaml`](vllm_dsv4pro_accuracy.yaml)

| Dataset      | Samples | Repeats | Extractor              | Scorer      |
| ------------ | ------- | ------- | ---------------------- | ----------- |
| AIME 2025    | 30      | 8       | `boxed_math_extractor` | `pass_at_1` |
| GPQA Diamond | 198     | 5       | `abcd_extractor`       | `pass_at_1` |

### Concurrency note

`target_concurrency: 4` is intentional. With `max_model_len=65536` and `max_new_tokens=32768`,
each in-flight request can occupy up to 32k tokens of KV cache. Four concurrent requests
fit within the fp8 KV cache budget without preemption on 8×B200.

### Thinking mode and `budget_tokens`

The `aime25::gptoss_budget_20k` preset enables DeepSeek's thinking mode
(`chat_template_kwargs: {thinking: True, budget_tokens: 20000}`). Without `budget_tokens`,
the model can spend all 32k tokens in the `<think>` block and return an empty boxed answer —
observed on ~85% of responses in early testing. Setting `budget_tokens=20000` caps the
reasoning phase and forces a final answer.

### Measured results (8×B200, `deepseekv4-cu130`)

| Dataset          | Score                                      |
| ---------------- | ------------------------------------------ |
| AIME 2025 pass@1 | **55.4%** (8 repeats, budget_tokens=20000) |

## MLPerf Inference Accuracy Suite

The MLPerf DeepSeek-R1 accuracy check uses 5 sub-datasets (4388 total samples):

| Sub-dataset   | Samples | Metric              | File                                       |
| ------------- | ------- | ------------------- | ------------------------------------------ |
| AIME 1983     | 932     | exact_match         | `mlperf_deepseek_r1_math_accuracy.parquet` |
| MATH-500      | 499     | exact_match         | `mlperf_deepseek_r1_math_accuracy.parquet` |
| GPQA          | 198     | exact_match         | `mlperf_deepseek_r1_mcq_accuracy.parquet`  |
| MMLU-Pro      | 2410    | exact_match         | extracted by `extract_mlperf_subsets.py`   |
| LiveCodeBench | 349     | code_execute_verify | extracted by `extract_mlperf_subsets.py`   |

**Golden accuracy (fp32):** `exact_match = 81.3582%`, `TOKENS_PER_SAMPLE = 3886.2`
**MLPerf pass threshold:** ≥ 80.52% exact_match (99% of golden), tokens within ±10%

### Step 1 — Extract missing subsets

```bash
uv run python examples/09_DeepSeek-V4-Pro_Example/extract_mlperf_subsets.py
```

This writes:

- `datasets/deepseek/mlperf_deepseek_r1_mmlu_pro_accuracy.parquet`
- `datasets/deepseek/mlperf_deepseek_r1_livecodebench_accuracy.parquet`

### Step 2 — Run math + MCQ accuracy

Uncomment MMLU-Pro in [`vllm_dsv4pro_mlperf_accuracy.yaml`](vllm_dsv4pro_mlperf_accuracy.yaml), then:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/09_DeepSeek-V4-Pro_Example/vllm_dsv4pro_mlperf_accuracy.yaml
```

### Step 3 — Run LiveCodeBench accuracy

LiveCodeBench requires the `lcb-service` container (executes generated Python code in an
isolated environment). See the
[LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md) for
container setup. Once running on port 13835, uncomment the `mlperf-livecodebench` dataset in
`vllm_dsv4pro_mlperf_accuracy.yaml` and re-run.

## Troubleshooting

**Container exits immediately or health check never passes**

```bash
docker logs <container_id> | tail -40
```

Common causes:

- `TimeoutError: Timed out waiting for engine core processes to start` — set `VLLM_ENGINE_READY_TIMEOUT_S=3600` (already set in `launch_server.sh`)
- OOM during weight loading — verify `--max-model-len` is not too large for available GPU memory
- `MODEL_PATH` not mounted correctly — check that `/model/config.json` exists inside the container

**`At least one performance dataset required`**

Every benchmark config must include at least one `type: performance` dataset entry, even for
accuracy-only runs. Use the perf-warmup entry with `n_samples_to_issue: 1`.

**Empty boxed answers / low AIME accuracy**

The model exhausted `max_new_tokens` in the thinking phase. Add `budget_tokens` to the preset:

```yaml
- name: aime25::gptoss_budget_20k # uses budget_tokens=20000
```

**`uv: cannot execute binary file: Exec format error`**

The `uv` binary in `~/.local/bin/uv` has the wrong architecture. Use the venv directly:

```bash
.venv/bin/inference-endpoint benchmark from-config -c <config.yaml>
```
