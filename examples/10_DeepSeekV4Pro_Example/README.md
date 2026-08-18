# DeepSeek-V4-Pro Benchmark (SGLang)

Pass@1 accuracy suite for `deepseek-ai/DeepSeek-V4-Pro` on SGLang (ROCm / MI35x):
**AIME25 (I+II) → GPQA → LiveCodeBench**.

Validated client/server recipe used for the measured scores below:

| Setting | Value |
| ------- | ----- |
| Client concurrency | `target_concurrency: 64` / `num_workers: 64` |
| `max_new_tokens` | `256000` |
| Server `CONC` | `64` (`--max-running-requests` / cuda-graph max bs) |
| TP | `8` |
| Image | `rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs` |
| API | `/v1/chat/completions` (`api_type: openai`), streaming on |
| Chat template kwargs | `thinking: true`, `reasoning_effort: max` |

### Measured pass@1 (this recipe)

| Dataset | pass@1 | Samples |
| ------- | ------ | ------- |
| AIME25 (I+II) | 100.00% | 30/30 |
| GPQA | 90.91% | 198/198 |
| LiveCodeBench | 94.22% | 994/1055 |

## Layout (files needed for this suite)

```text
examples/10_DeepSeekV4Pro_Example/
├── start_sglang_server.sh              # launch SGLang (host or Docker)
├── docker_common.sh                    # shared docker/log helpers
├── chat_templates/deepseek_v4_thinking.jinja
├── data/perf_stub.jsonl                # tiny performance stub (required by online mode)
├── sglang_deepseek_v4_pro_aime_pass1.yaml
├── sglang_deepseek_v4_pro_gpqa_pass1.yaml
├── sglang_deepseek_v4_pro_lcb_pass1.yaml
├── run_sglang_accuracy_benchmark.sh    # AIME → GPQA → LCB
├── start_lcb_service.sh                # optional containerized LCB scorer
└── README.md
```

Accuracy datasets download from HuggingFace (`aime25::deepseek_v4`, `gpqa::deepseek_v4`,
`livecodebench::deepseek_v4`). GPQA is gated — set `HF_TOKEN`.

## Environment

```bash
export HF_HOME=<writable HF cache, e.g. $PWD/results/hf_cache>
export HF_TOKEN=<your HuggingFace token>   # required for GPQA
export ALLOW_LCB_LOCAL_EVAL=true           # default for LCB local scoring

# Model path used by the pass@1 YAMLs / tokenizer metrics:
export MODEL_PATH=/data/workloads-inference/hf_hub_cache/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1
export TOKENIZER_MODEL_PATH=${MODEL_PATH}
```

If your checkout lives under a different HF cache root, update `model_params.name` in the
three `*_pass1.yaml` files (and `TOKENIZER_MODEL_PATH`) to match `GET /v1/models`.

## Launch server (CONC=64)

```bash
export MODEL_PATH=/data/workloads-inference/hf_hub_cache/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1
export HF_CACHE_ROOT=/data/workloads-inference/hf_hub_cache   # so snapshot→blob symlinks resolve in Docker
export RUN_MODE=docker
export SGLANG_IMAGE=rocm/mlperf-inference:v0.5.16-rocm720-mi35x-20260803_prs
export SGLANG_PORT=30000
export TP=8
export CONC=64
export ISL=8192
export MAX_MODEL_LEN=327680
./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
```

On a host with the `_prs` SGLang build installed, omit `RUN_MODE=docker`.

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `CONC` | `512` in script; **use `64` for this suite** | `--max-running-requests` / `--cuda-graph-max-bs` |
| `TP` | `8` | Tensor parallel size |
| `ISL` | `8192` | Chunked-prefill size |
| `MAX_MODEL_LEN` | `327680` | `--context-length` (≥ 256k generations) |
| `HF_CACHE_ROOT` | _(unset)_ | Mount HF cache repo root so snapshot blob symlinks work |
| `SGLANG_IMAGE` | `…20260803_prs` | Docker image when `RUN_MODE=docker` |

Read-only HF snapshots: the launcher patches `model_type` to `deepseek_v3` when writable,
otherwise passes `--json-model-override-args '{"model_type":"deepseek_v3"}'`.

Verify:

```bash
curl -sf http://127.0.0.1:30000/health || curl -sf http://127.0.0.1:30000/v1/models
```

## Run pass@1 suite

Full suite (AIME → GPQA → LCB):

```bash
export HF_TOKEN=<token>
export ALLOW_LCB_LOCAL_EVAL=true
export WAIT_FOR_SGLANG_S=120
./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

Or one dataset at a time:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_aime_pass1.yaml \
  --timeout 1209600 --mode both

uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_gpqa_pass1.yaml \
  --timeout 1209600 --mode both

uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_lcb_pass1.yaml \
  --timeout 1209600 --mode both
```

Scores land under:

```text
results/sglang_deepseek_v4_pro_aime_pass1/accuracy/accuracy_results.json
results/sglang_deepseek_v4_pro_gpqa_pass1/accuracy/accuracy_results.json
results/sglang_deepseek_v4_pro_lcb_pass1/accuracy/accuracy_results.json
```

## LiveCodeBench scoring

**Local (default for this suite):** `ALLOW_LCB_LOCAL_EVAL=true` runs the LCB scorer as a
subprocess. Ensure test cases exist under `dataset_cache/livecodebench/release_v6/` (the
accuracy helper regenerates them if missing).

**Optional container:** `./start_lcb_service.sh` (needs `docker login dhi.io`). WebSocket
scoring is preferred when the service is up; otherwise the client falls back to local eval.

## Troubleshooting

**Cannot connect to SGLang**

- `curl http://127.0.0.1:30000/health` / `/v1/models`
- Confirm `CONC=64` server is the process bound to port 30000

**`address already in use` on `--port`**

- Do not leave `SGLANG_PORT` set in the SGLang process environment (ZMQ conflict).
  `start_sglang_server.sh` passes `--port` and runs with `env -u SGLANG_PORT`.

**Model load / unknown architecture**

- Expect `model_type` override to `deepseek_v3` (in-place patch or JSON override)
- Use the `_prs` image so baked DSv4 patches are present

**GPQA download fails**

- Export a valid `HF_TOKEN` (gated dataset)

**OOM / SIGABRT at high concurrency**

- Keep server `CONC` and client `target_concurrency` at **64** for 256k-token runs
- Higher concurrency has aborted the dsv4 scheduler when the KV pool saturates

**Snapshot symlinks broken in Docker**

- Set `HF_CACHE_ROOT` to the HF cache root that contains `models--deepseek-ai--DeepSeek-V4-Pro`
