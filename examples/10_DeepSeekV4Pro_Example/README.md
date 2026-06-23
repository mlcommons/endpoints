# DeepSeek-V4-Pro Benchmark (vLLM / SGLang)

End-to-end example for benchmarking `deepseek-ai/DeepSeek-V4-Pro` with vLLM or SGLang, using the same
performance and accuracy datasets as the [GPT-OSS-120B example](../04_GPTOSS120B_Example/Readme.md)
(AIME25, GPQA, LiveCodeBench).

Both backends use `/v1/chat/completions` (`api_type: openai`) with text prompts from the dataset.
The server applies the DeepSeek-V4 chat template and reasoning parser.

## Getting the Dataset

The performance dataset must be obtained from the LLM task-force (parquet format). Place it at:

```
examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet
```

The accuracy datasets (AIME25, GPQA, LiveCodeBench) are downloaded automatically from HuggingFace.

## Environment Setup

```bash
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
export HF_TOKEN=<your HuggingFace token>  # required for GPQA (gated) and faster HF downloads
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
export MODEL_DIR=/data/workloads-inference/models
export MODEL_PATH=${MODEL_DIR}/deepseek-ai/DeepSeek-V4-Pro
export TOKENIZER_MODEL_PATH=${MODEL_PATH}  # host path for ISL/OSL/TPOT metrics
```

Preflight scripts (`run_benchmark.sh`, `run_sglang_benchmark.sh`, `run_*_accuracy_benchmark.sh`) probe the inference server with `GET /health` and `GET /v1/models`. Override the base URL or wait time while a server is starting:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000      # default when VLLM_PORT=8000
export WAIT_FOR_VLLM_S=120                       # seconds; 0 = single attempt

export SGLANG_BASE_URL=http://127.0.0.1:30000    # default when SGLANG_PORT=30000
export WAIT_FOR_SGLANG_S=120
```

## Download Model

Download weights to the shared model store and mount them into the serving container:

```bash
mkdir -p "${MODEL_PATH}"
hf download "${MODEL_NAME}" --local-dir "${MODEL_PATH}"
```

---

## vLLM

### Launch Server

DeepSeek-V4-Pro requires multiple GPUs. Adjust `--tensor-parallel-size` to match your hardware.

```bash
docker run --runtime nvidia --gpus all \
  -v "${MODEL_PATH}:/models/deepseek-ai/DeepSeek-V4-Pro:ro" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai-rocm:v0.22.0 \
  /models/deepseek-ai/DeepSeek-V4-Pro \
  --tensor-parallel-size 8 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.8 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --moe-backend aiter \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

### Run Benchmark (vLLM)

[`vllm_deepseek_v4_pro_example.yaml`](vllm_deepseek_v4_pro_example.yaml) runs performance +
AIME25 + GPQA + LiveCodeBench accuracy at concurrency 512:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/vllm_deepseek_v4_pro_example.yaml \
  --timeout 60
```

Or use the helper script (checks vLLM + lcb-service first):

```bash
./examples/10_DeepSeekV4Pro_Example/run_benchmark.sh
```

Performance-only:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/vllm_deepseek_v4_pro_perf.yaml \
  --timeout 60
```

### Run Accuracy (vLLM)

Same workflow as SGLang: start vLLM, set `HF_TOKEN` (required for gated GPQA), then run the
accuracy helper (checks prerequisites, tees logs under `results/docker_logs/accuracy/`):

```bash
export HF_TOKEN=<your HuggingFace token>
export HF_HOME=~/.cache/huggingface

# Start vLLM (see Launch Server above), then:
./examples/10_DeepSeekV4Pro_Example/run_vllm_accuracy_benchmark.sh
```

YAML-only (equivalent):

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/vllm_deepseek_v4_pro_accuracy.yaml \
  --timeout 3600
```

Python script (legacy `run_accuracy.py` path):

```bash
USE_PYTHON_SCRIPT=true ./examples/10_DeepSeekV4Pro_Example/run_vllm_accuracy_benchmark.sh
```

| Argument / env         | Default      | Description                                              |
| ---------------------- | ------------ | -------------------------------------------------------- |
| `HF_TOKEN`             | _(required)_ | HuggingFace token for GPQA download                      |
| `VLLM_PORT`            | `8000`       | vLLM HTTP port                                           |
| `TIMEOUT`              | `3600`       | Benchmark timeout (seconds)                              |
| `ALLOW_LCB_LOCAL_EVAL` | `true`       | Subprocess LCB scoring when `lcb-service` is unavailable |
| `USE_PYTHON_SCRIPT`    | `false`      | Use `run_accuracy.py` instead of YAML                    |

Accuracy config uses `max_new_tokens: 88000`, concurrency `num_workers: 64`, and phase order
AIME25 ×8 → GPQA ×5 → LiveCodeBench ×3 so math scores are recorded before the LCB phase.

---

## SGLang (ROCm / MI35x)

SGLang serves DeepSeek-V4-Pro on ROCm using the DSv4-tuned image and FP4-experts env flags from the
`amd/deepseek_v4` branch. Before launch, patch `config.json` so `model_type` is `deepseek_v3`
(SGLang registry compatibility).

### Launch Server

**Option A: helper script (host or container)**

On a ROCm host with SGLang installed:

```bash
export MODEL_PATH=/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro
export SGLANG_PORT=30000
export TP=8
export CONC=512
./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
```

Inside the DSv4 Docker image (model directory must exist on the host):

```bash
export MODEL_PATH=/data/workloads-inference/models/deepseek-ai/DeepSeek-V4-Pro
export RUN_MODE=docker
export SGLANG_IMAGE=rocm/sgl-dev:rocm720-mi35x-f96ac98-20260526-DSv4
./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
```

Optional overrides:

| Variable                    | Default   | Description                                                                                                                   |
| --------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_PORT` / `HTTP_PORT` | `30000`   | HTTP listen port (`start_sglang_server.sh` unsets `SGLANG_PORT` before launch — SGLang uses that name for internal ZMQ ports) |
| `TP`                        | `8`       | Tensor parallel size                                                                                                          |
| `CONC`                      | `512`     | `--max-running-requests`                                                                                                      |
| `MAX_MODEL_LEN`             | `98304`   | `--context-length`                                                                                                            |
| `DP_ATTENTION`              | `false`   | Set `true` to enable DP attention                                                                                             |
| `EP_SIZE`                   | `1`       | Expert parallel size (`>1` adds `--ep-size`)                                                                                  |
| `CHAT_TEMPLATE`             | _(unset)_ | Optional path to `deepseek_v4_thinking.jinja`                                                                                 |

The script exports the FP4-experts ROCm flags (`SGLANG_DSV4_FP4_EXPERTS=True`,
`SGLANG_FORCE_TRITON_MOE_FP8=0`, `SGLANG_REASONING_EFFORT=max`, etc.) and launches:

```text
python3 -m sglang.launch_server \
  --model-path $MODEL \
  --tensor-parallel-size $TP \
  --attention-backend compressed \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  ...
```

**Option B: manual launch** (same flags as `start_sglang_server.sh`)

```bash
# Patch HF config (once per cache checkout)
python3 <<'PYEOF'
import json
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="deepseek-ai/DeepSeek-V4-Pro", filename="config.json")
with open(path) as f:
    config = json.load(f)
if config.get("model_type") == "deepseek_v4":
    config["model_type"] = "deepseek_v3"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
PYEOF

# Export env block from start_sglang_server.sh, then:
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 30000 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend compressed \
  --max-running-requests 512 \
  --mem-fraction-static 0.90 \
  --swa-full-tokens-ratio 0.15 \
  --page-size 256 \
  --context-length 98304 \
  --chunked-prefill-size 8192 \
  --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --watchdog-timeout 1800
```

Verify:

```bash
curl http://127.0.0.1:30000/health
```

### Run Benchmark (SGLang)

[`sglang_deepseek_v4_pro_example.yaml`](sglang_deepseek_v4_pro_example.yaml) targets
`http://localhost:30000` with the same datasets and concurrency as the vLLM config:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_example.yaml \
  --timeout 60
```

Or use the helper script:

```bash
./examples/10_DeepSeekV4Pro_Example/run_sglang_benchmark.sh
```

Performance-only:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_perf.yaml \
  --timeout 60
```

### Run Accuracy (SGLang)

Same workflow as [GPT-OSS `run.py`](../04_GPTOSS120B_Example/Readme.md#accuracy-suite-script):
start SGLang, start `lcb-service`, set `HF_TOKEN` (required for gated GPQA), then run the
accuracy helper (checks all prerequisites, tees logs under `results/docker_logs/accuracy/`):

```bash
export HF_TOKEN=<your HuggingFace token>
export HF_HOME=~/.cache/huggingface

./examples/10_DeepSeekV4Pro_Example/start_sglang_server.sh
./examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh
./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

YAML-only (equivalent):

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/sglang_deepseek_v4_pro_accuracy.yaml \
  --timeout 3600
```

Python script (GPT-OSS `run.py` style):

```bash
USE_PYTHON_SCRIPT=true ./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

| Argument / env          | Default      | Description                                  |
| ----------------------- | ------------ | -------------------------------------------- |
| `HF_TOKEN`              | _(required)_ | HuggingFace token for GPQA download          |
| `TIMEOUT`               | `3600`       | Benchmark timeout (seconds)                  |
| `USE_PYTHON_SCRIPT`     | `false`      | Use `run_accuracy_sglang.py` instead of YAML |
| `DOCKER_LOG_STORAGE_GB` | `16`         | Container writable layer size when supported |

Accuracy config uses `max_new_tokens: 88000`, concurrency `num_workers: 64`, and phase order
AIME25 ×8 → GPQA ×5 → LiveCodeBench ×3.

### Docker log storage

Server and LCB containers mount `results/docker_logs/<service>/` on the host at `/workspace`.
Scripts also pass `--storage-opt size=16G` on `docker run` when the daemon uses overlay2
(override with `DOCKER_LOG_STORAGE_GB`). SGLang server stdout is written to
`results/docker_logs/sglang/server.log`.

### Config notes (both backends)

- **Performance dataset**: `text_input` → `prompt`. Server applies DeepSeek-V4 chat template.
- **Accuracy datasets**: `::deepseek_v4` presets (same prompt formatting as GPT-OSS).
- **Reasoning output**: server streams reasoning separately; client accumulates `reasoning_content`
  and final `content` for scoring.
- **SGLang `model_params.name`**: use the HuggingFace id (`deepseek-ai/DeepSeek-V4-Pro`). vLLM YAML
  uses the container mount path (`/models/deepseek-ai/DeepSeek-V4-Pro`). Set `TOKENIZER_MODEL_PATH`
  to the host weights path for ISL/OSL/TPOT in both cases.

---

## LiveCodeBench Setup

LiveCodeBench has dependency conflicts with the main package. Two options:

### Option A: containerized scorer (recommended when `dhi.io` access is available)

Follow the [LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container).
Requires `docker login dhi.io`, then:

```bash
./examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh
curl http://127.0.0.1:13835/info
```

### Option B: local subprocess scorer (no `docker login dhi.io`)

Same fallback as the GPT-OSS example — runs `lcb_serve` as a subprocess on the host:

```bash
export ALLOW_LCB_LOCAL_EVAL=true
./examples/10_DeepSeekV4Pro_Example/run_vllm_accuracy_benchmark.sh
# or
./examples/10_DeepSeekV4Pro_Example/run_sglang_accuracy_benchmark.sh
```

Both accuracy helpers skip the `:13835` preflight when this is set (default `true`).
WebSocket scoring is attempted first if `lcb-service` is up; otherwise scoring falls
back to the subprocess path automatically.

---

## Re-score from Existing Report

If inference completed but scoring failed (e.g. `lcb-service` was not running), re-score from
`events.jsonl` without re-running inference:

```bash
cd endpoints
uv run python examples/10_DeepSeekV4Pro_Example/rescore_accuracy.py \
  --report-dir results/vllm_deepseek_v4_pro_accuracy \
  --write-results-json
# or results/sglang_deepseek_v4_pro_accuracy for SGLang runs
```

Skip LiveCodeBench until the container is ready:

```bash
uv run python examples/10_DeepSeekV4Pro_Example/rescore_accuracy.py \
  --report-dir results/sglang_deepseek_v4_pro_accuracy \
  --skip-lcb
```

---

## Troubleshooting

**Cannot connect to vLLM server**

- Verify: `curl http://localhost:8000/health`
- Ensure `model_params.name` in the vLLM YAML matches the model path passed to vLLM

**Cannot connect to SGLang server**

- Verify: `curl http://localhost:30000/health`
- Confirm `SGLANG_PORT` matches `endpoint_config.endpoints` in the SGLang YAML
- For ROCm, use the DSv4 image: `rocm/sgl-dev:rocm720-mi35x-f96ac98-20260526-DSv4`

**SGLang `address already in use` on `--port`**

- Do not export `SGLANG_PORT` into the SGLang process (upstream reserves it for ZMQ).
  Use `start_sglang_server.sh`, which passes `--port` on the CLI and runs with
  `env -u SGLANG_PORT`.

**SGLang fails to load model / unknown architecture**

- Run the `config.json` patch (`model_type`: `deepseek_v4` → `deepseek_v3`) before launch
- Confirm `SGLANG_DSV4_FP4_EXPERTS=True` for the original HF checkpoint with MXFP4 experts

**LiveCodeBench scoring fails / Connection refused on port 13835**

- Start `lcb-service` (`docker login dhi.io` + `start_lcb_service.sh`), or
- `export ALLOW_LCB_LOCAL_EVAL=true` and re-run (see LiveCodeBench Setup above)

**Out of memory**

- Increase `TP` / `--tensor-parallel-size`
- Lower `--mem-fraction-static` (SGLang) or `--gpu-memory-utilization` (vLLM)

**Model not found in container**

- Confirm the host path exists: `ls "${MODEL_PATH}"`
- Mount into the container at the path used by `--model-path`

**docker build fails with `dhi.io ... unauthorized`**

- Run `docker login dhi.io` with your Docker Hub credentials (PAT with read access to hardened images)
