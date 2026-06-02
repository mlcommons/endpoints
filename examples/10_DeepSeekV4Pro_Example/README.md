# DeepSeek-V4-Pro Benchmark (vLLM)

End-to-end example for benchmarking `deepseek-ai/DeepSeek-V4-Pro` with vLLM, using the same
performance and accuracy datasets as the [GPT-OSS-120B example](../04_GPTOSS120B_Example/Readme.md)
(AIME25, GPQA, LiveCodeBench).

Unlike GPT-OSS-120B, DeepSeek-V4-Pro uses vLLM's native chat template and reasoning parser.
Requests go to `/v1/chat/completions` (`api_type: openai`) with text prompts from the dataset.

## Getting the Dataset

The performance dataset must be obtained from the LLM task-force (parquet format). Place it at:

```
examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet
```

The accuracy datasets (AIME25, GPQA, LiveCodeBench) are downloaded automatically from HuggingFace.

## Environment Setup

```bash
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
export HF_TOKEN=<your HuggingFace token>  # required for GPQA and faster HF downloads
export MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
export MODEL_DIR=/data/workloads-inference/models
export MODEL_PATH=${MODEL_DIR}/deepseek-ai/DeepSeek-V4-Pro
export TOKENIZER_MODEL_PATH=${MODEL_PATH}  # host path for ISL/OSL/TPOT metrics
```

## Download Model

Download weights to the shared model store and mount them into the vLLM container:

```bash
mkdir -p "${MODEL_PATH}"
hf download "${MODEL_NAME}" --local-dir "${MODEL_PATH}"
```

The container sees the model at `/models/deepseek-ai/DeepSeek-V4-Pro`, which matches
`model_params.name` in the same way as GPT-OSS example YAML configs.

## Launch vLLM Server

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

## LiveCodeBench Setup

LiveCodeBench has dependency conflicts with the main package and must be run via the
containerized workflow (same as GPT-OSS). Follow the instructions in the
[LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container).

**Prerequisite:** authenticate with Docker Hardened Images before building:

```bash
docker login dhi.io
```

Then start the service (helper script):

```bash
./examples/10_DeepSeekV4Pro_Example/start_lcb_service.sh
```

Verify:

```bash
curl http://127.0.0.1:13835/info
```

## Run Benchmark

Same workflow as GPT-OSS: start vLLM, start `lcb-service`, then run from YAML config.

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

Performance-only run:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/10_DeepSeekV4Pro_Example/vllm_deepseek_v4_pro_perf.yaml \
  --timeout 60
```

### Config notes

- **Performance dataset**: Uses `text_input` → `prompt` remapping. vLLM applies the DeepSeek-V4
  chat template server-side (contrast with GPT-OSS, which sends pre-tokenized IDs via
  `openai_completions`).
- **Accuracy datasets**: Use the `::deepseek_v4` preset (same prompt formatting as GPT-OSS).
- **Reasoning output**: vLLM streams reasoning via `--reasoning-parser deepseek_v4`; the client
  accumulates both reasoning and final content for scoring.

---

## Re-score from Existing Report

If inference completed but scoring failed (e.g. `lcb-service` was not running), re-score from
`events.jsonl` without re-running inference:

```bash
cd /home/karverma/endpoints
uv run python examples/10_DeepSeekV4Pro_Example/rescore_accuracy.py \
  --report-dir results/vllm_deepseek_v4_pro_accuracy \
  --write-results-json
```

Skip LiveCodeBench until the container is ready:

```bash
uv run python examples/10_DeepSeekV4Pro_Example/rescore_accuracy.py \
  --report-dir results/vllm_deepseek_v4_pro_accuracy \
  --skip-lcb
```

---

## Troubleshooting

**Cannot connect to vLLM server**

- Verify it is running: `curl http://localhost:8000/health`
- Ensure `model_params.name` in the YAML matches the model path passed to vLLM

**LiveCodeBench scoring fails / Connection refused on port 13835**

- Start `lcb-service` before running the benchmark (see [LiveCodeBench Setup](#livecodebench-setup))
- Do not use `ALLOW_LCB_LOCAL_EVAL` unless following the non-containerized fallback in the GPT-OSS readme

**CUDA out of memory**

- Increase `--tensor-parallel-size`
- Lower `--gpu-memory-utilization` or `--max-model-len`

**Model not found in container**

- Confirm the host path exists: `ls "${MODEL_PATH}"`
- Confirm the mount: `-v "${MODEL_PATH}:/models/deepseek-ai/DeepSeek-V4-Pro:ro"`

**docker build fails with `dhi.io ... unauthorized`**

- Run `docker login dhi.io` with your Docker Hub credentials (PAT with read access to hardened images)
