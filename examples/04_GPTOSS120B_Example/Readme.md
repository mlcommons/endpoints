# GPT-OSS-120B Benchmark

End-to-end example for benchmarking `openai/gpt-oss-120b` with vLLM or SGLang, including YAML configs
and Python scripts for AIME25, GPQA, and LiveCodeBench accuracy evaluation.

## Getting the Dataset

The performance dataset must be obtained from the LLM task-force (parquet format, currently being finalized).
Place it at:

```
examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet
```

The accuracy datasets (AIME25, GPQA, LiveCodeBench) are downloaded automatically from HuggingFace.

## Environment Setup

```bash
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
export HF_TOKEN=<your HuggingFace token>
export MODEL_NAME=openai/gpt-oss-120b
```

---

## vLLM

### Launch Server

GPT-OSS-120B requires multiple GPUs. Adjust `--tensor-parallel-size` to match your hardware.

```bash
docker run --runtime nvidia --gpus all \
  -v ${HF_HOME}:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model ${MODEL_NAME} \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 65536
```

### Run Benchmark

[`vllm_gptoss_120b_example.yaml`](vllm_gptoss_120b_example.yaml) runs performance + AIME25 + GPQA accuracy at concurrency 512:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/04_GPTOSS120B_Example/vllm_gptoss_120b_example.yaml \
  --timeout 60
```

> **Note:** The dataset's `prompt` column is mapped to the benchmark's `prompt` field and sent through the
> chat completions API. vLLM does not support pre-tokenized input via this endpoint, unlike SGLang's
> `input_tokens` path.

### vllm bench serve (Reference Comparison)

`vllm bench serve` supports custom datasets only in `jsonl` format. To convert the parquet file:

```python
import pandas as pd

df = pd.read_parquet('examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet')
df = df.rename(columns={'prompt': 'raw_prompt', 'text_input': 'prompt'})
df.to_json('examples/04_GPTOSS120B_Example/data/perf_eval_ref.jsonl', orient='records', lines=True)
```

```bash
vllm bench serve \
  --backend vllm \
  --model ${MODEL_NAME} \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path examples/04_GPTOSS120B_Example/data/perf_eval_ref.jsonl \
  --custom-output-len 2000 \
  --num-prompts 6396 \
  --max-concurrency 512 \
  --save-result \
  --save-detailed
```

Numbers are not directly comparable to `inference-endpoint` results but provide a reference for relative
performance given the output token distribution.

---

## SGLang

### Launch Server

**Option A: MLCommons MLPerf Inference Reference Implementation**

The official reference provides detailed instructions for model setup, data preparation, and deployment:

```bash
git clone https://github.com/mlcommons/inference.git
cd inference/language/gpt-oss-120b
# Follow the README at:
# https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b
./sglang/run_server.sh \
    --model_path /path/to/gpt-oss-120b/model/ \
    --dp <Number of GPUs> \
    --stream_interval 100
```

**Option B: Direct SGLang Installation**

If you already have the model weights, follow [SGLang's GPT-OSS instructions](https://docs.sglang.io/basic_usage/gpt_oss.html).
The server must run on port 30000.

```bash
docker run --runtime nvidia --gpus all --net host \
  -v ${HF_HOME}:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path ${MODEL_NAME} \
  --host 0.0.0.0 \
  --port 30000 \
  --data-parallel-size=1 \
  --max-running-requests 512 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 16384 \
  --ep-size=1 \
  --enable-metrics \
  --stream-interval 500
```

### Run Benchmark

[`sglang_gptoss_120b_example.yaml`](sglang_gptoss_120b_example.yaml) runs performance + AIME25 + GPQA +
LiveCodeBench accuracy at concurrency 512:

```bash
uv run inference-endpoint benchmark from-config \
  -c examples/04_GPTOSS120B_Example/sglang_gptoss_120b_example.yaml \
  --timeout 60
```

For a performance-only run, use [`gptoss_120b_example.yaml`](gptoss_120b_example.yaml). It is
configured for SGLang on `http://localhost:30000` by default. To target vLLM instead, update
`endpoint_config.endpoints` to your server (e.g. `http://localhost:8000`) **and** change
`endpoint_config.api_type` to `"openai"` so requests route to `/v1/chat/completions` rather than
`/generate`.

### LiveCodeBench Setup

LiveCodeBench has dependency conflicts with the main package and should be run via the containerized
workflow. Follow the instructions in the
[LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container).

**Non-containerized (not recommended):**

```bash
source /path/to/inference-endpoint/venv/bin/activate
pip install datasets==3.6.0
pip install fastapi==0.128.0 uvicorn[standard]==0.40.0
export ALLOW_LCB_LOCAL_EVAL=true
```

With `ALLOW_LCB_LOCAL_EVAL=true`, the `LiveCodeBenchScorer` falls back to running `lcb_serve` as a
subprocess on the host.

---

## Accuracy Suite Script

`run.py` runs all three accuracy benchmarks (GPQA, AIME25, LiveCodeBench) in sequence via SGLang:

```bash
cd examples/04_GPTOSS120B_Example
python run.py \
    --report-dir ./results \
    --num-repeats 1 \
    --min-duration 10 \
    --max-duration 600
```

| Argument             | Default                  | Description                          |
| -------------------- | ------------------------ | ------------------------------------ |
| `--report-dir`       | `sglang_accuracy_report` | Directory to save results            |
| `--num-repeats`      | `1`                      | Repeats per dataset                  |
| `--min-duration`     | `10`                     | Minimum benchmark duration (seconds) |
| `--max-duration`     | `600`                    | Maximum benchmark duration (seconds) |
| `--force-regenerate` | off                      | Force dataset regeneration           |

---

## Individual Evaluation Scripts

Run after `run.py` to re-score from an existing report directory.

### GPQA

```bash
python eval_gpqa.py \
    --dataset-path datasets/gpqa/diamond/gpqa_diamond.parquet \
    --report-dir ./results
```

### AIME25

```bash
python eval_aime.py \
    --dataset-path datasets/aime25/aime25.parquet \
    --report-dir ./results
```

### LiveCodeBench

```bash
python eval_livecodebench.py \
    --dataset-path datasets/livecodebench/release_v6/livecodebench_release_v6.parquet \
    --report-dir ./results \
    --lcb-version release_v6 \
    --timeout 60
```

---

## Debugging

[mitmproxy](https://www.mitmproxy.org/) can inspect HTTP traffic in reverse-proxy mode:

```bash
mitmproxy -p 8001 --mode reverse:http://localhost:8000/
```

Run the server on port 8000 and point the client at 8001. All requests and responses are logged
transparently.

---

## Troubleshooting

**Cannot connect to SGLang server**

- Verify it is running: `curl http://localhost:30000/health`
- Check firewall settings for remote servers
- Ensure the port matches in both server and client configs

**CUDA out of memory**

- Increase `--tensor-parallel-size` / `--data-parallel-size`
- Use `--mem-fraction-static` to reduce static memory allocation
- Check GPU utilization with `nvidia-smi`

**LiveCodeBench dependency conflicts**

- Use the containerized workflow (recommended)
- If running standalone, ensure `datasets==3.6.0` is installed

**Slow inference / benchmark taking too long**

- Check GPU utilization with `nvidia-smi`
- Increase `num_workers` in `run.py` or the YAML `settings.client.num_workers`
- Consider enabling FlashInfer or other SGLang optimizations
