# GPT-OSS-120B Benchmark

## Getting dataset

The dataset can be obtained from the LLM task-force which is in the process of finalizing the contents of the dataset for both performance and accuracy. The dataset is in a parquet format. Place it at:

```
examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet
```

## Environment setup

```bash
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
export HF_TOKEN=<your HuggingFace token>
export MODEL_NAME=openai/gpt-oss-120b
```

## vLLM

### Launch server

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

### Run benchmark

The config [`vllm_gptoss_120b_example.yaml`](vllm_gptoss_120b_example.yaml) runs performance + AIME25 + GPQA accuracy at concurrency 512:

```bash
inference-endpoint benchmark from-config \
  -c examples/04_GPTOSS120B_Example/vllm_gptoss_120b_example.yaml \
  --timeout 60
```

> **Note:** In the YAML config, the dataset's `prompt` column is mapped into the benchmark's expected `prompt` field, which is then sent through the chat completions API. vLLM does not support pre-tokenized input via this endpoint, unlike SGLang's `input_tokens` path.

### vllm bench serve (reference comparison)

`vllm bench serve` supports custom datasets only in `jsonl` format. To convert the parquet file:

```python
import pandas as pd

parquet_file = 'examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet'
json_file = 'examples/04_GPTOSS120B_Example/data/perf_eval_ref.jsonl'

df = pd.read_parquet(parquet_file)
df = df.rename(columns={'prompt': 'raw_prompt'})
df = df.rename(columns={'text_input': 'prompt'})
df.to_json(json_file, orient='records', lines=True)
```

This renames `text_input` to `prompt` as the custom dataloader requires the pre-processed prompt under that name. The benchmarking command must point to the `completions` endpoint (not `chat-completions`) since the prompt is pre-processed. Numbers are not directly comparable to inference-endpoint results, but provide a reference for relative performance given the output token distribution.

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

## SGLang

### Launch server

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

### Run benchmark

The config [`sglang_gptoss_120b_example.yaml`](sglang_gptoss_120b_example.yaml) runs performance + AIME25 + GPQA + LiveCodeBench accuracy at concurrency 512:

```bash
inference-endpoint benchmark from-config \
  -c examples/04_GPTOSS120B_Example/sglang_gptoss_120b_example.yaml \
  --timeout 60
```

For a performance-only run with a simpler config, see [`gptoss_120b_example.yaml`](gptoss_120b_example.yaml). Update `endpoint_config.endpoints` in that file to match your server port (e.g. `http://localhost:8000` for vLLM, `http://localhost:30000` for SGLang).

## Debugging

[mitmproxy](https://www.mitmproxy.org/) can inspect HTTP traffic between the benchmarking client and the server in reverse-proxy mode:

```bash
mitmproxy -p 8001 --mode reverse:http://localhost:8000/
```

This forwards port `8001` to `8000`. Run the server on port `8000` and point the client at port `8001`. All requests and responses are logged transparently.
