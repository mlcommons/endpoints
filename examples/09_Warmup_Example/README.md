# Warmup Example

This example demonstrates the **warmup phase** feature using
[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
a small 0.5B parameter model that is easy to run locally.

The warmup phase issues randomly generated requests to the endpoint before the timed
performance window begins.

## What warmup does

Before the benchmark clock starts, the warmup phase sends a configurable number of
requests using randomly generated token sequences. This primes the endpoint by:

- Establishing and reusing TCP connections
- Filling KV caches to steady-state occupancy
- Triggering JIT compilation / CUDA graph capture in the inference runtime

Warmup samples are **excluded from all reported metrics** — they complete before
`TEST_STARTED` is recorded, so they do not affect throughput, latency, TTFT, or TPOT.

## Warmup configuration

Add a `warmup` block to any YAML config:

```yaml
warmup:
  num_samples: 64 # number of warmup requests to issue
  input_seq_length: 256 # ISL: target input token count
  output_seq_length: 64 # OSL: max_new_tokens for warmup requests
  range_ratio: 0.9 # ISL variance: generates ISL in [256*0.9, 256]
  random_seed: 42
```

No real dataset is needed for warmup — sequences are generated at runtime from random
token IDs using the model's own tokenizer.

## Quick test with the echo server

The built-in echo server lets you verify the warmup flow locally without a GPU.

```bash
# Terminal 1 — start the echo server
python -m inference_endpoint.testing.echo_server --port 8000

# Terminal 2 — run offline benchmark with warmup
inference-endpoint benchmark from-config -c examples/09_Warmup_Example/warmup_offline.yaml
```

The log output will show the warmup phase completing before the performance run starts:

```
INFO  Warmup dataset ready: 64 samples (ISL=256, OSL=64)
INFO  Warmup: issuing samples...
INFO  Warmup samples issued, waiting for responses to drain...
INFO  Warmup complete
INFO  Running...
```

## Running against a real endpoint

### Prerequisites

```bash
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<path to your HuggingFace cache, e.g. ~/.cache/huggingface>
```

Download the model before launching so vLLM can reuse the local cache:

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

### Launch a vLLM server

The `--trust-request-chat-template` flag is required because the CNN DailyMail dataset
sends requests with a custom chat template.

```bash
docker run --runtime nvidia --gpus all \
  -v ${HF_HOME}:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  -p 8000:8000 --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trust-request-chat-template
```

### Offline benchmark with warmup

```bash
inference-endpoint benchmark from-config -c examples/09_Warmup_Example/warmup_offline.yaml
```

### Online benchmark with warmup

```bash
inference-endpoint benchmark from-config -c examples/09_Warmup_Example/warmup_online.yaml
```

## Tuning warmup parameters

| Parameter           | Guidance                                                                |
| ------------------- | ----------------------------------------------------------------------- |
| `num_samples`       | Use enough to saturate the KV cache; 32–128 is typical for small models |
| `input_seq_length`  | Match the ISL distribution of your real workload                        |
| `output_seq_length` | Match the OSL distribution; lower values make warmup finish faster      |
| `range_ratio`       | `1.0` = fixed ISL; `0.8`–`0.9` adds light variance for broader coverage |
| `random_seed`       | Change to vary which token sequences are generated                      |
