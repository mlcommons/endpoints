# Quick Start — Qwen2.5-0.5B

All commands run from the **repository root**.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python examples/08_Qwen2.5-0.5B_Example/prepare_dataset.py
```

## Option A — Automated (vLLM or SGLang)

```bash
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm offline
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm online
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh sglang online
```

## Option B — Manual step-by-step

**1. Start server** (pick one):

```bash
# vLLM
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True -p 8000:8000 --ipc=host \
  --name vllm-qwen -d vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.85

# SGLang
docker run --runtime nvidia --gpus all --net host \
  -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \
  --name sglang-qwen -d lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 30000 --mem-fraction-static 0.9 --attention-backend flashinfer
```

**2. Wait for ready:**

```bash
until curl -sf http://localhost:8000/v1/models > /dev/null; do sleep 5; done   # vLLM
until curl -sf http://localhost:30000/health   > /dev/null; do sleep 5; done   # SGLang
```

**3. Run concurrency sweep:**

```bash
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml   # vLLM
  # or: --config examples/08_Qwen2.5-0.5B_Example/sglang_online_qwen_benchmark.yaml

# Add --verbose to stream output live; add --concurrency / --duration-ms to customize
```

**4. Summarize and plot:**

```bash
python scripts/concurrency_sweep/summarize.py \
  results/qwen_online_benchmark/concurrency_sweep/         # vLLM
  # or: results/qwen_sglang_online_benchmark/concurrency_sweep/
```

Writes `metrics_summary.csv`, `metrics_summary.md`, and `metrics_summary.png`.

**5. Stop server:**

```bash
docker stop vllm-qwen && docker rm vllm-qwen
# or: docker stop sglang-qwen && docker rm sglang-qwen
```

---

For TRT-LLM setup, config customization, and output file locations, see [README.md](README.md).
