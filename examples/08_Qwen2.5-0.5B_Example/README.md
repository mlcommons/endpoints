# Qwen2.5-0.5B-Instruct Benchmark Example

Benchmarks `Qwen/Qwen2.5-0.5B-Instruct` with offline (max-throughput) and online
(concurrency sweep) load patterns. Designed for small GPUs (8–16 GB VRAM).

Supported inference servers: **vLLM**, **SGLang**, **TRT-LLM**.

---

## Requirements

- Python 3.12+
- Docker with NVIDIA GPU support (`--runtime nvidia`)
- NVIDIA GPU with at least 8 GB VRAM

---

## Step 1 — Install and prepare dataset

Run all commands from the **repository root**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

python examples/08_Qwen2.5-0.5B_Example/prepare_dataset.py
```

This converts `tests/datasets/dummy_1k.pkl` into
`examples/08_Qwen2.5-0.5B_Example/data/test_dataset.pkl`.

---

## Step 2 — Start the inference server

Pick one backend. The server must be fully ready before running benchmarks.

### vLLM (port 8000)

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -p 8000:8000 \
  --ipc=host \
  --name vllm-qwen \
  -d \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-memory-utilization 0.85
```

### SGLang (port 30000)

```bash
docker run --runtime nvidia --gpus all --net host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  --name sglang-qwen \
  -d \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.9 \
  --attention-backend flashinfer
```

### TRT-LLM (port 8000)

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  --name trtllm-qwen \
  -d \
  nvcr.io/nvidia/tritonserver:latest \
  # Add your TRT-LLM engine launch arguments here
```

> **Note:** No pre-built TRT-LLM config is provided. Use
> `examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml` as a template and
> point `endpoint_config.endpoints` at `http://localhost:8000`.

---

## Step 3 — Wait for the server to be ready

Poll until the health endpoint responds:

```bash
# vLLM / TRT-LLM (port 8000)
until curl -sf http://localhost:8000/v1/models > /dev/null; do
  echo "Waiting for server..."; sleep 5
done
echo "Server ready."

# SGLang (port 30000)
until curl -sf http://localhost:30000/health > /dev/null; do
  echo "Waiting for server..."; sleep 5
done
echo "Server ready."
```

---

## Step 4 — Run the concurrency sweep

Choose the config that matches your server. The sweep script overrides
`load_pattern` and `report_dir` for each concurrency level, leaving all other
settings (model, dataset, endpoint) from the config file.

```bash
# vLLM
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml

# SGLang
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/sglang_online_qwen_benchmark.yaml

# TRT-LLM (use the vLLM config or a custom one pointing at port 8000)
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml
```

**Common options:**

| Flag | Default | Description |
|---|---|---|
| `--concurrency N [N ...]` | `1 2 4 8 16 32 64 128 256 512 1024` | Concurrency levels to test |
| `--duration-ms MS` | `600000` (10 min) | Duration per run |
| `--output-dir DIR` | from `report_dir` in config | Root directory for sweep output |
| `--timeout-seconds S` | `720` (12 min) | Per-run subprocess timeout |
| `--verbose` | off | Stream output live to the terminal (useful for debugging) |

Example — quick 3-minute sweep at a few concurrency levels:

```bash
python scripts/concurrency_sweep/run.py \
  --config examples/08_Qwen2.5-0.5B_Example/online_qwen_benchmark.yaml \
  --concurrency 1 4 16 64 \
  --duration-ms 180000 \
  --verbose
```

Results land in subdirectories under the config's `report_dir`:

```
results/qwen_online_benchmark/concurrency_sweep/
  concurrency_1/   benchmark.log   result_summary.json
  concurrency_4/   benchmark.log   result_summary.json
  ...
  summary.json     summary.csv
```

If a run fails, check the per-run log:

```bash
cat results/qwen_online_benchmark/concurrency_sweep/concurrency_64/benchmark.log
```

---

## Step 5 — Summarize results and generate plots

```bash
# vLLM
python scripts/concurrency_sweep/summarize.py \
  results/qwen_online_benchmark/concurrency_sweep/

# SGLang
python scripts/concurrency_sweep/summarize.py \
  results/qwen_sglang_online_benchmark/concurrency_sweep/
```

This prints formatted tables to stdout and writes three files into the sweep
directory:

| File | Contents |
|---|---|
| `metrics_summary.csv` | All metrics in CSV form |
| `metrics_summary.md` | Markdown tables with throughput, latency, TTFT, TPOT |
| `metrics_summary.png` | Line plots of TPS, TTFT P99, and TPOT P50 vs concurrency |

Pass `--no-save` to print tables only without writing files.

---

## Step 6 — Stop the server

```bash
docker stop vllm-qwen    # or sglang-qwen / trtllm-qwen
docker rm   vllm-qwen
```

---

## Offline (max-throughput) benchmark

For a single offline run (no sweep):

```bash
# vLLM
inference-endpoint benchmark from-config \
  -c examples/08_Qwen2.5-0.5B_Example/offline_qwen_benchmark.yaml

# SGLang
inference-endpoint benchmark from-config \
  -c examples/08_Qwen2.5-0.5B_Example/sglang_offline_qwen_benchmark.yaml
```

Results: `results/qwen_offline_benchmark/` or `results/qwen_sglang_offline_benchmark/`.

---

## Automated wrapper

`run_benchmark.sh` automates Steps 2–4 (dataset prep, container start, benchmark):

```bash
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm offline
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh vllm online
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh sglang offline
bash examples/08_Qwen2.5-0.5B_Example/run_benchmark.sh sglang online
```

---

## Config files

| File | Server | Mode |
|---|---|---|
| `offline_qwen_benchmark.yaml` | vLLM (`:8000`) | Offline |
| `online_qwen_benchmark.yaml` | vLLM (`:8000`) | Online sweep |
| `sglang_offline_qwen_benchmark.yaml` | SGLang (`:30000`) | Offline |
| `sglang_online_qwen_benchmark.yaml` | SGLang (`:30000`) | Online sweep |
| `prepare_dataset.py` | — | Converts `dummy_1k.pkl` to example dataset |
| `run_benchmark.sh` | vLLM / SGLang | Automated wrapper |
