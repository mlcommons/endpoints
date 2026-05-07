# Local Testing Guide

How to run and test the CLI locally using the built-in echo server and the included dummy dataset, without a real inference endpoint.

## Quick Start: Testing CLI with Echo Server

### 1. Prepare Test Environment

**Dataset:** The repo includes `tests/datasets/dummy_1k.jsonl` (1000 samples)
**Format:** Automatically inferred from the file extension. Common local formats include `jsonl`, `json`, `csv`, `parquet`, and HuggingFace datasets.

### 2. Start the Echo Server

The echo server is included for local testing and mirrors requests back as responses.

```bash
# Terminal 1: Start echo server on port 8765
uv run python -m inference_endpoint.testing.echo_server --port 8765

# Or use default port 12345
uv run python -m inference_endpoint.testing.echo_server
```

The server will log:

```
Server ready on port 8765
Server is running. Press Ctrl+C to stop...
```

### 3. Test the Probe Command

```bash
# Terminal 2: Test probe command
uv run inference-endpoint -v probe \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --requests 5

# With custom prompt and model
uv run inference-endpoint -v probe \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --requests 10 \
  --prompt "Tell me a joke in 20 words"
```

**Expected Output:**

```
Probing: http://localhost:8765
Sending 5 requests...
  Issued 1/5 requests
  ...
  Issued 5/5 requests
Waiting for 5 responses...
  Processed 5/5 responses
✓ Completed: 5/5 successful
✓ Avg latency: 184ms
✓ Range: 184ms - 184ms
✓ Sample responses (5 collected):
  [probe-0] Please write me a joke in 30 words.
  [probe-1] Please write me a joke in 30 words.
  ...
✓ Probe successful
```

### 4. Test Benchmark Commands

#### Offline Benchmark (Max Throughput)

```bash
# Quick test (model is required)
uv run inference-endpoint -v benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --duration 0

# Production test with custom params and report generation
uv run inference-endpoint -v benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --num-samples 5000 \
  --workers 4 \
  --report-dir benchmark_report

# Note: Set HF_TOKEN environment variable if using non-public models
# export HF_TOKEN=your_huggingface_token
```

**Expected Output:**

```
Loading: dummy_1k.jsonl
Loaded 1000 samples
Mode: TestMode.PERF, QPS: 10.0, Responses: False
Streaming: disabled (auto, offline mode)
Min Duration: 0.0s, Expected samples: 1000
Scheduler: MaxThroughputScheduler (pattern: max_throughput)
Connecting: http://localhost:8765
Running...
Completed in 0.5s
Results: 1000/1000 successful
Estimated QPS: 2000.0
Cleaning up...
```

#### Online Benchmark (Poisson Distribution)

```bash
# Test sustained QPS with latency focus
uv run inference-endpoint -v benchmark online \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --duration 0 \
  --load-pattern poisson \
  --target-qps 100 \
  --report-dir online_benchmark_report
```

**Expected Output:**

```
Loading: dummy_1k.jsonl
Loaded 1000 samples
Mode: TestMode.PERF, QPS: 100.0, Responses: False
Streaming: enabled (auto, online mode)
Min Duration: 0.0s, Expected samples: 1000
Scheduler: PoissonDistributionScheduler (pattern: poisson)
Connecting: http://localhost:8765
Running...
Completed in 10.0s
Results: 1000/1000 successful
Estimated QPS: 100.0
Cleaning up...
```

### 5. Test Other Commands

```bash
# Show info
uv run inference-endpoint -v info

# Generate template
uv run inference-endpoint init offline

# Validate config
uv run inference-endpoint validate-yaml --config offline_template.yaml

# Test with existing dataset
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/ds_samples.jsonl \
  -v
```

### 6. View Results

A report directory is always created (at `--report-dir` if specified, or at a default path
otherwise), containing benchmark artifacts: `result_summary.json`, `runtime_settings.json`,
`sample_idx_map.json`, `report.txt`, and `events.jsonl`.

### 7. Stop the Echo Server

Press `Ctrl+C` in the terminal running the echo server, or:

```bash
pkill -f echo_server
```

## Echo Server Options

```bash
# Custom host and port
uv run python -m inference_endpoint.testing.echo_server --host 0.0.0.0 --port 9000

# Check help
uv run python -m inference_endpoint.testing.echo_server --help
```

## Request Format

The echo server expects OpenAI-compatible format but simplifies it:

**What workers send (internal):**

```json
{
  "prompt": "Your query text",
  "model": "model-name",
  "max_completion_tokens": 50,
  "stream": false
}
```

The HTTP client's OpenAI adapter converts this to proper OpenAI format with `messages` array internally.

## Troubleshooting

### Connection Refused

```
Error: Connection failed
```

**Solution:** Ensure echo server is running and port is correct

### Validation Errors

```
Error: prompt not found in query.data
```

**Solution:** Use `"prompt"` format in Query data, not `"messages"` (client converts it)

### Probe Times Out

```
Error: Timeout (>60s)
```

**Solution:** Echo server might not be running, check logs at `/tmp/echo_server.log`

## Complete Testing Workflow

### Full Benchmark Test

```bash
# 1. Start echo server
uv run python -m inference_endpoint.testing.echo_server --port 8000 &

# 2. Generate fresh dataset if needed
uv run python scripts/create_dummy_dataset.py

# 3. Set HF_TOKEN if using non-public models (optional)
export HF_TOKEN=your_huggingface_token

# 4. Test probe first
uv run inference-endpoint probe --endpoints http://localhost:8000 --model Qwen/Qwen3-8B --requests 10

# 5. Run benchmark with report generation
uv run inference-endpoint -v benchmark offline \
  --endpoints http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --workers 4 \
  --report-dir benchmark_report

# 6. Stop server
pkill -f echo_server
```

### Testing Different Modes

```bash
# Offline (max throughput)
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --report-dir offline_report

# Online (Poisson distribution)
uv run inference-endpoint benchmark online \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --load-pattern poisson \
  --target-qps 500 \
  --report-dir online_report

# With explicit sample count
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --num-samples 500

# Force streaming on for offline mode (to test TTFT metrics)
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --streaming on

# Concurrency mode (fixed concurrent requests)
uv run inference-endpoint benchmark online \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --load-pattern concurrency \
  --concurrency 32
```

## Tips

**Key Requirements:**

- Model name is **required** for all benchmark and probe commands
- Online mode requires `--load-pattern` to specify the scheduler type (poisson or concurrency)
  - `--load-pattern poisson` requires `--target-qps`
  - `--load-pattern concurrency` requires `--concurrency`
- Set `HF_TOKEN` environment variable for non-public models (public models like Qwen/Qwen3-8B don't need it)

**Sample Count Control:**

- Use `--duration 0` when you want a local test to stop after exhausting the dataset instead of running for the default timed duration
- Sample priority: `--num-samples` > dataset size (when `--duration 0`) > calculated (target_qps × duration)
- Default duration: 600000ms (10 minutes)

**Testing & Debugging:**

- Use `-v` for INFO logging, `-vv` for DEBUG
- Echo server mirrors prompts back - perfect for quick testing without real inference
- Press `Ctrl+C` to gracefully interrupt benchmarks
- Default test dataset: `tests/datasets/dummy_1k.jsonl` (1000 samples)

**Advanced:**

- Streaming: `auto` (default), `on`, or `off` - auto enables for online, disables for offline
- Use `--report-dir` for detailed metrics reports with TTFT, TPOT, and token analysis
- Dataset format auto-inferred from file extension
