# Local Testing Guide

## Quick Start: Testing CLI with Echo Server

### 1. Prepare Test Environment

**Dataset:** The repo includes `tests/datasets/dummy_1k.pkl` (1000 samples, ~133 KB)
**Format:** Automatically inferred (supports: pkl, HuggingFace; coming soon: jsonl)

### 2. Start the Echo Server

The echo server is included for local testing and mirrors requests back as responses.

```bash
# Terminal 1: Start echo server on port 8765
python -m inference_endpoint.testing.echo_server --port 8765

# Or use default port 12345
python -m inference_endpoint.testing.echo_server
```

The server will log:

```
Server ready on port 8765
Server is running. Press Ctrl+C to stop...
```

### 3. Test the Probe Command

```bash
# Terminal 2: Test probe command
inference-endpoint -v probe \
  --endpoint http://localhost:8765 \
  --model gpt-3.5-turbo \
  --requests 5

# With custom prompt and model
inference-endpoint -v probe \
  --endpoint http://localhost:8765 \
  --model llama-2-70b \
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
# Quick test - model is optional (defaults to gpt-3.5-turbo with warning)
inference-endpoint -v benchmark offline \
  --endpoint http://localhost:8765 \
  --dataset tests/datasets/dummy_1k.pkl \
  --duration 5 \
  --workers 2

# Production test with explicit model
inference-endpoint -v benchmark offline \
  --endpoint http://localhost:8765 \
  --model llama-2-70b \
  --dataset tests/datasets/dummy_1k.pkl \
  --duration 30 \
  --workers 4 \
  --concurrency 50 \
  --output benchmark_results.json
```

**Expected Output:**

```
Loading: dummy_1k.pkl
Loaded 1000 samples
Mode: perf, QPS: 10.0, Responses: False
Scheduler: MaxThroughputScheduler (offline burst mode, all queries at t=0)
Connecting: http://localhost:8765
Running...
Completed in 0.4s
Results: 1000/1000 successful
QPS: 2257.9
Saved: benchmark_results.json
```

#### Online Benchmark (Poisson Distribution)

```bash
# Test sustained QPS with latency focus
inference-endpoint -v benchmark online \
  --endpoint http://localhost:8765 \
  --model gpt-3.5-turbo \
  --dataset tests/datasets/dummy_1k.pkl \
  --qps 100 \
  --duration 10 \
  --workers 2
```

**Expected Output:**

```
Loading: dummy_1k.pkl
Loaded 1000 samples
Mode: perf, QPS: 100.0, Responses: False
Scheduler: PoissonDistributionScheduler (online mode, 100.0 QPS target)
Connecting: http://localhost:8765
Running...
Completed in 10.7s
Results: 1000/1000 successful
QPS: 93.8
```

### 5. Test Other Commands

```bash
# Show info
inference-endpoint -v info

# Generate template
inference-endpoint init --template offline

# Validate config
inference-endpoint validate --config offline_template.yaml

# Test with existing dataset
inference-endpoint benchmark offline \
  --endpoint http://localhost:8765 \
  --dataset tests/datasets/ds_samples.pkl \
  --duration 5 \
  -v
```

### 6. View Results

```bash
# View benchmark results
cat benchmark_results.json | jq

# Example output:
{
  "config": {
    "endpoint": "http://localhost:8765",
    "mode": null,
    "qps": 10
  },
  "results": {
    "total": 1000,
    "successful": 1000,
    "failed": 0,
    "elapsed_time": 1.8,
    "qps": 555.6
  }
}
```

### 7. Stop the Echo Server

Press `Ctrl+C` in the terminal running the echo server, or:

```bash
pkill -f echo_server
```

## Echo Server Options

```bash
# Custom host and port
python -m inference_endpoint.testing.echo_server --host 0.0.0.0 --port 9000

# Check help
python -m inference_endpoint.testing.echo_server --help
```

## Request Format

The echo server expects OpenAI-compatible format but simplifies it:

**What workers send (internal):**

```json
{
  "prompt": "Your query text",
  "model": "model-name",
  "max_tokens": 50,
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
Error: prompt not found in json_value
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
python -m inference_endpoint.testing.echo_server --port 8000 &

# 2. Generate fresh dataset if needed
python scripts/create_dummy_dataset.py

# 3. Test probe first
inference-endpoint probe --endpoint http://localhost:8000 --requests 10

# 4. Run benchmark
inference-endpoint -v benchmark offline \
  --endpoint http://localhost:8000 \
  --dataset tests/datasets/dummy_1k.pkl \
  --duration 10 \
  --workers 4 \
  --concurrency 50 \
  --output benchmark_results.json

# 5. Check results
cat benchmark_results.json | jq '.results'

# 6. Stop server
pkill -f echo_server
```

### Testing Different Modes

```bash
# Offline (max throughput)
inference-endpoint benchmark offline \
  --endpoint http://localhost:8765 \
  --dataset tests/datasets/dummy_1k.pkl \
  --workers 4

# Online (Poisson distribution)
inference-endpoint benchmark online \
  --endpoint http://localhost:8765 \
  --dataset tests/datasets/dummy_1k.pkl \
  --qps 500 \
  --workers 4
```

## Tips

- Use `-v` for INFO logging, `-vv` for DEBUG
- Echo server mirrors back the prompt as the response
- Perfect for testing CLI without real LLM endpoint
- Fast responses (no actual inference)
- Press `Ctrl+C` to gracefully interrupt benchmarks
- Probe shows progress indicators for large request counts
- Default test dataset: `tests/datasets/dummy_1k.pkl` (1000 samples, ~133 KB)
- Model name optional for testing (defaults to gpt-3.5-turbo with warning)
- Dataset format auto-inferred from file extension (pkl, hf directory)
