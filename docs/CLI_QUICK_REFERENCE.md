# CLI Quick Reference

## Commands

### Performance Benchmarking

```bash
# Offline (max throughput)
inference-endpoint benchmark offline \
  --endpoint URL \
  --model llama-2-70b \
  --dataset tests/datasets/dummy_1k.pkl

# Online (sustained QPS with latency focus)
inference-endpoint benchmark online \
  --endpoint URL \
  --model llama-2-70b \
  --dataset tests/datasets/dummy_1k.pkl \
  --qps 500

# YAML-based (reproducible)
inference-endpoint benchmark \
  --config test.yaml \
  --endpoint URL \
  --api-key KEY
```

**Default Test Dataset:** Use `tests/datasets/dummy_1k.pkl` (1000 samples, ~133 KB) for local testing.

### Accuracy Evaluation (stub - future implementation)

```bash
inference-endpoint eval --dataset gpqa,aime --endpoint URL
```

### Pre-flight Testing

```bash
# Test endpoint connectivity
inference-endpoint probe \
  --endpoint URL \
  --model gpt-3.5-turbo \
  --api-key KEY

# Validate YAML config
inference-endpoint validate --config test.yaml
```

### Utilities

```bash
# Generate config templates
inference-endpoint init --template offline        # or: online, eval, submission

# Show system info
inference-endpoint info
```

## Common Options

- `--endpoint, -e URL` - Endpoint URL (required for benchmarks and probe)
- `--api-key KEY` - API authentication
- `--model NAME` - Model name (e.g., llama-2-70b, gpt-3.5-turbo)
- `--dataset, -d PATH` - Dataset file (pkl, jsonl, or HuggingFace directory)
- `--config, -c PATH` - YAML configuration file
- `--output, -o PATH` - Save results to JSON
- `--verbose, -v` - Increase verbosity (-vv for debug)

## Benchmark Options

- `--qps N` - Queries per second (usage varies by mode)
- `--duration SEC` - Test duration in seconds
- `--workers N` - Number of HTTP workers
- `--concurrency N` - Max concurrent requests
- `--min-tokens N` - Min output tokens (OSL control)
- `--max-tokens N` - Max output tokens (OSL control)
- `--mode MODE` - Test mode: `perf` (default), `acc`, or `both`

## Dataset Formats

**Supported:**

- `pkl` - Pickle format (default)
- `hf` - HuggingFace datasets

**Coming Soon:**

- `jsonl` - JSON Lines format

## Test Modes

**perf** (default) - Performance only (no response storage)

- Max throughput testing
- Metrics: QPS, latency, TTFT, TPOT
- Fastest - no response collection overhead

**acc** - Accuracy only (collect all responses)

- Response collection and evaluation
- Metrics: Accuracy %
- Use for evaluation runs

**both** - Combined (for official submissions)

- Performance datasets: metrics only
- Accuracy datasets: collect + evaluate
- Selective collection based on dataset type

## Load Patterns

**max_throughput** - Offline mode

- All queries issued at t=0 (burst)
- Measures maximum sustainable throughput
- Use with `benchmark offline`

**poisson** - Online mode (fixed QPS)

- Queries follow Poisson distribution
- Sustains target QPS
- Use with `benchmark online --qps N`

**concurrency** - Online mode (fixed concurrency) - NOT YET IMPLEMENTED

- Maintains N concurrent requests
- QPS emerges from concurrency/latency
- Will be available in future release

## Examples

### Quick Test

```bash
inference-endpoint benchmark offline \
  --endpoint http://localhost:8000 \
  --model gpt-3.5-turbo \
  --dataset tests/datasets/dummy_1k.pkl
```

### Production Benchmark

```bash
inference-endpoint benchmark online \
  --endpoint https://api.production.com \
  --model llama-2-70b \
  --dataset prod_queries.pkl \
  --qps 100 \
  --duration 300 \
  --workers 16 \
  --output results.json \
  -v
```

### Official Submission

```bash
# 1. Generate template
inference-endpoint init --template submission

# 2. Edit submission_template.yaml (set model, datasets, ruleset)

# 3. Run
inference-endpoint benchmark \
  --config submission_template.yaml \
  --endpoint https://your-endpoint.com \
  --api-key $API_KEY \
  --output official_results.json
```

### Validate First

```bash
inference-endpoint probe \
  --endpoint https://api.example.com \
  --model gpt-3.5-turbo
inference-endpoint validate --config submission.yaml
```

## YAML Config Structure

```yaml
name: "test-name"
type: "submission" # offline|online|eval|submission

baseline:
  locked: true
  model: "llama-2-70b"
  ruleset: "mlperf-inference-v6.0"

model_params:
  temperature: 0.7
  top_k: 50
  top_p: 0.9
  max_new_tokens: 2048

datasets:
  - name: "perf"
    type: "performance"
    path: "openorca.pkl"
  - name: "gpqa"
    type: "accuracy"
    path: "gpqa.pkl"
    eval_method: "exact_match"

settings:
  load_pattern:
    type: "max_throughput"
  client:
    workers: 4
    max_concurrency: 50

environment:
  endpoint: "http://localhost:8000"
```

## Tips

- Use `--mode both` for combined perf + accuracy runs
- Use `--min-tokens` and `--max-tokens` to control output length
- Locked baselines in official configs prevent accidental changes
- Share YAML configs for reproducible results across systems
