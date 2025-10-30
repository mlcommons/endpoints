# CLI Quick Reference

## Commands

### Performance Benchmarking

```bash
# Offline (max throughput - CLI mode)
inference-endpoint benchmark offline \
  --endpoint URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl

# Online (sustained QPS - CLI mode)
inference-endpoint benchmark online \
  --endpoint URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --qps 500

# With detailed report generation
inference-endpoint benchmark offline \
  --endpoint URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --report-path my_benchmark_report

# YAML-based (YAML mode - no CLI overrides)
inference-endpoint benchmark from-config \
  --config test.yaml \
  --output results.json
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

- `--endpoint, -e URL` - Endpoint URL (required for CLI mode)
- `--model NAME` - Model name (required for CLI mode, e.g., Qwen/Qwen3-8B)
- `--dataset, -d PATH` - Dataset file (required for CLI mode)
- `--config, -c PATH` - YAML config file (required for from-config mode)
- `--output, -o PATH` - Save results to JSON
- `--report-path PATH` - Save detailed benchmark report with metrics
- `--verbose, -v` - Increase verbosity (-vv for debug)

## Benchmark Options (CLI Mode Only)

- `--api-key KEY` - API authentication
- `--qps N` - Queries per second (default: 10.0)
- `--duration SEC` - Test duration in seconds (default: 10)
- `--workers N` - HTTP workers (default: 4)
- `--mode MODE` - Test mode: `perf` (default), `acc`, or `both`
- `--min-tokens N` - Min output tokens
- `--max-tokens N` - Max output tokens

## Online-Specific Options

- `--load-pattern TYPE` - Load pattern: `poisson` (default), `concurrency`
- `--concurrency N` - Max concurrent requests (default: -1 unlimited)

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
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl
```

### Production Benchmark

```bash
inference-endpoint benchmark online \
  --endpoint https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.pkl \
  --qps 100 \
  --duration 300 \
  --workers 16 \
  --output results.json \
  --report-path production_report \
  -v
```

### Official Submission

```bash
# 1. Generate template
inference-endpoint init --template submission

# 2. Edit submission_template.yaml (set model, datasets, ruleset, endpoint)

# 3. Run (YAML mode - no CLI overrides)
inference-endpoint benchmark from-config \
  --config submission_template.yaml \
  --output official_results.json
```

### Validate First

```bash
# Test connectivity
inference-endpoint probe \
  --endpoint https://api.example.com \
  --model Qwen/Qwen3-8B

# Validate YAML config
inference-endpoint validate --config submission.yaml
```

## YAML Config Structure

```yaml
name: "test-name"
type: "submission" # offline|online|eval|submission
benchmark_mode: "offline" # Required for submission: offline or online

submission_ref:
  model: "Qwen/Qwen3-8B"
  ruleset: "mlperf-inference-v5.1"

model_params:
  temperature: 0.7
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
  runtime:
    min_duration_ms: 600000 # 10 minutes
    random_seed: 42
  load_pattern:
    type: "max_throughput"
    qps: 10.0
  client:
    workers: 4
    max_concurrency: -1 # -1 = unlimited

metrics:
  collect: ["throughput", "latency", "ttft", "tpot"]

endpoint_config:
  endpoint: "http://localhost:8000"
  api_key: null
```

## CLI vs YAML Modes

**CLI Mode** (`benchmark offline/online`):

- All parameters from command line
- Quick testing and iteration
- Examples: `benchmark offline --endpoint URL --model NAME --dataset FILE`

**YAML Mode** (`benchmark from-config`):

- All configuration from YAML file
- Reproducible, shareable configs
- No CLI parameter mixing (only --output auxiliary allowed)
- Example: `benchmark from-config --config file.yaml --output results.json`

## Tips

- Use `--mode both` for combined perf + accuracy runs
- Use `--min-tokens` and `--max-tokens` to control output length
- Default duration: 10 seconds (use --duration to override)
- Default max_concurrency: -1 (unlimited)
- Share YAML configs for reproducible results across systems
- Use `--report-path` to generate detailed metrics reports with TTFT, TPOT, and token-based analysis
- Set `HF_TOKEN` environment variable for non-public models (e.g., `export HF_TOKEN=your_token`)
- Model name is used to load tokenizer automatically for token-based metrics in reports
