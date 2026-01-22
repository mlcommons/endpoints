# CLI Quick Reference

## Commands

### Performance Benchmarking

```bash
# Offline (max throughput - CLI mode)
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl

# Online (sustained QPS - CLI mode - requires --target-qps, --load-pattern)
inference-endpoint benchmark online \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --load-pattern poisson \
  --target-qps 100

# With detailed report generation
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --report-dir my_benchmark_report

# YAML-based (YAML mode - no CLI overrides)
inference-endpoint benchmark from-config \
  --config test.yaml
```

**Default Test Dataset:** Use `tests/datasets/dummy_1k.pkl` (1000 samples, ~133 KB) for local testing.

### Accuracy Evaluation (stub - future implementation)

```bash
inference-endpoint eval --dataset gpqa,aime --endpoints URL
```

### Pre-flight Testing

```bash
# Test endpoint connectivity
inference-endpoint probe \
  --endpoints URL \
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

- `--endpoints, -e URL` - Endpoint URL (required for CLI mode)
- `--model NAME` - Model name (required for CLI mode, e.g., Qwen/Qwen3-8B)
- `--dataset, -d PATH` - Dataset file (required for CLI mode)
- `--config, -c PATH` - YAML config file (required for from-config mode)
- `--report-dir PATH` - Save detailed benchmark report with metrics
- `--verbose, -v` - Increase verbosity (-vv for debug)

## Benchmark Options (CLI Mode Only)

- `--api-key KEY` - API authentication
- `--target-qps N` - Target queries per second (required when --load-pattern=poisson)
- `--duration SEC` - Test duration in seconds (default: 0 - run until dataset exhausted)
- `--num-samples N` - Number of samples to issue (overrides dataset size and duration calculation)
- `--streaming MODE` - Streaming control: `auto` (default), `on`, or `off`. Streaming will enable token streaming in response.
- `--workers N` - HTTP workers (default: 4)
- `--mode MODE` - Test mode: `perf` (default), `acc`, or `both`
- `--min-output-tokens N` - Min output tokens
- `--max-output-tokens N` - Max output tokens

## Online-Specific Options

- `--load-pattern TYPE` - Load pattern (required): `poisson`, `concurrency`
- `--concurrency N` - Max concurrent requests (required when --load-pattern=concurrency)

## Dataset Formats

**Supported:**

- `pkl` - Pickle format (default)
- `hf` - HuggingFace datasets
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
- Use with `benchmark online --target-qps N`

**concurrency** - Online mode (fixed concurrency)

- Maintains N concurrent requests
- QPS emerges from concurrency/latency
- Use with `benchmark online --load-pattern concurrency --concurrency N`

## Examples

### Quick Test

```bash
inference-endpoint benchmark offline \
  --endpoints http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl
```

### Production Benchmark

```bash
# With explicit sample count
inference-endpoint benchmark online \
  --endpoints https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.pkl \
  --load-pattern poisson \
  --target-qps 100 \
  --num-samples 10000 \
  --workers 16 \
  --output results.json \
  --report-path production_report \
  -v

# Or with duration (calculates samples from target_qps * duration)
inference-endpoint benchmark online \
  --endpoints https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.pkl \
  --load-pattern poisson \
  --target-qps 100 \
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
  --report-dir official_results
```

### Validate First

```bash
# Test connectivity
inference-endpoint probe \
  --endpoints https://api.example.com \
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
    n_samples_to_issue: null # Optional: explicit sample count (null = auto-calculate)
    scheduler_random_seed: 42 # For Poisson/distribution sampling
    dataloader_random_seed: 42 # For dataset shuffling
  load_pattern:
    type: "max_throughput"
    target_qps: 10.0
  client:
    workers: 4

metrics:
  collect: ["throughput", "latency", "ttft", "tpot"]

endpoint_config:
  endpoints:
    - "http://localhost:8000"
  api_key: null
```

## CLI vs YAML Modes

**CLI Mode** (`benchmark offline/online`):

- All parameters from command line
- Quick testing and iteration
- Examples: `benchmark offline --endpoints URL --model NAME --dataset FILE`

**YAML Mode** (`benchmark from-config`):

- All configuration from YAML file
- Reproducible, shareable configs
- No CLI parameter mixing (only --output auxiliary allowed)
- Example: `benchmark from-config --config file.yaml --output results.json`

## Tips

**Sample Count Control:**

- Sample priority: `--num-samples` > dataset size (duration=0) > calculated (target_qps × duration)
- Default duration: 0 (runs until dataset exhausted or max_duration reached)

**Mode Requirements:**

- Online mode requires `--load-pattern` (poisson or concurrency)
  - `--load-pattern poisson` requires `--target-qps`
  - `--load-pattern concurrency` requires `--concurrency`
- Use `--mode both` for combined perf + accuracy runs
- Streaming: auto (default) enables streaming responses for online, disables for offline

**Best Practices:**

- Share YAML configs for reproducible results across systems
- Use `--report-path` for detailed metrics with TTFT, TPOT, and token analysis
- Set `HF_TOKEN` environment variable for non-public models
- Use `--min-output-tokens` and `--max-output-tokens` to control output length
