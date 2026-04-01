# CLI Quick Reference

## Architecture

The CLI is auto-generated from Pydantic models in `config/schema.py` using
cyclopts. schema.py is the single source of truth for both YAML configs and CLI flags.

- **All schema fields** available as CLI flags on each subcommand (dotted kebab-case)
- **Shorthand aliases** declared via `cyclopts.Parameter(alias="--flag")` on schema fields
- **`${VAR}` interpolation** in YAML files (with `${VAR:-default}` fallback)

## Commands

### Performance Benchmarking

```bash
# Offline (max throughput)
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl

# Online (sustained QPS - requires --load-pattern, --target-qps)
inference-endpoint benchmark online \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --load-pattern poisson \
  --target-qps 100

# Multiple datasets (--dataset is repeatable, prefix with perf: or acc:)
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset perf:performance.jsonl \
  --dataset acc:accuracy.jsonl \
  --mode both

# With detailed report generation
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --report-dir my_benchmark_report

# YAML-based
inference-endpoint benchmark from-config --config test.yaml
```

**Default Test Dataset:** Use `tests/datasets/dummy_1k.jsonl` (1000 samples) for local testing.

**Dataset format:** `--dataset [perf|acc:]<path>[,key=value...]` — TOML-style dotted paths. Type prefix is optional (defaults to `perf`):

```bash
--dataset data.jsonl                                         # simple path
--dataset acc:eval.jsonl                                     # accuracy dataset
--dataset data.csv,samples=500,parser.prompt=article         # with options
--dataset perf:data.jsonl,format=.jsonl,parser.prompt=text    # explicit format + remap
```

### Accuracy Evaluation (stub - future implementation)

```bash
inference-endpoint eval --dataset gpqa,aime --endpoints URL
```

### Pre-flight Testing

```bash
# Test endpoint connectivity
inference-endpoint probe \
  --endpoints URL \
  --model gpt-3.5-turbo

# Validate YAML config
inference-endpoint validate-yaml -c test.yaml
```

### Utilities

```bash
# Generate config templates
inference-endpoint init offline        # or: online, eval, submission
inference-endpoint init submission

# Show system info
inference-endpoint info
```

## Common Options (Benchmark Subcommands)

Flag names shown as `--full.dotted.path --alias`. Both forms work.

**Required:**

- `--endpoint-config.endpoints --endpoints` - Endpoint URL(s)
- `--model-params.name --model` - Model name (e.g., Qwen/Qwen3-8B)
- `--dataset` - Dataset file path

**Optional (with aliases):**

- `--model-params.max-new-tokens --max-output-tokens` - Max output tokens (default: 1024)
- `--model-params.osl-distribution.min --min-output-tokens` - Min output tokens (default: 1)
- `--model-params.streaming --streaming` - Streaming mode: auto/on/off (default: auto)
- `--runtime.min-duration-ms --duration` - Min duration: ms default, or with suffix (600s, 10m) (default: 600000)
- `--runtime.n-samples-to-issue --num-samples` - Explicit sample count override
- `--client.num-workers --workers` - HTTP workers (-1=auto, default: -1)
- `--client.max-connections --max-connections` - Max TCP connections (-1=unlimited)
- `--endpoint-config.api-key --api-key` - API authentication
- `--endpoint-config.api-type --api-type` - API type: openai/sglang (default: openai)
- `--report-dir` - Report output directory
- `--timeout` - Global timeout in seconds
- `--enable-cpu-affinity / --no-cpu-affinity` - NUMA-aware CPU pinning (default: true)

**Online-specific:**

- `--load-pattern.type --load-pattern` - Load pattern: poisson or concurrency (required for online)
- `--load-pattern.target-qps --target-qps` - Target QPS (required for poisson)
- `--load-pattern.target-concurrency --concurrency` - Concurrent requests (required for concurrency)

**All other schema fields** are accessible via dotted paths (e.g., `--model-params.temperature`, `--model-params.top-k`, `--runtime.scheduler-random-seed`). Run `--help` to see the full list.

## Environment Variables

**In YAML files** — use `${VAR}` or `${VAR:-default}` syntax:

```yaml
endpoint_config:
  endpoints:
    - "${ENDPOINT_URL}"
  api_key: "${API_KEY:-sk-test}"
model_params:
  name: "${MODEL_NAME:-Qwen/Qwen3-8B}"
```

## Dataset Formats

Format is auto-detected from file extension. Override with `format:` in the dataset string.

**Supported:** `csv`, `json`, `jsonl`, `parquet`, `huggingface`

## Test Modes

**perf** (default) - Performance only (no response storage)

- Max throughput testing
- Metrics: QPS, latency, TTFT, TPOT
- Fastest - no response collection overhead

**acc** - Accuracy only (collect all responses)

- Response collection and evaluation
- Metrics: Accuracy %
- Requires `accuracy_config` on datasets (eval_method, extractor)

**both** - Combined (for official submissions)

- Performance datasets: metrics only
- Accuracy datasets: collect + evaluate
- Selective collection based on dataset type

Accuracy config is supported in both CLI and YAML:

```bash
# CLI — accuracy config via dotted paths
--dataset acc:eval.jsonl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer,accuracy_config.extractor=boxed_math_extractor

# Combined perf + accuracy
inference-endpoint benchmark offline \
  --endpoints URL --model M \
  --dataset perf:perf.jsonl \
  --dataset acc:eval.jsonl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer \
  --mode both
```

> **Note:** Submission runs (`type: submission`) are YAML-only — they require `submission_ref` and `benchmark_mode` fields not exposed in CLI.

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
  --dataset tests/datasets/dummy_1k.jsonl
```

### Production Benchmark

```bash
# With explicit sample count
inference-endpoint benchmark online \
  --endpoints https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.jsonl \
  --load-pattern poisson \
  --target-qps 100 \
  --num-samples 10000 \
  --workers 16 \
  --report-dir production_report \
  -v

# Or with duration (calculates samples from target_qps * duration)
inference-endpoint benchmark online \
  --endpoints https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.jsonl \
  --load-pattern poisson \
  --target-qps 100 \
  --duration 5m \
  --workers 16 \
  --report-dir production_report \
  -v
```

### Official Submission

```bash
# 1. Generate template
inference-endpoint init submission

# 2. Edit submission_template.yaml (set model, datasets, ruleset, endpoint)

# 3. Run (YAML mode)
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
inference-endpoint validate-yaml --config submission.yaml
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
    path: "openorca.jsonl"
  - name: "gpqa"
    type: "accuracy"
    path: "gpqa.jsonl"
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
    num_workers: -1 # auto

metrics:
  collect: ["throughput", "latency", "ttft", "tpot"]

endpoint_config:
  endpoints:
    - "http://localhost:8000"
  api_key: null
```

Note: For submission configs, `model_params.name` is optional when `submission_ref.model` is provided — the model name is resolved automatically.

## CLI vs YAML Modes

**CLI Mode** (`benchmark offline/online`):

- All parameters from command line
- Quick testing and iteration
- Example: `benchmark offline --endpoints URL --model NAME --dataset FILE`

**YAML Mode** (`benchmark from-config`):

- All configuration from YAML file
- Reproducible, shareable configs
- Supports `${VAR}` env var interpolation
- Optional `--timeout` and `--mode` overrides
- Example: `benchmark from-config -c file.yaml --timeout 600`

## Tips

**Sample Count Control:**

- Priority: `--num-samples` > calculated (target_qps × duration) > dataset size
- Default duration: 600000ms (10 minutes)

**Mode Requirements:**

- Online mode requires `--load-pattern` (poisson or concurrency)
  - `poisson` requires `--target-qps`
  - `concurrency` requires `--concurrency`
- Use `--mode both` for combined perf + accuracy runs
- Streaming: auto (default) resolves to off for offline, on for online

**Best Practices:**

- Share YAML configs for reproducible results across systems
- Use `--report-dir` for detailed metrics with TTFT, TPOT, and token analysis
- Set `HF_TOKEN` environment variable for non-public models
- Use `--min-output-tokens` and `--max-output-tokens` to control output length
- Use `${VAR:-default}` in YAML for environment-specific configs
