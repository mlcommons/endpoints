# CLI Quick Reference

Command-line reference for all `inference-endpoint` subcommands, flags, load patterns, and usage examples.

> **Note:** Commands below assume an activated venv (`source .venv/bin/activate`). Without activation, prefix all commands with `uv run`.

## Commands

### Performance Benchmarking

```bash
# Offline (max throughput)
inference-endpoint benchmark offline \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/assets/datasets/dummy_1k.jsonl

# Online (sustained QPS - requires --load-pattern, --target-qps)
inference-endpoint benchmark online \
  --endpoints URL \
  --model Qwen/Qwen3-8B \
  --dataset tests/assets/datasets/dummy_1k.jsonl \
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
  --dataset tests/assets/datasets/dummy_1k.jsonl \
  --report-dir my_benchmark_report

# YAML-based
inference-endpoint benchmark from-config --config test.yaml
```

**Default Test Dataset:** Use `tests/assets/datasets/dummy_1k.jsonl` (1000 samples) for local testing.

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
inference-endpoint init offline        # or: online, concurrency, eval, submission

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
- `--runtime.n-samples-to-issue --num-samples` - Explicit sample count (omit to issue the dataset once — the default)
- `--client.num-workers --workers` - HTTP workers (-1=auto, default: -1)
- `--client.max-connections --max-connections` - Max TCP connections (-1=unlimited)
- `--endpoint-config.api-key --api-key` - API authentication
- `--endpoint-config.api-type --api-type` - API type: openai/sglang (default: openai)
- `--report-dir` - Report output directory
  Note: `benchmark from-config` also accepts `--report-dir` as an override of the YAML value;
  when neither is set a default report directory is used.
- `--timeout` - Whole-run watchdog in seconds (off by default). If it fires, the run is aborted, the report is marked INTERRUPTED, and the process exits non-zero.
- `--enable-cpu-affinity / --no-cpu-affinity` - NUMA-aware CPU pinning (default: true)
- `--no-early-stopping` - opt out of the MLPerf early-stopping percentile estimates in `result_summary.json` (default: on; see [early_stopping.md](early_stopping.md))

**Online-specific:**

- `--load-pattern.type --load-pattern` - Load pattern: poisson or concurrency (required for online)
- `--load-pattern.target-qps --target-qps` - Target QPS (required for poisson)
- `--load-pattern.target-concurrency --concurrency` - Concurrent requests (required for concurrency)

**All other schema fields** are accessible via dotted paths (e.g., `--model-params.temperature`, `--model-params.top-k`, `--runtime.scheduler-random-seed`). Run `--help` to see the full list.

## Time Knobs

All give-up deadlines live under `settings.timeouts`; the only workload duration is
`settings.runtime.max_duration_ms`; endpoint-client worker lifecycle timeouts are client
internals under `settings.client`. `null`/unset means "wait indefinitely" (or "off") everywhere.

Where every knob acts over the life of a run:

```text
run_benchmark ── run_timeout_s deadline captured here ─────────────────────────────┐
│                                                                                  │
├─ setup: dataset + tokenizer load          (counts against run_timeout_s)         │
├─ launch metrics/event-logger services     ── service_ready_timeout_s             │
├─ start endpoint-client workers            ── client.worker_initialization_timeout│
│                                                                                  │
├─ WARMUP       issue ──────────┤ drain ─┤  ── warmup_drain_timeout_s              │
│                                                                                  │
├─ PERFORMANCE  issue ──────────┤ drain ─┤                                         │
│               │               │        └─ performance_drain_timeout_s            │
│               └───────────────┴─ max_duration_ms caps ISSUING only; reaching it  │
│                                  ends the phase NORMALLY (valid report) and      │
│                                  SKIPS the drain — both knobs may be set; the    │
│                                  drain timeout applies only when issuance        │
│                                  finishes before the cap                         │
│                                                                                  │
├─ ACCURACY     issue ──────────┤ drain ─┤  ── accuracy_drain_timeout_s            │
│                                                                                  │
├─ metrics drain (tokenize buffered ISL/OSL)── metrics_drain_timeout_s             │
│                                              (expiry FAILS the run:              │
│                                              complete: false + non-zero exit)    │
├─ worker shutdown                          ── client.worker_graceful_shutdown_wait│
│                                              then client.worker_force_kill_timeout
│                                                                                  │
 run_timeout_s (whole-run watchdog) ───────────────────────────────────────────────┘
 firing at ANY point above aborts the run: report marked INTERRUPTED, non-zero exit
└─ finalize: score accuracy, write artifacts   (OUTSIDE the watchdog: a timed-out
                                                run SKIPS scoring, writes its
                                                INTERRUPTED artifacts, exits
                                                non-zero; scoring of a run that
                                                finished within budget is not
                                                deadline-bounded)
```

| YAML path                                       | CLI flag                                 | Semantics                                                                                                                                                                                                                     |
| ----------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `settings.runtime.min_duration_ms`              | `--runtime.min-duration-ms`              | Poisson only (requires explicit `target_qps`). Sizes the run by time: issue `target_qps` × duration samples (ms, or suffix: `600s`, `10m`). Explicit `--num-samples` wins; both unset = issue the dataset once                |
| `settings.runtime.max_duration_ms`              | `--runtime.max-duration-ms`              | Caps performance-phase issuing (ms, or suffix: `600s`, `10m`); reaching it ends the phase NORMALLY — the report stays valid — and skips the performance drain                                                                 |
| `settings.timeouts.run_timeout_s`               | `--timeout`                              | Whole-run watchdog from setup through worker shutdown; firing aborts the run — report marked INTERRUPTED, non-zero exit. Finalization (accuracy scoring, artifact writes) runs after the watchdog and is not deadline-bounded |
| `settings.timeouts.service_ready_timeout_s`     | `--service-ready-timeout`                | Wait for the metrics-aggregator/event-logger services to become ready (default 30)                                                                                                                                            |
| `settings.timeouts.warmup_drain_timeout_s`      | `--warmup-drain-timeout`                 | Bound on in-flight warmup requests after the warmup phase ends (default 240)                                                                                                                                                  |
| `settings.timeouts.performance_drain_timeout_s` | `--performance-drain-timeout`            | Bound on in-flight performance requests after the phase stops issuing (default: wait indefinitely)                                                                                                                            |
| `settings.timeouts.accuracy_drain_timeout_s`    | `--accuracy-drain-timeout`               | Bound on in-flight accuracy requests after the phase ends (default: wait indefinitely)                                                                                                                                        |
| `settings.timeouts.metrics_drain_timeout_s`     | `--metrics-drain-timeout`                | Budget for the metrics aggregator to finish tokenizing buffered samples after the run ends (default: wait indefinitely); expiring fails the run with `complete: false` artifacts                                              |
| `settings.client.worker_initialization_timeout` | `--client.worker-initialization-timeout` | Wait for endpoint-client worker processes to start (default 60)                                                                                                                                                               |
| `settings.client.worker_graceful_shutdown_wait` | `--client.worker-graceful-shutdown-wait` | Post-run wait for workers to exit gracefully (default 0.5)                                                                                                                                                                    |
| `settings.client.worker_force_kill_timeout`     | `--client.worker-force-kill-timeout`     | Wait after the graceful window before force-killing workers (default 0.5)                                                                                                                                                     |

How the knobs compose:

1. **`--num-samples` / dataset-once defines the work.** An explicit `runtime.n_samples_to_issue`
   sets the sample count; omitting it issues the performance dataset once.
2. **`runtime.max_duration_ms` caps performance-phase issuing** and ends the phase normally —
   remaining samples are not issued, in-flight requests are abandoned (no drain), the report is
   valid. It does not bound the drain: issuing and draining are consecutive, never concurrent.
3. **Per-phase drain timeouts bound the post-phase wait** for requests still in flight after a
   phase stops issuing on its own.
4. **`timeouts.run_timeout_s` is the only total-wall-time bound** (setup, every phase, every
   drain) — firing aborts the run, marks the report INTERRUPTED, and exits non-zero.

### Ctrl-C (SIGINT)

One handler owns SIGINT for the whole run:

- **First ^C**: graceful abort. The session stops issuing, in-flight drains are
  released, buffered samples still reach the metrics aggregator, and the
  artifacts land honest — `final_snapshot.json` `state: interrupted`,
  `result_summary.json` `complete: false`, `events.jsonl` flushed. Exit 130.
- **Further ^C**: no-op. One keystroke can be delivered repeatedly (process
  runners like `uv run` forward the terminal's group SIGINT to a child that
  already received it), so repeats are indistinguishable from the first. A
  wedged teardown is bounded by `run_timeout_s` or killed externally.
- **^C during setup** (dataset/tokenizer load, before services): immediate
  abort, exit 130, no artifacts.

A ^C'd run never exits 0 and never writes `complete: true` artifacts.

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

Format is auto-detected from file extension. Override with `format=<ext>` in the dataset string.

**Supported:** `.csv`, `.json`, `.jsonl`, `.parquet`, `huggingface`

## Test Modes

**perf** (default) - Performance only (no response storage)

- Max throughput testing
- Metrics: QPS, latency, TTFT, TPOT
- Ordinary configured scoring remains available, but external scorers are skipped
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
  --dataset acc:eval.jsonl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer,accuracy_config.extractor=boxed_math_extractor \
  --mode both
```

> **Note:** Submission runs (`type: submission`) are YAML-only — they require `submission_ref` and `benchmark_mode` fields not exposed in CLI.

Report directories contain a sanitized `config.yaml`: credentials and other
secret values are replaced with `<redacted>`. Restore those values before
reusing that file as benchmark input.

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
  --dataset tests/assets/datasets/dummy_1k.jsonl
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

# Without --num-samples, the dataset is issued once (the default)
inference-endpoint benchmark online \
  --endpoints https://api.production.com \
  --model Qwen/Qwen3-8B \
  --dataset prod_queries.jsonl \
  --load-pattern poisson \
  --target-qps 100 \
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
  --config submission_template.yaml
# from-config accepts --config, --timeout, --mode, --accuracy-only, and
# --report-dir; everything else comes from the YAML.
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
    n_samples_to_issue: null # Optional: explicit sample count (null = issue the dataset once)
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

- `--num-samples` sets an explicit sample count; without it the dataset is issued once
- Behavior change: bare configs (no `--num-samples`) now run the dataset once instead of deriving 10 minutes' worth of samples from the target QPS

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
