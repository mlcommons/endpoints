# MLPerf Inference Endpoint Benchmarking System

[![Tests](https://github.com/mlcommons/endpoints/actions/workflows/test.yml/badge.svg)](https://github.com/mlcommons/endpoints/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mlcommons/endpoints/branch/main/graph/badge.svg)](https://codecov.io/gh/mlcommons/endpoints)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

A high-performance benchmarking tool for LLM inference endpoints, targeting 50k+ QPS. Part of [MLCommons](https://mlcommons.org/).

## Quick Start

**Requirements:** Python 3.12+ (3.12 recommended)

```bash
git clone https://github.com/mlcommons/endpoints.git
cd endpoints
uv sync
```

<details>
<summary>Using pip + venv instead (backward-compatible)</summary>

> **Note:** Does not use `uv.lock` — dependency versions may differ from the lockfile.

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install .
```

After activating the venv, commands work without the `uv run` prefix.

</details>

```bash
# Test endpoint connectivity
uv run inference-endpoint probe \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B

# Run offline benchmark (max throughput)
uv run inference-endpoint benchmark offline \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/assets/datasets/dummy_1k.jsonl

# Run online benchmark (sustained QPS)
uv run inference-endpoint benchmark online \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/assets/datasets/dummy_1k.jsonl \
  --load-pattern poisson \
  --target-qps 100
```

### Local Testing

```bash
# Start local echo server and run a benchmark against it
uv run python -m inference_endpoint.testing.echo_server --port 8765 &
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model test-model \
  --dataset tests/assets/datasets/dummy_1k.jsonl
pkill -f echo_server
```

See [Local Testing Guide](docs/LOCAL_TESTING.md) for more details.

## Architecture

```
Dataset Manager ──> Load Generator ──> Endpoint Client ──> External Endpoint
                         |
                    Metrics Collector (EventRecorder + MetricsReporter)
```

| Component           | Purpose                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Load Generator**  | Central orchestrator: `BenchmarkSession` owns lifecycle, `Scheduler` controls timing                                              |
| **Endpoint Client** | Multi-process HTTP workers communicating via ZMQ IPC                                                                              |
| **Dataset Manager** | Loads JSONL, HuggingFace, CSV, JSON, Parquet datasets                                                                             |
| **Metrics**         | SQLite-backed event recording, aggregation (QPS, latency, TTFT, TPOT), MLPerf early-stopping percentile estimates (on by default) |
| **Config**          | Pydantic-based YAML schema, CLI auto-generated via cyclopts                                                                       |

### Benchmark Modes

- **Offline** (`max_throughput`): Burst all queries at once for peak throughput measurement
- **Online** (`poisson`): Fixed QPS with Poisson arrival distribution for latency profiling
- **Concurrency**: Fixed concurrent request count

### Endpoint liveness

An in-flight request can stall without returning a response. Opt in to a no-progress deadline to end the benchmark rather than waiting for its overall wall-time limit:

```yaml
settings:
  runtime:
    # Choose a value above the longest expected interval without a response.
    # For a non-streaming endpoint, that is the full request latency.
    no_progress_timeout_s: 30
```

For direct `benchmark offline` or `benchmark online` invocations, use `--no-progress-timeout 30` (equivalently, `--runtime.no-progress-timeout-s 30`). `benchmark from-config` reads the setting from YAML.

The deadline is armed only while requests are in flight. Each streamed response chunk or final response resets it, so normal long streaming generations are not failed. Setting the timeout is the explicit opt-in. This is a client-side liveness guard; it does not replace the inference engine's own request timeout or detect a server that hangs before it accepts requests.

For TensorRT-LLM disaggregated serving, match this to the executor hang-detection timeout, not the KV cache-transfer timeout. TensorRT-LLM's default `hang_detection_timeout` is 300 seconds; start with `no_progress_timeout_s: 300` and raise it if a normal p99 interval without a response is longer than 300 seconds.

### Performance Design

The hot path is optimized for minimal overhead:

- Multi-process workers with ZMQ IPC (not threads)
- `uvloop` + `eager_task_factory` for async performance
- `msgspec` for zero-copy serialization on the data path
- Custom HTTP connection pooling with `httptools` parser
- CPU affinity support for performance tuning

## Accuracy Evaluation

Run accuracy evaluation with Pass@1 scoring using pre-defined benchmarks:

- **GPQA** (default: GPQA Diamond)
- **AIME** (default: AIME 2025)
- **LiveCodeBench** (default: lite, release_v6) — requires [additional setup](src/inference_endpoint/dataset_manager/predefined/livecodebench/README.md)

## Documentation

| Guide                                                          | Description                           |
| -------------------------------------------------------------- | ------------------------------------- |
| [CLI Quick Reference](docs/CLI_QUICK_REFERENCE.md)             | Command-line interface guide          |
| [CLI Design](docs/CLI_DESIGN.md)                               | CLI architecture and design decisions |
| [Local Testing](docs/LOCAL_TESTING.md)                         | Test with the echo server             |
| [Client Performance Tuning](docs/CLIENT_PERFORMANCE_TUNING.md) | Endpoint client optimization          |
| [Performance Architecture](docs/PERF_ARCHITECTURE.md)          | Performance architecture deep dive    |
| [Development Guide](docs/DEVELOPMENT.md)                       | Development setup and workflow        |
| [CONTRIBUTING.md](CONTRIBUTING.md)                             | How to contribute                     |

## Contributing

We welcome contributions from the community. See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup and prerequisites
- Code style (ruff, mypy, conventional commits)
- Testing requirements (>90% coverage, pytest markers)
- Pull request process and review expectations

Issues are tracked on our [project board](https://github.com/orgs/mlcommons/projects/57). Look for [`good first issue`](https://github.com/mlcommons/endpoints/labels/good%20first%20issue) or [`help wanted`](https://github.com/mlcommons/endpoints/labels/help%20wanted) to get started.

## Acknowledgements

This project draws inspiration from:

- [MLCommons Inference](https://github.com/mlcommons/inference) — MLPerf Inference benchmark suite
- [AIPerf](https://github.com/ai-dynamo/aiperf) — AI model performance profiling
- [SGLang GenAI-Bench](https://github.com/sgl-project/genai-bench) — Token-level performance evaluation
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) — Performance benchmarking for vLLM
- [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) - LLM inference optimization toolkit

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
