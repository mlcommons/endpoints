# MLPerf Inference Endpoint Benchmarking System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://pre-commit.com/)

A high-performance benchmarking tool for LLM inference endpoints, targeting 50k+ QPS. Part of [MLCommons](https://mlcommons.org/).

## Quick Start

**Requirements:** Python 3.12+ (3.12 recommended)

```bash
git clone https://github.com/mlcommons/endpoints.git
cd endpoints
python3.12 -m venv venv && source venv/bin/activate
pip install .
```

```bash
# Test endpoint connectivity
inference-endpoint probe \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B

# Run offline benchmark (max throughput)
inference-endpoint benchmark offline \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl

# Run online benchmark (sustained QPS)
inference-endpoint benchmark online \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --load-pattern poisson \
  --target-qps 100
```

### Local Testing

```bash
# Start local echo server and run a benchmark against it
python -m inference_endpoint.testing.echo_server --port 8765 &
inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model test-model \
  --dataset tests/datasets/dummy_1k.jsonl
pkill -f echo_server
```

See [Local Testing Guide](docs/LOCAL_TESTING.md) for more details.

## Architecture

```
Dataset Manager ──> Load Generator ──> Endpoint Client ──> External Endpoint
                         |
                    Metrics Collector (EventRecorder + MetricsReporter)
```

| Component           | Purpose                                                                              |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Load Generator**  | Central orchestrator: `BenchmarkSession` owns lifecycle, `Scheduler` controls timing |
| **Endpoint Client** | Multi-process HTTP workers communicating via ZMQ IPC                                 |
| **Dataset Manager** | Loads JSONL, HuggingFace, CSV, JSON, Parquet datasets                                |
| **Metrics**         | SQLite-backed event recording, aggregation (QPS, latency, TTFT, TPOT)                |
| **Config**          | Pydantic-based YAML schema, CLI auto-generated via cyclopts                          |

### Benchmark Modes

- **Offline** (`max_throughput`): Burst all queries at once for peak throughput measurement
- **Online** (`poisson`): Fixed QPS with Poisson arrival distribution for latency profiling
- **Concurrency**: Fixed concurrent request count

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

All contributors must sign the [MLCommons CLA](https://mlcommons.org/membership/membership-overview/).

## Acknowledgements

This project draws inspiration from:

- [MLCommons Inference](https://github.com/mlcommons/inference) — MLPerf Inference benchmark suite
- [AIPerf](https://github.com/ai-dynamo/aiperf) — AI model performance profiling
- [SGLang GenAI-Bench](https://github.com/sgl-project/genai-bench) — Token-level performance evaluation
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) — Performance benchmarking for vLLM

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
