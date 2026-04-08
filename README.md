# MLPerf® Inference Endpoint Benchmarking System

A high-performance benchmarking tool for LLM endpoints.

## Quick Start

### Installation

**Requirements**: Python 3.12+ (Python 3.12 is recommended for optimal performance. GIL-less mode in higher Python versions is not yet supported.)

```bash
# Clone the repository
# Note: This repo will be migrated to https://github.com/mlcommons/endpoints
git clone https://github.com/mlcommons/endpoints.git
cd endpoints

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# As a user
pip install .

# As a developer (with development and test extras)
pip install -e ".[dev,test]"
pre-commit install
```

### Basic Usage

```bash
# Show help
inference-endpoint --help

# Show system information
inference-endpoint -v info

# Test endpoint connectivity
inference-endpoint probe \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B

# Run offline benchmark (max throughput - uses all dataset samples)
inference-endpoint benchmark offline \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl

# Run online benchmark (sustained QPS - requires --target-qps, --load-pattern)
inference-endpoint benchmark online \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --load-pattern poisson \
  --target-qps 100

# With explicit sample count
inference-endpoint benchmark offline \
  --endpoints http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl \
  --num-samples 5000
```

### Running Locally

```bash
# Start local echo server
python3 -m inference_endpoint.testing.echo_server --port 8765 &

# Test with dummy dataset (included in repo)
inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl

# Stop echo server
pkill -f echo_server
```

See [Local Testing Guide](docs/LOCAL_TESTING.md) for detailed instructions.

### Running Tests and Examples

```bash
# Install test dependencies
pip install ".[test]"

# Run tests (excluding performance and explicit-run tests)
pytest -m "not performance and not run_explicitly"

# Run examples: follow instructions in examples/*/README.md
```

## 📚 Documentation

- [AGENTS.md](AGENTS.md) - Architecture, conventions, and AI agent guidelines
- [CLI Quick Reference](docs/CLI_QUICK_REFERENCE.md) - Command-line interface guide
- [Local Testing Guide](docs/LOCAL_TESTING.md) - Test with echo server
- [Development Guide](docs/DEVELOPMENT.md) - How to contribute and develop
- [Performance Architecture](docs/PERF_ARCHITECTURE.md) - Hot-path design and tuning
- [Performance Tuning](docs/CLIENT_PERFORMANCE_TUNING.md) - CPU affinity and client tuning
- [GitHub Setup Guide](docs/GITHUB_SETUP.md) - GitHub authentication and setup

### Component Design Specs

Each top-level component under `src/inference_endpoint/` has a corresponding spec:

| Component         | Spec                                                             |
| ----------------- | ---------------------------------------------------------------- |
| Core types        | [docs/core/DESIGN.md](docs/core/DESIGN.md)                       |
| Load generator    | [docs/load_generator/DESIGN.md](docs/load_generator/DESIGN.md)   |
| Endpoint client   | [docs/endpoint_client/DESIGN.md](docs/endpoint_client/DESIGN.md) |
| Metrics           | [docs/metrics/DESIGN.md](docs/metrics/DESIGN.md)                 |
| Config            | [docs/config/DESIGN.md](docs/config/DESIGN.md)                   |
| Async utils       | [docs/async_utils/DESIGN.md](docs/async_utils/DESIGN.md)         |
| Dataset manager   | [docs/dataset_manager/DESIGN.md](docs/dataset_manager/DESIGN.md) |
| Commands (CLI)    | [docs/commands/DESIGN.md](docs/commands/DESIGN.md)               |
| OpenAI adapter    | [docs/openai/DESIGN.md](docs/openai/DESIGN.md)                   |
| SGLang adapter    | [docs/sglang/DESIGN.md](docs/sglang/DESIGN.md)                   |
| Evaluation        | [docs/evaluation/DESIGN.md](docs/evaluation/DESIGN.md)           |
| Testing utilities | [docs/testing/DESIGN.md](docs/testing/DESIGN.md)                 |
| Profiling         | [docs/profiling/DESIGN.md](docs/profiling/DESIGN.md)             |
| Plugins           | [docs/plugins/DESIGN.md](docs/plugins/DESIGN.md)                 |
| Utils             | [docs/utils/DESIGN.md](docs/utils/DESIGN.md)                     |

## 🎯 Architecture

The system follows a modular, event-driven architecture:

```
Dataset Manager ──► Load Generator ──► Endpoint Client ──► External Endpoint
                          │
                    Metrics Collector
                 (event logging + reporting)
```

- **Dataset Manager**: Loads benchmark datasets and applies transform pipelines
- **Load Generator**: Central orchestrator — controls timing (scheduler), issues queries, and emits sample events
- **Endpoint Client**: Multi-process HTTP worker pool communicating over ZMQ IPC
- **Metrics Collector**: Receives sample events from Load Generator; writes to SQLite (EventRecorder), aggregates after the run (MetricsReporter)

## Accuracy Evaluation

You can run accuracy evaluation with Pass@1 scoring by specifying accuracy datasets in the benchmark
configuration. Currently, Inference Endpoints provides the following pre-defined accuracy benchmarks:

- GPQA (default: GPQA Diamond)
- AIME (default: AIME 2025)
- LiveCodeBench (default: lite, release_v6)

However, LiveCodeBench will not work out-of-the-box and requires some additional setup. See the
[LiveCodeBench](src/inference_endpoint/evaluation/livecodebench/README.md) documentation for
details and explanations.

## 🚧 Pending Features

The following features are planned for future releases:

- [ ] **Submission Ruleset Integration** - Full MLPerf submission workflow support
- [ ] **Documentation Generation and Hosting** - Sphinx-based API documentation with GitHub Pages

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/DEVELOPMENT.md) for details on:

- Setting up your development environment
- Code style and quality standards
- Testing requirements
- Pull request process

## 🙏 Acknowledgements

This project draws inspiration from and learns from the following excellent projects:

- [MLCommons Inference](https://github.com/mlcommons/inference) - MLPerf Inference benchmark suite
- [AIPerf](https://github.com/ai-dynamo/aiperf) - AI model performance profiling framework
- [SGLang GenAI-Bench](https://github.com/sgl-project/genai-bench) - Token-level performance evaluation tool
- [vLLM Benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) - Performance benchmarking tools for vLLM
- [InferenceMAX](https://github.com/InferenceMAX/InferenceMAX) - LLM inference optimization toolkit

We are grateful to these communities for their contributions to LLM benchmarking and performance analysis.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.md) file for
details.

## 🔗 Links

- [MLCommons](https://mlcommons.org/) - Machine Learning Performance Standards
- [Project Repository](https://github.com/mlcommons/endpoints)
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference/)

## 👥 Contributors

Credits to core contributors of the project:

- MLCommons Committee
- NVIDIA: Zhihan Jiang, Rashid Kaleem, Viraat Chandra, Alice Cheng
- ...

See [ATTRIBUTION](ATTRIBUTION) for detailed attribution information.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mlcommons/endpoints/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlcommons/endpoints/discussions)
- **Documentation**: See [docs/](docs/) directory for guides
