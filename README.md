# MLPerf® Inference Endpoint Benchmarking System

A high-performance benchmarking tool for LLM endpoints.

## Quick Start

### Installation

**Requirements**: Python 3.12+ (Python 3.12 is recommended for optimal performance)

```bash
# Clone the repository
# Note: This repo will be migrated to https://github.com/mlcommons/endpoints
git clone https://github.com/mlcommons/endpoints.git
cd endpoints

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements/dev.txt

# Install pre-commit hooks
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
  --endpoint http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B

# Run offline benchmark (max throughput - uses all dataset samples)
inference-endpoint benchmark offline \
  --endpoint http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl

# Run online benchmark (sustained QPS - requires --target-qps)
inference-endpoint benchmark online \
  --endpoint http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --target-qps 100

# With explicit sample count
inference-endpoint benchmark offline \
  --endpoint http://your-endpoint:8000 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl \
  --num-samples 5000
```

### Local Testing

```bash
# Start local echo server
python -m inference_endpoint.testing.echo_server --port 8765 &

# Test with dummy dataset (included in repo)
inference-endpoint benchmark offline \
  --endpoint http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl

# Stop echo server
pkill -f echo_server
```

See [Local Testing Guide](docs/LOCAL_TESTING.md) for detailed instructions.

## 📚 Documentation

- [CLI Quick Reference](docs/CLI_QUICK_REFERENCE.md) - Command-line interface guide
- [Local Testing Guide](docs/LOCAL_TESTING.md) - Test with echo server
- [Development Guide](docs/DEVELOPMENT.md) - How to contribute and develop
- [GitHub Setup Guide](docs/GITHUB_SETUP.md) - GitHub authentication and setup

## 🎯 Architecture

The system follows a modular, event-driven architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset      │    │   Load          │    │   Endpoint      │
│   Manager      │───▶│   Generator     │───▶│   Client        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Metrics      │    │   Configuration │    │   Endpoint      │
│   Collector    │◄───│   Manager       │    │   (External)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

- **Load Generator**: Central orchestrator managing query lifecycle
- **Dataset Manager**: Handles benchmark datasets and preprocessing
- **Endpoint Client**: Abstract interface for endpoint communication
- **Metrics Collector**: Performance measurement and analysis
- **Configuration Manager**: System configuration (TBD)

## 🚧 Pending Features

The following features are planned for future releases:

- [ ] **Accuracy Evaluation** - Comprehensive accuracy metrics and validation
- [ ] **Performance Tuning** - Advanced performance optimization features
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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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
