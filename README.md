# MLPerf Inference Endpoint Benchmarking System

A high-performance benchmarking tool for LLM endpoints with 50k QPS capability.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
# TODO: It's not here yet
git clone https://github.com/mlperf/inference-endpoint.git
cd inference-endpoint

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
inference-endpoint probe --endpoint http://your-endpoint:8000

# Run offline benchmark (max throughput)
inference-endpoint benchmark offline \
  --endpoint http://your-endpoint:8000 \
  --dataset tests/datasets/dummy_1k.pkl \
  --duration 60

# Run online benchmark (sustained QPS)
inference-endpoint benchmark online \
  --endpoint http://your-endpoint:8000 \
  --dataset tests/datasets/dummy_1k.pkl \
  --qps 100 \
  --duration 60
```

### Local Testing

```bash
# Start local echo server
python -m inference_endpoint.testing.echo_server --port 8765 &

# Test with dummy dataset (included in repo)
inference-endpoint benchmark offline \
  --endpoint http://localhost:8765 \
  --dataset tests/datasets/dummy_1k.pkl \
  --duration 10

# Stop echo server
pkill -f echo_server
```

See [Local Testing Guide](docs/LOCAL_TESTING.md) for detailed instructions.

## 📚 Documentation

- [Development Guide](docs/DEVELOPMENT.md) - How to contribute and develop
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Component interfaces and usage
- [Performance Guide](docs/PERFORMANCE.md) - Optimization and tuning

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

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/DEVELOPMENT.md) for details on:

- Setting up your development environment
- Code style and quality standards
- Testing requirements
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [MLPerf](https://mlperf.org/) - Machine Learning Performance Standards
- [Project Issues](https://github.com/mlperf/inference-endpoint/issues)
- [Project Wiki](https://github.com/mlperf/inference-endpoint/wiki)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mlperf/inference-endpoint/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlperf/inference-endpoint/discussions)
- **Documentation**: [Project Wiki](https://github.com/mlperf/inference-endpoint/wiki)
