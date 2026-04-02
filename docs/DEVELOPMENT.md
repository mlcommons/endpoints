# Development Guide

This guide provides everything you need to contribute to the MLPerf Inference Endpoint Benchmarking System.

## Getting Started

### Prerequisites

- **Python**: 3.12+ (Python 3.12 is recommended for optimal performance)
- **Git**: Latest version
- **Virtual Environment**: Python venv or conda
- **IDE**: VS Code, PyCharm, or your preferred editor

### Development Environment Setup

```bash
# 1. Fork https://github.com/mlcommons/endpoints on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/endpoints.git
cd endpoints

# 2. Add the upstream repo as a remote
git remote add upstream https://github.com/mlcommons/endpoints.git

# 3. Create virtual environment (Python 3.12+ required)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install development dependencies
pip install -e ".[dev,test]"

# 5. Install pre-commit hooks
pre-commit install

# 6. Verify installation
inference-endpoint --version
pytest --version
```

## Project Structure

```
endpoints/
├── src/inference_endpoint/     # Main package source
│   ├── main.py                 # Entry point and CLI app
│   ├── exceptions.py           # Project-wide exception types
│   ├── async_utils/            # Event loop, ZMQ transport, pub/sub
│   ├── commands/               # CLI command implementations
│   ├── config/                 # Configuration and schema management
│   ├── core/                   # Core types and orchestration
│   ├── dataset_manager/        # Dataset handling and loading
│   ├── endpoint_client/        # HTTP/ZMQ endpoint communication
│   ├── evaluation/             # Accuracy evaluation and scoring
│   ├── load_generator/         # Load generation and scheduling
│   ├── metrics/                # Performance measurement and reporting
│   ├── openai/                 # OpenAI API compatibility
│   ├── plugins/                # Plugin system
│   ├── profiling/              # Performance profiling tools
│   ├── sglang/                 # SGLang API adapter
│   ├── testing/                # Test utilities (echo server, etc.)
│   └── utils/                  # Common utilities
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   └── datasets/               # Test datasets
├── docs/                       # Documentation
├── examples/                   # Usage examples
└── scripts/                    # Utility scripts
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only (no timeout)

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_core_types.py

# Run with output to file (recommended)
pytest -v 2>&1 | tee test_results.log
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions with real servers
- **Performance Tests** (`tests/performance/`): Test performance characteristics (marked with @pytest.mark.performance, no timeout)
- **Test Datasets** (`tests/datasets/`): Sample datasets for testing (dummy_1k.jsonl, squad_pruned/)

### Writing Tests

```python
import pytest
from inference_endpoint.core.types import Query

class TestQuery:
    @pytest.mark.unit
    def test_query_creation(self):
        """Test creating a basic query."""
        query = Query(data={"prompt": "Test", "model": "test-model"})
        assert query.data["prompt"] == "Test"
        assert query.data["model"] == "test-model"

    @pytest.mark.asyncio(mode="strict")
    async def test_async_operation(self):
        """Test async operations."""
        # Your async test here
        pass
```

## Code Quality

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality.

Hooks that run automatically on commit:

- trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements
- `ruff` (lint + autofix) and `ruff-format`
- `mypy` type checking
- `prettier` for YAML/JSON/Markdown
- License header enforcement (Apache 2.0 SPDX header required on all Python files, added by `scripts/add_license_header.py`)

**Always run `pre-commit run --all-files` before committing.**

```bash
# Install hooks (done during setup)
pre-commit install

# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files
```

### Code Formatting

Configuration: `ruff` (line-length 88, target Python 3.12), `ruff-format` (double quotes, space indent).

```bash
# Format code with ruff
ruff format src/ tests/

# Check formatting without changing files
ruff format --check src/ tests/
```

### Linting

```bash
# Run ruff linter
ruff check src/ tests/

# Run mypy for type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## Development Workflow

### 1. Feature Development

```bash
# Sync your fork with upstream before starting
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch on your fork
git checkout -b feature/your-feature-name

# Make changes and test
pytest
pre-commit run --all-files

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork and open a PR against mlcommons/endpoints
git push origin feature/your-feature-name
```

### 2. Component Development

When developing a new component:

1. **Create the component directory** in `src/inference_endpoint/`
2. **Add `__init__.py`** with component description
3. **Implement the component** following the established patterns
4. **Add tests** in the corresponding `tests/unit/` directory
5. **Update main package** `__init__.py` if needed
6. **Add dependencies** to `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`

### 3. Testing Strategy

- **Unit Tests**: >90% coverage required
- **Integration Tests**: Test component interactions
- **Performance Tests**: Ensure no performance regressions
- **Documentation**: Update docs for new features

## Documentation

### Writing Documentation

- **Code Comments**: Add comments only where the _why_ is not obvious from the code; avoid restating what the code does
- **README Updates**: Update README.md for user-facing changes
- **Examples**: Provide usage examples for new features

## Performance Considerations

### Development Guidelines

- **Async First**: Use async/await for I/O operations
- **Memory Efficiency**: Minimize object creation in hot paths
- **Profiling**: Use pytest-benchmark for performance testing
- **Monitoring**: Add performance metrics for critical operations

### Performance Testing

```bash
# Run performance tests
pytest -m performance

# Run benchmarks
pytest --benchmark-only

# Compare with previous runs
pytest --benchmark-compare
```

## Debugging

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Test Failures**: Check test data and mock objects
3. **Performance Issues**: Use profiling tools to identify bottlenecks
4. **Async Issues**: Ensure proper event loop handling

### Debug Tools

```bash
# Run with debug logging
inference-endpoint --verbose

# Run tests with debug output
pytest -s -v

# Use Python debugger
python -m pdb -m pytest test_file.py
```

## Package Management

### Adding Dependencies

Add dependencies to `pyproject.toml` (always pin to exact versions with `==`):

- **Runtime dependencies**: `[project.dependencies]`
- **Optional groups** (dev, test, etc.): `[project.optional-dependencies]`

Install after updating:

```bash
pip install -e ".[dev,test]"
```

## Troubleshooting

### Common Problems

**Pre-commit hooks failing:**

```bash
# Update pre-commit
pre-commit autoupdate

# Skip hooks temporarily
git commit --no-verify
```

**Tests failing:**

```bash
# Clear Python cache
find . -type d -name "__pycache__" -delete
find . -type f -name "*.pyc" -delete

# Reinstall package
pip install -e .
```

**Import errors:**

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Ensure src is in path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Contributing Guidelines

### Pull Request Process

1. **Fork** `mlcommons/endpoints` on GitHub
2. **Clone your fork** and add `upstream` as a remote (see [Development Environment Setup](#development-environment-setup))
3. **Sync with upstream** (`git fetch upstream && git merge upstream/main`) before starting work
4. **Create a feature branch** on your fork (`git checkout -b feature/your-feature-name`)
5. **Make your changes** following the coding standards
6. **Add tests** for new functionality
7. **Update documentation** as needed
8. **Run all checks** locally: `pytest` and `pre-commit run --all-files`
9. **Push to your fork** and open a PR against `mlcommons/endpoints:main`
10. **Address review comments** promptly

### Commit Message Format

Use conventional commit format:

```
type(scope): description

feat(core): add query lifecycle management
fix(api): resolve endpoint connection issue
docs(readme): update installation instructions
test(loadgen): add performance benchmarks
```

Allowed types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `ci`.

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Security implications are reviewed
- [ ] Error handling is appropriate

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/mlcommons/endpoints/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlcommons/endpoints/discussions)
- **Documentation**: Check this guide and project docs
- **Team**: Reach out to the development team
