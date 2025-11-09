# Development Guide

This guide provides everything you need to contribute to the MLPerf Inference Endpoint Benchmarking System.

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.12+ (Python 3.12 is recommended for optimal performance)
- **Git**: Latest version
- **Virtual Environment**: Python venv or conda
- **IDE**: VS Code, PyCharm, or your preferred editor

### Development Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/mlperf/inference-endpoint.git
cd inference-endpoint

# 2. Create virtual environment (Python 3.12+ required)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e .
pip install -r requirements/dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify installation
inference-endpoint --version
pytest --version
```

## 🏗️ Project Structure

```
inference-endpoint/
├── src/inference_endpoint/     # Main package source
│   ├── cli.py                  # Command-line interface
│   ├── commands/               # CLI command implementations
│   ├── config/                 # Configuration and schema management
│   ├── core/                   # Core types and orchestration
│   ├── dataset_manager/        # Dataset handling and loading
│   ├── endpoint_client/        # HTTP/ZMQ endpoint communication
│   ├── load_generator/         # Load generation and scheduling
│   ├── metrics/                # Performance measurement and reporting
│   ├── openai/                 # OpenAI API compatibility
│   ├── profiling/              # Performance profiling tools
│   ├── runtime/                # Runtime configuration
│   ├── testing/                # Test utilities (echo server, etc.)
│   └── utils/                  # Common utilities
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── performance/            # Performance tests
│   └── datasets/               # Test datasets
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── requirements/               # Dependency management
└── scripts/                    # Utility scripts
```

## 🧪 Testing

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
- **Test Datasets** (`tests/datasets/`): Sample datasets for testing (dummy_1k.pkl, squad_pruned/)

### Writing Tests

```python
import pytest
from inference_endpoint.core.types import Query

class TestQuery:
    def test_query_creation(self):
        """Test creating a basic query."""
        query = Query(prompt="Test", model="test-model")
        assert query.prompt == "Test"
        assert query.model == "test-model"

    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations."""
        # Your async test here
        pass
```

## 📝 Code Quality

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
# Install hooks (done during setup)
pre-commit install

# Run all hooks on staged files
pre-commit run

# Run all hooks on all files
pre-commit run --all-files

# Skip hooks (use sparingly)
git commit --no-verify
```

### Code Formatting

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check formatting without changing files
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/

# Run mypy for type checking
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## 🔧 Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest
pre-commit run --all-files

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Component Development

When developing a new component:

1. **Create the component directory** in `src/inference_endpoint/`
2. **Add `__init__.py`** with component description
3. **Implement the component** following the established patterns
4. **Add tests** in the corresponding `tests/unit/` directory
5. **Update main package** `__init__.py` if needed
6. **Add dependencies** to appropriate `requirements/` files

### 3. Testing Strategy

- **Unit Tests**: >90% coverage required
- **Integration Tests**: Test component interactions
- **Performance Tests**: Ensure no performance regressions
- **Documentation**: Update docs for new features

## 📚 Documentation

### Writing Documentation

- **Code Comments**: Use docstrings for all public APIs
- **README Updates**: Update README.md for user-facing changes
- **API Documentation**: Document new interfaces and changes
- **Examples**: Provide usage examples for new features

### Documentation Standards

```python
def process_query(query: Query) -> QueryResult:
    """
    Process a query and return the result.

    Args:
        query: The query to process

    Returns:
        QueryResult containing the processed response

    Raises:
        QueryError: If the query cannot be processed

    Example:
        >>> query = Query(prompt="Hello")
        >>> result = process_query(query)
        >>> print(result.content)
        'Hello there!'
    """
    # Implementation here
    pass
```

## 🚀 Performance Considerations

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

## 🔍 Debugging

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

## 📦 Package Management

### Adding Dependencies

1. **Base Dependencies** (`requirements/base.txt`): Required for package to function
2. **Development Dependencies** (`requirements/dev.txt`): Development tools, linters, and pre-commit hooks
3. **Test Dependencies** (`requirements/test.txt`): Testing framework and utilities (pytest, pytest-asyncio, etc.)

### Updating Dependencies

```bash
# Update all dependencies
pip install --upgrade -r requirements/dev.txt

# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package-name
```

## 🚨 Troubleshooting

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

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run all checks** locally before submitting
6. **Create a PR** with clear description and tests
7. **Address review comments** promptly

### Commit Message Format

Use conventional commit format:

```
type(scope): description

feat(core): add query lifecycle management
fix(api): resolve endpoint connection issue
docs(readme): update installation instructions
test(loadgen): add performance benchmarks
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is adequate
- [ ] Documentation is updated
- [ ] Performance impact is considered
- [ ] Security implications are reviewed
- [ ] Error handling is appropriate

## 📞 Getting Help

- **Issues**: [GitHub Issues](https://github.com/mlperf/inference-endpoint/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlperf/inference-endpoint/discussions)
- **Documentation**: Check this guide and project docs
- **Team**: Reach out to the development team

## 🎯 Next Steps

1. **Set up your environment** using this guide
2. **Explore the codebase** to understand the architecture
3. **Pick a component** to work on from the project board
4. **Start with tests** to understand the expected behavior
5. **Implement incrementally** with regular testing
6. **Ask questions** when you need help

Happy coding! 🚀
