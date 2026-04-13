# Development Guide

This guide covers the development setup and workflow for the MLPerf Inference Endpoint Benchmarking System. For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Getting Started

### Prerequisites

- **Python**: 3.12+ (3.12 recommended)
- **Git**: Latest version
- **OS**: Linux or macOS (Windows is not supported)

### Development Environment Setup

```bash
# 1. Fork https://github.com/mlcommons/endpoints on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/endpoints.git
cd endpoints

# 2. Add the upstream repo as a remote
git remote add upstream https://github.com/mlcommons/endpoints.git

# 3. Create virtual environment (Python 3.12+ required)
python3.12 -m venv venv
source venv/bin/activate

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
│   ├── performance/            # Performance benchmarks
│   └── datasets/               # Test data (dummy_1k.jsonl, squad_pruned/)
├── docs/                       # Documentation
├── examples/                   # Usage examples
└── scripts/                    # Utility scripts
```

## Testing

### Running Tests

```bash
# All tests (excludes slow/performance)
pytest

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Single file with verbose output
pytest -xvs tests/unit/path/to/test_file.py

# With coverage
pytest --cov=src --cov-report=html
```

### Test Markers

Every test function **must** have a marker:

```python
import pytest

@pytest.mark.unit
def test_something():
    ...

@pytest.mark.unit
@pytest.mark.asyncio  # strict mode is configured globally in pyproject.toml
async def test_async_something():
    ...
```

Available markers: `unit`, `integration`, `slow`, `performance`, `run_explicitly`

### Key Fixtures

Defined in `tests/conftest.py` — use these instead of mocking:

- `mock_http_echo_server` — real HTTP echo server on dynamic port
- `mock_http_oracle_server` — dataset-driven response server
- `dummy_dataset` — in-memory test dataset
- `events_db` — pre-populated SQLite events database

### Coverage

Target **>90% coverage** for all new code.

## Code Quality

### Pre-commit Hooks

All of these run automatically on commit:

- trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements
- `ruff` (lint + autofix) and `ruff-format`
- `mypy` type checking
- `prettier` for YAML/JSON/Markdown
- License header enforcement
- YAML template validation and regeneration

**IMPORTANT: Always run `pre-commit run --all-files` before every commit.** Hooks may modify files. If files are modified, stage the changes and commit once.

```bash
# Run all hooks
pre-commit run --all-files

# Install hooks (done during setup)
pre-commit install
```

### Code Style

- **Formatter/Linter**: `ruff` (line-length 88, target Python 3.12)
- **Type checking**: `mypy`
- **Formatting**: `ruff-format` (double quotes, space indent)
- **License headers**: Required on all Python files (auto-added by pre-commit)
- **Commit messages**: [Conventional commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`
- **Comments**: Only where the _why_ isn't obvious from the code

## Development Workflow

### Feature Development

```bash
# Sync your fork with upstream before starting
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch on your fork
git checkout -b feat/your-feature-name

# Make changes and test
pytest
pre-commit run --all-files

# Commit changes
git add <specific files>
git commit -m "feat: add your feature description"

# Push to your fork and open a PR against mlcommons/endpoints
git push origin feat/your-feature-name
```

### Branch Naming

```
feat/short-description
fix/short-description
docs/short-description
```

## YAML Config Templates

Config templates in `src/inference_endpoint/config/templates/` are auto-generated from schema defaults. When you change `config/schema.py`, regenerate them:

```bash
python scripts/regenerate_templates.py
```

The pre-commit hook auto-regenerates templates when `schema.py`, `config.py`, or `regenerate_templates.py` change. CI validates templates are up to date via `--check` mode.

Two variants are generated per mode (offline, online, concurrency):

- `_template.yaml` — minimal: only required fields + placeholders
- `_template_full.yaml` — all fields with schema defaults + inline `# options:` comments

## Package Management

### Adding Dependencies

Add dependencies to `pyproject.toml` (always pin to exact versions with `==`):

- **Runtime dependencies**: `[project.dependencies]`
- **Optional groups** (dev, test, etc.): `[project.optional-dependencies]`

After adding a dependency, run `pip-audit` (included in `dev` extras) to verify it has no known vulnerabilities.

```bash
pip install -e ".[dev,test]"
```

## Performance Considerations

Code in `load_generator/`, `endpoint_client/worker.py`, and `async_utils/transport/` is latency-critical. In these paths:

- No `match` statements — use dict dispatch
- Use `dataclass(slots=True)` or `msgspec.Struct` for frequently instantiated classes
- Minimize async suspends
- Use `msgspec` over `json`/`pydantic` for serialization
- The HTTP client uses custom `ConnectionPool` with `httptools` parser — not `aiohttp`/`requests`

## Debugging

```bash
# Run with verbose logging
inference-endpoint -v benchmark offline ...

# Run tests with stdout visible
pytest -xvs tests/unit/path/to/test.py

# Use Python debugger
python -m pdb -m pytest tests/unit/path/to/test.py
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/mlcommons/endpoints/issues)
- **Project Board**: [Q2 Board](https://github.com/orgs/mlcommons/projects/57)
- **Documentation**: See [docs/](.) directory for guides
