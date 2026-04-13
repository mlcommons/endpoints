# Contributing to MLPerf Inference Endpoints

Welcome! We're glad you're interested in contributing. This project is part of
[MLCommons](https://mlcommons.org/) and aims to build a high-performance
benchmarking tool for LLM inference endpoints targeting 50k+ QPS.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style and Conventions](#code-style-and-conventions)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [MLCommons CLA](#mlcommons-cla)

## Ways to Contribute

- **Report bugs** — use the [Bug Report](https://github.com/mlcommons/endpoints/issues/new?template=100-bug-report.yml) template
- **Request features** — use the [Feature Request](https://github.com/mlcommons/endpoints/issues/new?template=200-feature-request.yml) template
- **Report performance issues** — use the [Performance Issue](https://github.com/mlcommons/endpoints/issues/new?template=300-performance.yml) template
- **Request dataset support** — use the [Dataset Integration](https://github.com/mlcommons/endpoints/issues/new?template=400-dataset-integration.yml) template
- **Improve documentation** — fix typos, clarify guides, add examples
- **Pick up an issue** — look for [`good first issue`](https://github.com/mlcommons/endpoints/labels/good%20first%20issue) or [`help wanted`](https://github.com/mlcommons/endpoints/labels/help%20wanted)
- **Review PRs** — thoughtful reviews are as valuable as code

## Development Setup

### Prerequisites

- Python 3.12+ (3.12 recommended)
- Git
- A Unix-like OS (Linux or macOS)

### Getting Started

```bash
# Fork and clone
git clone https://github.com/<your-username>/endpoints.git
cd endpoints

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install with dev and test extras
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Verify your setup
pytest -m unit -x --timeout=60
```

### Local Testing with Echo Server

```bash
# Start a local echo server
python -m inference_endpoint.testing.echo_server --port 8765

# Run a quick probe
inference-endpoint probe --endpoints http://localhost:8765 --model test-model
```

## Code Style and Conventions

### Formatting and Linting

We use [ruff](https://docs.astral.sh/ruff/) for formatting and linting, and
[mypy](https://mypy-lang.org/) for type checking. Pre-commit hooks enforce
these automatically.

```bash
# Run all checks manually
pre-commit run --all-files
```

### Key Conventions

- **Line length:** 88 characters
- **Quotes:** Double quotes
- **License headers:** Required on all Python files (auto-added by pre-commit)
- **Commit messages:** [Conventional commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `perf:`
- **Comments:** Only where the _why_ isn't obvious from the code. No over-documenting.

### Serialization

- **Hot-path data** (Query, QueryResult, StreamChunk): `msgspec.Struct` — encode/decode with `msgspec.json`, not stdlib json
- **Configuration**: `pydantic.BaseModel` for validation
- **Do not** use `dataclass` where neighboring types use `msgspec`

### Performance-Sensitive Code

Code in `load_generator/`, `endpoint_client/worker.py`, and `async_utils/transport/`
is latency-critical. In these paths:

- No `match` statements — use dict dispatch
- Minimize async suspends
- No pydantic validation or excessive logging
- Use `msgspec` over `json`/`pydantic` for serialization

## Testing

### Running Tests

```bash
# All tests (excludes slow/performance)
pytest

# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Single file
pytest -xvs tests/unit/path/to/test_file.py

# With coverage
pytest --cov=src --cov-report=html
```

### Test Markers

Every test function **must** have a marker:

```python
@pytest.mark.unit
@pytest.mark.asyncio  # strict mode is configured globally in pyproject.toml
async def test_something():
    ...
```

Available markers: `unit`, `integration`, `slow`, `performance`, `run_explicitly`

### Coverage

Target **>90% coverage** for all new code. Use existing fixtures from
`tests/conftest.py` (e.g., `mock_http_echo_server`, `mock_http_oracle_server`,
`dummy_dataset`) rather than mocking.

## Submitting Changes

### Branch Naming

```
feat/short-description
fix/short-description
docs/short-description
```

### Pull Request Process

1. **Create a focused PR** — one logical change per PR
2. **Fill out the PR template** — describe what, why, and how to test
3. **Ensure CI passes** — `pre-commit run --all-files` and `pytest -m unit` locally before pushing
4. **Link related issues** — use `Closes #123` or `Relates to #123`
5. **Expect review within 2-3 business days** — reviewers are auto-assigned based on changed files

### What We Look For in Reviews

- Does it follow existing patterns in the codebase?
- Are tests included and meaningful (not mock-heavy)?
- Is it focused — no unrelated refactoring or over-engineering?
- Does it avoid adding unnecessary dependencies?

### After Review

- Address feedback with new commits (don't force-push during review)
- Once approved, a maintainer will merge

## Issue Guidelines

### Before Filing

1. Search [existing issues](https://github.com/mlcommons/endpoints/issues) for duplicates
2. Use the appropriate issue template
3. Provide enough detail to reproduce or understand the request

### Issue Lifecycle

New issues are auto-added to our [project board](https://github.com/orgs/mlcommons/projects/57)
and flow through: **Inbox → Triage → Ready → In Progress → In Review → Done**

### Priority Levels

| Priority        | Meaning                            |
| --------------- | ---------------------------------- |
| **ShowStopper** | Drop everything — critical blocker |
| **P0**          | Blocks release or users            |
| **P1**          | Must address this cycle            |
| **P2**          | Address within quarter             |
| **P3**          | Backlog, nice to have              |

## MLCommons CLA

All contributors must sign the
[MLCommons Contributor License Agreement](https://mlcommons.org/membership/membership-overview/).
A CLA bot will check your PR automatically.

To sign up:

1. Visit the [MLCommons Subscription form](https://mlcommons.org/membership/membership-overview/)
2. Submit your GitHub username
3. The CLA bot will verify on your next PR

Pull requests from non-members are welcome — you'll be prompted to sign the CLA
during the PR process.

## Questions?

File an [issue](https://github.com/mlcommons/endpoints/issues). We aim to respond within a few business days.
