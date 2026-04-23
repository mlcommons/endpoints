# AGENTS.md

Guidelines for AI coding agents working in the MLPerf Inference Endpoint Benchmarking System repository.

## Project Overview

High-performance benchmarking tool for LLM inference endpoints targeting 50k+ QPS. Python 3.12+, Apache 2.0 licensed.

## Common Commands

```bash
# Development setup
python3.12 -m venv venv && source venv/bin/activate
pip install -e ".[dev,test]"
pre-commit install

# Testing
pytest                                        # All tests (excludes slow/performance)
pytest -m unit                                # Unit tests only
pytest -m integration                         # Integration tests only
pytest --cov=src --cov-report=html            # With coverage
pytest -xvs tests/unit/path/to/test_file.py  # Single test file

# Code quality — MUST run before every commit, no exceptions
pre-commit run --all-files

# Local testing with echo server
python -m inference_endpoint.testing.echo_server --port 8765
inference-endpoint probe --endpoints http://localhost:8765 --model test-model

# CLI usage
inference-endpoint benchmark offline --endpoints URL --model NAME --dataset PATH
inference-endpoint benchmark online --endpoints URL --model NAME --dataset PATH --load-pattern poisson --target-qps 100
inference-endpoint benchmark from-config --config config.yaml
```

## Architecture

### Core Data Flow

```
Dataset Manager --> Load Generator --> Endpoint Client --> External Endpoint
                        |
                   Metrics Collector (EventRecorder + MetricsReporter)
```

### Key Components

| Component           | Location                                                      | Purpose                                                                                                                                                                                                                          |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Load Generator**  | `src/inference_endpoint/load_generator/`                      | Central orchestrator: `BenchmarkSession` owns the lifecycle, `Scheduler` controls timing, `LoadGenerator` issues queries                                                                                                         |
| **Endpoint Client** | `src/inference_endpoint/endpoint_client/`                     | Multi-process HTTP workers communicating via ZMQ IPC. `HTTPEndpointClient` is the main entry point                                                                                                                               |
| **Dataset Manager** | `src/inference_endpoint/dataset_manager/`                     | Loads JSONL, HuggingFace, CSV, JSON, Parquet datasets. `Dataset` base class with `load_sample()`/`num_samples()` interface                                                                                                       |
| **Metrics**         | `src/inference_endpoint/metrics/`                             | `EventRecorder` writes to SQLite, `MetricsReporter` reads and aggregates (QPS, latency, TTFT, TPOT)                                                                                                                              |
| **Config**          | `src/inference_endpoint/config/`, `endpoint_client/config.py` | Pydantic-based YAML schema (`schema.py`), `HTTPClientConfig` (single Pydantic model for CLI/YAML/runtime), `RuntimeSettings`                                                                                                     |
| **CLI**             | `src/inference_endpoint/main.py`, `commands/benchmark/cli.py` | cyclopts-based, auto-generated from `schema.py` and `HTTPClientConfig` Pydantic models. Flat shorthands via `cyclopts.Parameter(alias=...)`                                                                                      |
| **Async Utils**     | `src/inference_endpoint/async_utils/`                         | `LoopManager` (uvloop + eager_task_factory), ZMQ transport layer, event publisher                                                                                                                                                |
| **OpenAI/SGLang**   | `src/inference_endpoint/openai/`, `sglang/`                   | Protocol adapters and response accumulators for different API formats                                                                                                                                                            |
| **VideoGen**           | `src/inference_endpoint/videogen/`                               | Client adapter and dataset loader for MLPerf WAN2.2-T2V-A14B. `VideoGenAdapter` POSTs `VideoPathRequest` directly to trtllm-serve's `/v1/videos/generations` with `response_format=video_path`; server saves video to Lustre and returns path, avoiding large byte payloads. |

### Hot-Path Architecture

Multi-process, event-loop design optimized for throughput:

- `BenchmarkSession` thread schedules samples with busy-wait timing
- Worker processes (N instances) handle HTTP requests via ZMQ IPC
- Uses `eager_task_factory` and `uvloop` for minimal async overhead
- CPU affinity support (`cpu_affinity.py`) for performance tuning
- Custom HTTP connection pooling (`http.py`) with `httptools` parser

### CLI Modes

CLI is auto-generated from `config/schema.py` Pydantic models via cyclopts. Fields annotated with `cyclopts.Parameter(alias="--flag")` get flat shorthands; all other fields get auto-generated dotted flags (kebab-case).

- **CLI mode** (`offline`/`online`): cyclopts constructs `OfflineBenchmarkConfig`/`OnlineBenchmarkConfig` (subclasses in `config/schema.py`) directly from CLI args. Type locked via `Literal`. `--dataset` is repeatable with TOML-style format `[perf|acc:]<path>[,key=value...]` (e.g. `--dataset data.csv,samples=500,parser.prompt=article`). Full accuracy support via `accuracy_config.eval_method=pass_at_1` etc.
- **YAML mode** (`from-config`): `BenchmarkConfig.from_yaml_file()` loads YAML, resolves env vars, and auto-selects the right subclass via Pydantic discriminated union. Optional `--timeout`/`--mode` overrides via `config.with_updates()`.
- **eval**: Not yet implemented (raises `CLIError` with a tracking issue link)

### Config Construction & Validation

Both CLI and YAML produce the same subclass via Pydantic discriminated union on `type`:

```
CLI offline/online:  cyclopts → OfflineBenchmarkConfig/OnlineBenchmarkConfig → with_updates(datasets) → run_benchmark
YAML from-config:    from_yaml_file(path) → discriminated union → same subclass → run_benchmark
```

`OfflineBenchmarkConfig` and `OnlineBenchmarkConfig` (in `config/schema.py`) inherit `BenchmarkConfig`:

- `type`: locked via `Literal[TestType.OFFLINE]` / `Literal[TestType.ONLINE]`
- `settings`: `OfflineSettings` (hides load pattern) / `OnlineSettings`
- `submission_ref`, `benchmark_mode`: `show=False` on base class

Validation is layered:

1. **Field-level** (Pydantic): `Field(ge=0)` on durations, `Field(ge=-1)` on workers, `Literal` on `benchmark_mode`
2. **Field validators**: `workers != 0` check
3. **Model validator** (`_resolve_and_validate`): streaming AUTO resolution, model name from `submission_ref`, load pattern vs test type, cross-field duration check, duplicate datasets

### Load Patterns

- `max_throughput`: Offline burst (all queries at t=0)
- `poisson`: Fixed QPS with Poisson arrival distribution
- `concurrency`: Fixed concurrent requests

## Code Organization

```
src/inference_endpoint/
├── main.py                    # Entry point + CLI app: cyclopts app, commands, error formatter, run()
├── exceptions.py              # CLIError, ExecutionError, InputValidationError, SetupError
├── commands/                  # Command execution logic
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── cli.py             # benchmark_app: offline, online, from-config subcommands
│   │   └── execute.py         # Phased execution: setup/run_threaded/finalize + BenchmarkContext
│   ├── probe.py               # ProbeConfig + execute_probe()
│   ├── info.py                # execute_info()
│   ├── validate.py            # execute_validate()
│   └── init.py                # execute_init()
├── core/types.py              # APIType, Query, QueryResult, StreamChunk, QueryStatus (msgspec Structs)
├── load_generator/
│   ├── session.py             # BenchmarkSession - top-level orchestrator
│   ├── load_generator.py      # LoadGenerator, SchedulerBasedLoadGenerator
│   ├── scheduler.py           # Scheduler, timing strategies
│   ├── sample.py              # SampleEventHandler
│   └── events.py              # SessionEvent, SampleEvent enums
├── endpoint_client/
│   ├── http_client.py         # HTTPEndpointClient - main client interface
│   ├── worker.py              # Worker process implementation
│   ├── worker_manager.py      # Manages worker lifecycle
│   ├── http.py                # ConnectionPool, HttpRequestTemplate, raw HTTP
│   ├── http_sample_issuer.py  # Bridges load generator to HTTP client
│   ├── config.py              # HTTPClientConfig (single Pydantic model — CLI/YAML/runtime)
│   ├── adapter_protocol.py    # HttpRequestAdapter protocol
│   ├── accumulator_protocol.py # Response accumulation protocol
│   ├── cpu_affinity.py        # CPU pinning
│   └── utils.py               # Port range helpers
├── async_utils/
│   ├── loop_manager.py        # LoopManager (uvloop + eager_task_factory)
│   ├── runner.py              # run_async() — uvloop + eager_task_factory entry point for CLI commands
│   ├── event_publisher.py     # Async event pub/sub
│   ├── services/
│   │   ├── event_logger/      # EventLoggerService: writes EventRecords to JSONL/SQLite
│   │   └── metrics_aggregator/ # MetricsAggregatorService: real-time metrics (TTFT, TPOT, ISL, OSL)
│   └── transport/             # ZMQ-based IPC transport layer
│       ├── protocol.py        # Transport protocols + TransportConfig base
│       ├── record.py          # Transport records
│       └── zmq/               # ZMQ implementation (context, pubsub, transport, ZMQTransportConfig)
├── dataset_manager/
│   ├── dataset.py             # Dataset base class, DatasetFormat enum
│   ├── factory.py             # Dataset factory
│   ├── transforms.py          # ColumnRemap and other transforms
│   └── predefined/            # Built-in datasets (aime25, cnndailymail, gpqa, etc.)
├── metrics/
│   ├── recorder.py            # EventRecorder (SQLite-backed)
│   ├── reporter.py            # MetricsReporter (aggregation)
│   └── metric.py              # Metric types (Throughput, etc.)
├── config/
│   ├── schema.py              # Single source of truth: Pydantic models + cyclopts annotations
│   ├── runtime_settings.py    # RuntimeSettings dataclass
│   ├── ruleset_base.py        # BenchmarkSuiteRuleset base
│   ├── ruleset_registry.py    # Ruleset registry
│   ├── user_config.py         # UserConfig dataclass for ruleset user overrides
│   ├── rulesets/mlcommons/    # MLCommons-specific rules, datasets, models
│   └── templates/             # YAML config templates (_template.yaml minimal, _template_full.yaml all defaults)
├── openai/                    # OpenAI-compatible API types and adapters
│   ├── types.py               # OpenAI response types
│   ├── openai_adapter.py      # Request/response adapter
│   ├── openai_msgspec_adapter.py  # msgspec-based adapter (fast path)
│   ├── accumulator.py         # Streaming response accumulator
│   └── harmony.py             # openai_harmony integration
├── sglang/                    # SGLang API adapter
├── videogen/                  # WAN2.2 text-to-video MLPerf workload
│   ├── __init__.py
│   ├── types.py               # Pydantic: VideoPathRequest, VideoPathResponse, HealthResponse
│   ├── adapter.py             # VideoGenAdapter (HttpRequestAdapter) + VideoGenAccumulator (no-op)
│   └── dataset.py             # VideoGenDataset — loads MLPerf prompt text files
├── evaluation/                # Accuracy evaluation (extractor, scoring, livecodebench)
├── plugins/                   # Plugin system
├── profiling/                 # line_profiler integration, pytest plugin
├── testing/
│   ├── echo_server.py         # Local echo server for testing
│   ├── max_throughput_server.py # Max throughput test server
│   └── docker_server.py       # Docker-based server management
└── utils/
    ├── logging.py             # Logging setup
    ├── version.py             # Version info
    ├── dataset_utils.py       # Dataset utilities
    └── benchmark_httpclient.py # HTTP client throughput benchmarking utility

tests/
├── conftest.py                # Shared fixtures (echo/oracle servers, datasets, settings)
├── test_helpers.py            # Test utility functions
├── unit/                      # Unit tests (mirror src/ structure)
├── integration/               # Integration tests (real servers, end-to-end)
│   ├── endpoint_client/       # HTTP client integration tests
│   └── commands/              # CLI command integration tests
├── performance/               # Performance benchmarks (pytest-benchmark)
└── datasets/                  # Test data (dummy_1k.jsonl, squad_pruned/)
```

## Development Standards

### Code Style and Pre-commit Hooks

- **Formatter/Linter**: `ruff` (line-length 88, target Python 3.12)
- **Type checking**: `mypy` (via pre-commit)
- **Formatting**: `ruff-format` (double quotes, space indent)
- **License headers**: Required on all Python files (enforced by pre-commit hook `scripts/add_license_header.py`)
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `chore:`

### Pre-commit Hooks

All of these run automatically on commit:

- trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements
- `ruff` (lint + autofix) and `ruff-format`
- `mypy` type checking
- `prettier` for YAML/JSON/Markdown
- License header enforcement
- `regenerate-templates`: auto-regenerates YAML config templates from schema defaults when `schema.py`, `config.py`, or `regenerate_templates.py` change

**IMPORTANT: Always run `pre-commit run --all-files` before every commit.** Hooks may modify files (prettier, ruff-format, license headers). If files are modified, stage the changes and commit once. Never commit without running pre-commit first.

See [Development Guide](docs/DEVELOPMENT.md) for full setup and workflow details.

### Data Types & Serialization

- **Core types** (`Query`, `QueryResult`, `StreamChunk`): `msgspec.Struct` with `frozen=True`, `array_like=True`, `gc=False`, `omit_defaults=True`
- **Config types**: `pydantic.BaseModel` for validation
- **Enums**: `str, Enum` pattern for serializable enums (e.g., `LoadPatternType`, `APIType`)
- **Serialization**: `msgspec.json` for hot-path (ZMQ transport), `pydantic` for config

### Testing

**Coverage target**: >90% for all new code.

**Test markers:**

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests (may need servers)
@pytest.mark.slow           # Skip in CI
@pytest.mark.performance    # No timeout, skip in CI
@pytest.mark.run_explicitly # Only run when explicitly selected
```

**Async tests**: Use `@pytest.mark.asyncio` — strict mode is configured globally in `pyproject.toml` (`asyncio_mode = "strict"`). Do NOT pass `mode="strict"` to the marker — it's not a valid argument.

**Key fixtures** (defined in `tests/conftest.py`):

- `mock_http_echo_server` — real HTTP echo server on dynamic port
- `mock_http_oracle_server` — dataset-driven response server
- `dummy_dataset` — in-memory test dataset
- `hf_squad_dataset` — HuggingFace squad dataset
- `events_db` — pre-populated SQLite events database
- `max_throughput_runtime_settings`, `poisson_runtime_settings`, `concurrency_runtime_settings` — preset configs
- `clean_sample_event_hooks` — ensures event hooks are cleared between tests

**Test data**: `tests/datasets/dummy_1k.jsonl` (1000 samples), `tests/datasets/squad_pruned/`

### Performance Guidelines

These apply especially to code in the hot path (load generator, endpoint client, transport):

- **No `match` statements in hot paths** — use dict dispatch instead
- **Use `dataclass(slots=True)`** for frequently instantiated classes (or `msgspec.Struct`)
- **Prefer generators** over list comprehensions for large datasets
- **Minimize async suspends** in hot path code
- **Use `msgspec`** over `json`/`pydantic` for serialization in the data path
- **Connection pooling**: The HTTP client uses custom `ConnectionPool` with `httptools` parser — not `aiohttp`/`requests`
- **Event loop**: `uvloop` with `eager_task_factory` via `LoopManager`
- **IPC**: ZMQ-based transport between main process and worker processes

### Key Dependencies

| Package               | Purpose                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| `uvloop`              | Performance-optimized event loop                                                       |
| `httptools`           | Fast HTTP parser for custom connection pool                                            |
| `msgspec`             | Fast serialization for core types and ZMQ transport                                    |
| `pyzmq`               | ZMQ IPC between main process and workers                                               |
| `pydantic`            | Configuration validation                                                               |
| `cyclopts`            | CLI framework — auto-generates flags from Pydantic                                     |
| `duckdb`              | Data aggregation                                                                       |
| `transformers`        | Tokenization for OSL reporting                                                         |

### Files to NOT Modify

- `src/inference_endpoint/openai/openai_types_gen.py` — auto-generated, excluded from ruff/pre-commit
- `src/inference_endpoint/openai/openapi.yaml` — OpenAI API spec, excluded from pre-commit

## Keeping AGENTS.md Up to Date

**This file is the source of truth for AI agents working in this repo.** If it is stale or wrong, every AI-assisted session starts from a broken foundation.

### When to Update

Update AGENTS.md as part of any PR that includes a **significant refactor**, meaning:

- **Moved, renamed, or deleted modules/packages** — update the Code Organization tree and Key Components table
- **Changed architectural boundaries** (e.g., new IPC transport, replaced pydantic with msgspec for config) — update Architecture and Data Types sections
- **Added or removed CLI commands/subcommands** — update CLI Modes and Common Commands
- **Changed test infrastructure** (new fixtures, changed markers, new test directories) — update Testing section
- **Added or removed key dependencies** — update Key Dependencies table
- **Changed build/tooling** (new pre-commit hooks, changed ruff config, new CI steps) — update [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **Changed hot-path patterns** (new transport, changed serialization, new performance constraints) — update Performance Guidelines

### How to Update

1. Treat AGENTS.md changes as part of the refactor itself — include them in the same PR, not as a follow-up
2. Keep descriptions factual and concise — what exists and where, not aspirational design docs
3. If you add a new top-level module under `src/inference_endpoint/`, add it to both the Key Components table and the Code Organization tree
4. If you remove something, remove it from AGENTS.md — stale entries are worse than missing ones

### Reviewer Checklist

When reviewing PRs with significant structural changes, verify:

- [ ] Code Organization tree matches the actual directory structure post-merge
- [ ] Key Components table reflects any moved/renamed/new components
- [ ] No references to deleted files, classes, or modules remain

## Common AI Coding Pitfalls

Known failure modes when AI tools generate code for this project. Reference these during code review of AI-assisted PRs.

### Architecture & Design

- **Inventing abstractions that don't exist**: AI may introduce new base classes, registries, or factory patterns that don't match the existing architecture. This project uses concrete types and explicit wiring — check that new code follows existing patterns rather than imposing unfamiliar frameworks.
- **Misunderstanding the multi-process boundary**: The endpoint client uses separate worker _processes_ (not threads) communicating over ZMQ. AI-generated code often assumes shared memory, passes unpicklable objects across processes, or adds synchronization primitives (locks, semaphores) that don't work cross-process.
- **Confusing hot-path vs. cold-path**: AI tends to treat all code uniformly. Code in `load_generator/`, `endpoint_client/worker.py`, and `async_utils/transport/` is latency-critical. Pydantic validation, excessive logging, or try/except blocks in these paths will degrade throughput.

### Serialization & Types

- **Using the wrong serialization library**: This project uses `msgspec` for hot-path data and `pydantic` for config. AI frequently defaults to `json.dumps`/`json.loads`, `dataclasses`, or applies pydantic where msgspec is required. If the type is a `msgspec.Struct`, encode/decode with `msgspec.json`, not stdlib json.
- **Breaking msgspec Struct constraints**: Core types (`Query`, `QueryResult`, `StreamChunk`) are `frozen=True` with `array_like=True`. AI may try to mutate fields directly (must use `force_setattr`), add mutable default fields, or assume dict-like serialization when the wire format is array-based.
- **Adding `dataclass` where `msgspec.Struct` is expected**: If neighboring types use msgspec, new types in the same module should too. AI defaults to `@dataclass` out of habit.

### Async & Concurrency

- **Mixing sync and async incorrectly**: AI may `await` in a sync context, call blocking I/O inside an `async def`, or use `asyncio.run()` when a loop is already running (this project manages its own loops via `LoopManager`).
- **Creating new event loops**: Workers and the main process already have managed event loops. AI may call `asyncio.new_event_loop()` or `asyncio.run()` which conflicts with the existing `LoopManager`/`uvloop` setup.
- **Ignoring `eager_task_factory`**: The project uses Python 3.12's `eager_task_factory` for performance. AI-generated code that creates coroutines expecting lazy scheduling will behave differently than expected.

### Testing

- **Generating mock-heavy tests for integration scenarios**: This project has real echo/oracle server fixtures. AI tends to mock HTTP calls even when `mock_http_echo_server` or `mock_http_oracle_server` fixtures exist and should be used.
- **Missing test markers**: Every test function needs `@pytest.mark.unit`, `@pytest.mark.integration`, or another marker. AI-generated tests almost always omit markers, which breaks CI filtering.
- **Wrong asyncio marker**: Tests must use bare `@pytest.mark.asyncio` — strict mode is configured globally in `pyproject.toml`. Do NOT pass `mode="strict"` to the marker (it's not a valid argument and will cause errors). AI sometimes hallucinates this parameter.
- **Fabricating fixture names**: AI may invent fixtures that don't exist in `conftest.py`. Always check that referenced fixtures actually exist before using them.

### Code Style & Repo Conventions

- **Missing license headers**: Every Python file needs the Apache 2.0 SPDX header. AI never generates these — the pre-commit hook will add them, but be aware of this when reviewing diffs.
- **Importing removed or renamed modules**: After refactors, AI (working from stale context) may import old module paths. Always verify imports resolve to actual files.
- **Over-documenting**: AI generates verbose docstrings, inline comments explaining obvious code, and type annotations on trivial variables. This project prefers minimal comments — only where the _why_ isn't obvious from the code.
- **Adding backwards-compatibility shims**: If something was renamed or removed, AI may add re-exports, aliases, or deprecation wrappers. In this project, just delete the old thing and update all call sites.
- **Empty except blocks**: Every `except` block must contain either a comment explaining why the exception is ignored, or a logging statement. Bare `except: pass` without explanation is disallowed. AI often generates empty handlers — always add the reason.
- **No lazy imports**: All imports must be at the top of the file. Imports inside functions, methods, or conditional blocks (other than `TYPE_CHECKING`) are disallowed. The only exceptions are: (1) circular import avoidance with a documenting comment, (2) optional dependencies with a top-level try/except that sets the import to `None`, (3) security sandboxing code that intentionally restricts imports.

### Dependency & Environment

- **Adding new dependencies without justification**: AI may `pip install` or add imports for packages not in `pyproject.toml`. Any new dependency must be justified, added to the correct optional group, and pinned to an exact version (`==`). After adding a dependency, run `pip-audit` (included in `dev` extras) to verify it has no known vulnerabilities.
- **Using `requests`/`aiohttp` for HTTP**: This project has its own HTTP client (`endpoint_client/http.py`) using `httptools`. AI defaults to `requests` or `aiohttp` — these should not appear in production code (test dependencies are fine).
