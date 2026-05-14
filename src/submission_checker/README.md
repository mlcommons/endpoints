# submission-checker

CLI tool for validating MLPerf Endpoints submissions against the §9.1 automated compliance checks.

## Installation

```bash
uv sync --extra dev
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Usage

### Check a submission

```bash
submission-checker check /path/to/submission
```

The tool expects the submission root to contain `systems/` and `pareto/` subdirectories as specified in §8.1.

**Options:**

| Flag | Description |
|------|-------------|
| `--strict` | Treat warnings as errors (exit 1 on any warning) |
| `--quiet` / `-q` | Suppress INFO-level passing checks |
| `--output FILE` / `-o FILE` | Write full results as JSON to *FILE* |

**Exit codes:** `0` = all checks passed, `1` = one or more errors (or warnings with `--strict`).

### Show region boundaries

```bash
submission-checker regions --max-concurrency 1024
```

Prints the concurrency ranges for each region given a declared Maximum Supported Concurrency *M* (§5.5).

## Submission structure

```
<org>/
├── systems/
│   └── <system_desc_id>.json      # §8.2 — hardware + software description
└── pareto/
    └── <system_desc_id>/
        └── <benchmark_model>/
            ├── runs/
            │   └── run_<N>.yaml   # §8.3 — one config per measurement point
            ├── results/
            │   └── run_<N>/
            │       └── mlperf_endpoints_log_summary.json
            └── accuracy/
                ├── accuracy.txt
                └── accuracy_result.json
```

## What gets checked

| Rule | Spec | Description |
|------|------|-------------|
| `path-exists` | §1 | Submission root directory exists |
| `required-dir` | §1 | `systems/` and `pareto/` present |
| `system-description-valid` | §1 | `systems/*.json` parses against schema |
| `src-dir` | §1 | `src/` present for Standardized submissions |
| `pareto-dir-exists` | §1 | `pareto/<system_id>/` directory exists |
| `pareto-subdir` | §1 | `runs/`, `results/`, `accuracy/` present |
| `measurement-runs-present` | §1 | At least one `run_*.yaml` found |
| `run-config-valid` | §1 | YAML parses against `RunConfig` schema |
| `run-filename-concurrency` | §1 | Filename concurrency matches declared value |
| `result-file-present` | §1 | Result log exists for each run config |
| `result-file-valid` | §1 | Result log parses against `RunSummary` schema |
| `run-count` | §2, §8 | 7–32 measurement runs |
| `run-cap` | §2, §8 | Run count does not exceed 32 |
| `low-latency-coverage` | §3 | At least one run in Low Latency region |
| `low-throughput-coverage` | §4 | At least one run in Low Throughput region |
| `med-throughput-coverage` | §5 | At least one run in Medium Throughput region |
| `high-throughput-coverage` | §6 | At least one run in High Throughput region |
| `max-concurrency-declared` | §7 | `max_supported_concurrency` field present |
| `region-computation` | §7 | *M* > 32 (required for region formula) |
| `concurrency-in-range` | §9 | Concurrency within region bounds (incl. 10% margin) |
| `load-pattern` | §10 | `load_pattern.type` is `concurrency` with `target_concurrency` set |
| `run-duration` | §11 | Run meets per-region minimum duration |
| `streaming-config` | §13 | `stream_all_chunks` is `True` |
| `metric-consistency-duration` | §14 | `duration_ns` > 0 |
| `metric-consistency-accounting` | §14 | `completed + failed == issued` |
| `metric-consistency-output-tokens` | §14 | `total_output_tokens` ≥ 0 |
| `accuracy-file` | §15 | `accuracy.txt` and `accuracy_result.json` present |
| `accuracy-valid` | §15 | `accuracy_result.json` parses correctly |
| `accuracy-gate` | §15 | Score ≥ quality target |
| `config-consistency-dataset` | §16 | All runs use the same dataset |
| `config-consistency-model` | §16 | Directory name matches `benchmark_model` |

## Programmatic API

```python
from submission_checker import SubmissionChecker, Report

checker = SubmissionChecker(Path("/submissions/acme_corp"))
report = checker.run()

if report.passed:
    print("All checks passed")
else:
    for result in report.errors:
        print(f"[{result.rule}] {result.message}")
```

The `Report` object also exposes `report.warnings` and serialises cleanly via `report.model_dump_json()`.

## Development

```bash
uv run pytest                                          # run tests (166 tests, 100% coverage)
uv run pytest --no-cov -x                             # fast fail on first error
uv run ruff check src/ tests/                         # lint
uv run ruff format src/ tests/                        # auto-format
uv run sphinx-build -W docs docs/_build/html          # build docs
```

## Architecture

```
cli.py          Entry point — Click commands, Rich table output
checker.py      SubmissionChecker — orchestrates loading and validation
loader.py       File I/O — JSON/YAML loading with structured error returns
models.py       Pydantic models — all rule logic lives in model_validators
structure.py    Directory structure validators (§8.1)
regions.py      Region boundary computation (§5.5 reference algorithm)
```

Validation logic is co-located with the data models: each Pydantic model runs its own `@model_validator` methods and accumulates results in a private `_check_results` list. `SubmissionChecker.run()` orchestrates loading, instantiates models, and collects results into a `Report`.
