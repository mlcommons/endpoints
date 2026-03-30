# CLI Design: Config Loading, Validation, and Execution

How `BenchmarkConfig` gets built, validated, and used — from user input to benchmark execution.

For usage examples and flag reference, see [CLI_QUICK_REFERENCE.md](CLI_QUICK_REFERENCE.md).

## Command Structure

```
$ inference-endpoint -h
Usage: inference-endpoint COMMAND

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ benchmark      Run benchmarks (offline, online, from-config)                 │
│ eval           Run accuracy evaluation. (not yet implemented)                │
│ info           Show system information.                                      │
│ init           Generate config template.                                     │
│ probe          Test endpoint connectivity.                                   │
│ validate-yaml  Validate YAML configuration file.                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

| Subcommand              | Purpose                             | Config source        |
| ----------------------- | ----------------------------------- | -------------------- |
| `benchmark offline`     | Max throughput (all queries at t=0) | CLI flags → Pydantic |
| `benchmark online`      | Sustained QPS with load pattern     | CLI flags → Pydantic |
| `benchmark from-config` | Run from YAML file                  | YAML → Pydantic      |

Global options: `--version`, `-v` (INFO), `-vv` (DEBUG). Verbosity is handled by the meta-app, not `BenchmarkConfig`.

## Config Construction

```
              CLI                                          YAML
    ┌─────────┴─────────┐                          benchmark from-config
 offline              online                               │
    │                    │                                 │  yaml.safe_load(path)
    │  cyclopts builds   │  cyclopts builds                │  resolve_env_vars(data)
    │  OfflineBenchmark  │  OnlineBenchmark                │  TypeAdapter picks subclass
    │  Config            │  Config                         │  by type field
    │                    │                                 │
    │  inject --dataset  │  inject --dataset               │  optional --timeout/--mode
    │  via with_updates  │  via with_updates               │  via with_updates
    │                    │                                 │
    ▼                    ▼                                 ▼
  ┌────────────────────────────────────────────────────────────┐
  │  _resolve_and_validate() (model_validator)                 │
  └─────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
                  setup → execute → finalize
```

Both paths produce the **same subclass with the same defaults**. A YAML file with `type: offline` gets `OfflineBenchmarkConfig` — identical to what `benchmark offline` constructs.

### CLI path

1. **cyclopts constructs the subclass directly.** `OfflineBenchmarkConfig` / `OnlineBenchmarkConfig` are Pydantic models in `config/schema.py` with `@cyclopts.Parameter(name="*")`. cyclopts generates flags from their fields.

2. **Type locked at class level.** `OfflineBenchmarkConfig.type` is `Literal[TestType.OFFLINE]` — determined by subcommand, not user input.

3. **Datasets injected after construction.** `--dataset` strings are parsed by a `BeforeValidator` on the `datasets` field, then merged via `config.with_updates(datasets=...)`.

### YAML path

1. **`from_yaml_file(path)`** loads YAML, resolves `${VAR}` env vars on parsed values, then passes the dict to a Pydantic `TypeAdapter` with `Discriminator`.

2. **Auto-selects subclass.** `type: "offline"` → `OfflineBenchmarkConfig`, `type: "online"` → `OnlineBenchmarkConfig`, others → base `BenchmarkConfig`.

3. **Optional CLI overrides.** `--timeout` and `--mode` applied via `config.with_updates(...)` which re-runs validators.

### Why subclasses?

`OfflineBenchmarkConfig` and `OnlineBenchmarkConfig` exist in the schema (not just CLI) so both paths share them:

```
BenchmarkConfig (base — submission/eval fallback)
  ├── OfflineBenchmarkConfig  (type=OFFLINE, OfflineSettings)
  └── OnlineBenchmarkConfig   (type=ONLINE, OnlineSettings)
```

They provide:

- **Type safety** — `Literal` type field, impossible to mismatch
- **Unified defaults** — CLI and YAML get identical subclass behavior
- **Per-mode `--help`** — each subcommand shows only relevant flags

| Aspect                | `offline`         | `online`         |
| --------------------- | ----------------- | ---------------- |
| **Streaming default** | AUTO → OFF        | AUTO → ON        |
| **Settings class**    | `OfflineSettings` | `OnlineSettings` |

## Dataset Specification

`--dataset` is repeatable and accepts a string with TOML-style dotted paths:

```
--dataset [perf|acc:]<path>[,key=value...]
```

The first segment is the file path, optionally prefixed with `perf:` or `acc:` to set the dataset type (defaults to performance). Additional comma-separated `key=value` pairs set Dataset fields using dotted paths for nesting.

```bash
# Simple
--dataset data.pkl

# Accuracy dataset
--dataset acc:eval.jsonl

# With samples limit and column remap
--dataset data.csv,samples=500,parser.prompt=article

# With accuracy config
--dataset acc:eval.pkl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer

# Multiple datasets
--dataset perf:train.pkl --dataset acc:eval.pkl,accuracy_config.eval_method=pass_at_1 --mode both
```

Parser remaps use `parser.TARGET=SOURCE` — "rename my dataset's SOURCE column to TARGET". Valid targets are derived from `MakeAdapterCompatible` (`prompt`, `system`). Invalid targets are rejected at parse time. Invalid source columns are rejected at dataset load time.

Pydantic validates all fields: `extra="forbid"` on `Dataset` and `AccuracyConfig` catches typos like `--dataset data.pkl,samles=500`. Format is auto-detected from file extension.

The only YAML-only features are `submission_ref` and `benchmark_mode` (for official submissions).

## Validation

Validation is layered, executing in order:

```
 1. cyclopts        → required args? unknown flags?
 2. Pydantic fields → type coercion, ge/le constraints
 3. Sub-model validators:
    ├── RuntimeConfig._validate_durations    → max >= min duration
    ├── LoadPattern._validate_completeness   → poisson needs qps, concurrency needs target
    └── HTTPClientConfig._workers_not_zero   → num_workers != 0
 4. BenchmarkConfig._resolve_and_validate:
    ├── resolve defaults (name, streaming, model name from submission_ref)
    ├── load pattern type vs test type (offline→max_throughput, online→poisson/concurrency)
    ├── submission needs benchmark_mode
    └── duplicate dataset detection
 5. Runtime (execute.py) → files exist, endpoints reachable
```

Sub-models self-validate their own constraints. `BenchmarkConfig` only handles cross-model checks.

### Error formatting

Errors from cyclopts (missing args, unknown flags, Pydantic validation) go through `cli_error_formatter` in `config/utils.py`:

```
$ inference-endpoint benchmark offline
╭── Error ─────────────────────────────────────────────────────────────────────╮
│ Required: --dataset                                                          │
╰──────────────────────────────────────────────────────────────────────────────╯

$ inference-endpoint benchmark offline --endpoints x --model M --dataset D --workers abc
╭── Error ─────────────────────────────────────────────────────────────────────╮
│   settings.client.num_workers: Input should be a valid integer, unable to    │
│ string as an integer                                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The formatter resolves aliases (shows `--dataset` not `--endpoint-config.endpoints`) and strips Pydantic boilerplate.

## Error Handling

```
Exception Class         Exit Code   When
─────────────────────   ─────────   ─────────────────────────────────
InputValidationError    2           Bad user input, invalid config
SetupError              3           Dataset load failure, connection error
ExecutionError          4           Benchmark failed after setup
CLIError                1           Generic CLI error (base class)
NotImplementedError     1           Unimplemented command (eval)
```

## Development Guide

### Adding a CLI flag

Annotate the schema field — zero CLI code changes:

```python
class HTTPClientConfig(WithUpdatesMixin, BaseModel):
    buffer_size: Annotated[
        int,
        cyclopts.Parameter(alias="--buffer-size", help="Socket buffer size"),
    ] = 4096
    # → --client.buffer-size AND --buffer-size
```

### Flag generation rules

- Dotted paths auto-generated in kebab-case from model hierarchy
- Shorthands explicit via `cyclopts.Parameter(alias=...)`
- Booleans get `--no-` negation
- `show=False` hides from `--help`

### Config modification

`BenchmarkConfig` is frozen. Use `with_updates()` to produce new instances with re-validation:

```python
config = config.with_updates(timeout=300, datasets=["new_data.pkl"])
```
