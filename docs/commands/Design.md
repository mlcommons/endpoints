# Commands — Design Spec

> Thin execution layer that maps Cyclopts CLI inputs to concrete command handlers and benchmark execution code. It owns dispatch and user-facing command boundaries, not core benchmarking logic.

**Component specs:** [async_utils](../async_utils/Design.md) · **commands** · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

The command layer is split across:

- `main.py` for top-level app setup, global flags, simple commands, and error-to-exit-code handling
- `commands/benchmark/cli.py` for the `benchmark` subcommands (`offline`, `online`, `from-config`)
- `commands/benchmark/execute.py` for benchmark setup, execution, and finalization
- One module per simple command: `probe.py`, `info.py`, `validate.py`, `init.py`

Cyclopts constructs typed config objects directly from CLI arguments, so command functions receive
already-parsed models rather than raw `argparse.Namespace` objects.

## Responsibilities

- Register CLI commands and subcommands
- Translate typed CLI inputs into command execution calls
- Keep benchmark execution flow separate from CLI declaration
- Surface validation, setup, execution, and CLI errors through stable exit codes

## Command Map

| Subcommand              | Entry point                 | Execution module                    | Status                    |
| ----------------------- | --------------------------- | ----------------------------------- | ------------------------- |
| `benchmark offline`     | `commands/benchmark/cli.py` | `commands/benchmark/execute.py`     | Implemented               |
| `benchmark online`      | `commands/benchmark/cli.py` | `commands/benchmark/execute.py`     | Implemented               |
| `benchmark from-config` | `commands/benchmark/cli.py` | `commands/benchmark/execute.py`     | Implemented               |
| `probe`                 | `main.py`                   | `commands/probe.py`                 | Implemented               |
| `info`                  | `main.py`                   | `commands/info.py`                  | Implemented               |
| `validate-yaml`         | `main.py`                   | `commands/validate.py`              | Implemented               |
| `init`                  | `main.py`                   | `commands/init.py`                  | Implemented               |
| `eval`                  | `main.py`                   | inline stub (`NotImplementedError`) | Reserved, not implemented |

## CLI Structure

```
inference-endpoint
  |
  +-- global launcher in main.py
  |     - applies -v / --verbose
  |     - configures logging
  |     - dispatches into Cyclopts app
  |
  +-- benchmark
  |     +-- offline
  |     +-- online
  |     +-- from-config
  |
  +-- probe
  +-- info
  +-- validate-yaml
  +-- init
  +-- eval
```

`benchmark` is registered lazily from `commands/benchmark/cli.py`, keeping startup light for
simple commands like `info` and `validate-yaml`.

## `benchmark` Command Flow

```
CLI / YAML input
  |
  v
Cyclopts
  |
  +-- offline / online:
  |     construct OfflineBenchmarkConfig / OnlineBenchmarkConfig
  |     pass repeatable --dataset strings separately
  |
  +-- from-config:
  |     load YAML path
  |     BenchmarkConfig.from_yaml_file()
  |     optionally apply --timeout / --mode overrides
  |
  v
commands/benchmark/cli.py::_run()
  |
  +-- inject CLI dataset strings via config.with_updates(datasets=...)
  +-- normalize dataset validation errors
  |
  v
commands/benchmark/execute.py::run_benchmark()
  |
  +-- prepare report dir and runtime context
  +-- load datasets
  +-- construct endpoint client + sample issuer
  +-- run BenchmarkSession in threaded wrapper
  +-- finalize metrics and optional accuracy scoring
```

## `probe` Command

`probe` is a lightweight connectivity check built on the same endpoint/client stack as the main
benchmark path. It issues a small number of synthetic prompts, then reports success rate, latency,
and sample responses. Its purpose is to validate endpoint reachability and request formatting
before launching a full benchmark.

## Utility Commands

| Command         | What it does                                                   |
| --------------- | -------------------------------------------------------------- |
| `info`          | Prints local system and environment information                |
| `validate-yaml` | Loads a YAML config and runs schema validation                 |
| `init`          | Copies a config template from `config/templates/` into the cwd |

## Design Decisions

**Cyclopts models are the CLI boundary**

The command layer does not parse raw strings manually unless a flag is intentionally free-form,
such as repeatable `--dataset` values. Most arguments are parsed straight into Pydantic models
defined in `config/schema.py`, which keeps command handlers small and pushes field validation to
the schema layer.

**Benchmark declaration and execution are split**

`commands/benchmark/cli.py` owns subcommand shape and input normalization. `commands/benchmark/execute.py`
owns the multi-phase benchmark lifecycle. This keeps the CLI definition readable while allowing the
execution path to grow without turning the CLI module into orchestration code.

**Simple commands stay in `main.py` when they are thin**

Top-level commands with small signatures (`info`, `init`, `validate-yaml`, `probe`) are registered
directly in `main.py` and delegate immediately to their implementation modules. That keeps the app
topology visible in one place without introducing extra wrapper files.

**`eval` is intentionally reserved**

The `eval` command is exposed in help output but still raises `NotImplementedError`. The benchmark
path already supports dataset-specific accuracy evaluation, but the standalone `eval` command has
not been implemented yet.

## Integration Points

| Dependency                  | Role                                                             |
| --------------------------- | ---------------------------------------------------------------- |
| `main.py`                   | App definition, logging setup, global error handling             |
| `config/`                   | Defines CLI/YAML schema models and config loading                |
| `dataset_manager/`          | Loads performance and accuracy datasets                          |
| `endpoint_client/`          | Sends requests to endpoint workers                               |
| `load_generator/session.py` | Runs the benchmark session                                       |
| `metrics/`                  | Aggregates and reports benchmark results                         |
| `evaluation/`               | Scores collected accuracy datasets during benchmark finalization |
