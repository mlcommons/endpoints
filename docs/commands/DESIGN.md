# Commands — Design Spec

> Thin execution layer that maps Cyclopts CLI inputs to concrete command handlers and benchmark execution code. It owns dispatch and user-facing command boundaries, not core benchmarking logic.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · **commands** · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

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

| Subcommand              | Entry point                 | Execution module                | Status                    |
| ----------------------- | --------------------------- | ------------------------------- | ------------------------- |
| `benchmark offline`     | `commands/benchmark/cli.py` | `commands/benchmark/execute.py` | Implemented               |
| `benchmark online`      | `commands/benchmark/cli.py` | `commands/benchmark/execute.py` | Implemented               |
| `benchmark from-config` | `commands/benchmark/cli.py` | `commands/benchmark/execute.py` | Implemented               |
| `probe`                 | `main.py`                   | `commands/probe.py`             | Implemented               |
| `info`                  | `main.py`                   | `commands/info.py`              | Implemented               |
| `validate-yaml`         | `main.py`                   | `commands/validate.py`          | Implemented               |
| `init`                  | `main.py`                   | `commands/init.py`              | Implemented               |
| `sysinfo from-config`   | `commands/sysinfo/cli.py`   | `sys_info/capture.py`           | Implemented               |
| `eval`                  | `main.py`                   | inline stub (`CLIError`)        | Reserved, not implemented |

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
  +-- sysinfo
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
  +-- if sys_info_capture is configured:
        write run_metadata.yml
        capture_system_info() → mlcflow (hardware + serving config)
        patch run_metadata.yml with serving config values
```

## System Info Capture

System info capture collects hardware/software details from one or more nodes and writes a structured JSON file for MLPerf inference submissions. It runs in two contexts:

- **Standalone** (`sysinfo from-config`): triggered manually, independent of any benchmark run.
- **Integrated** (`benchmark` finalization): triggered automatically after a benchmark if `sys_info_capture` is present in the config.

Both paths call `sys_info/capture.py::capture_system_info()` and produce the same output JSON. The integrated path additionally patches `run_metadata.yml` with serving configuration values extracted from the inference server's startup log.

### Config Reference (`SysInfoCaptureConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ssh_ids` | `list[str]` | — | **Required.** Nodes to collect hardware info from. Format: `user@host` or `user@host:port`. |
| `accelerator_backend` | `"cuda"` \| `"rocm"` \| `"none"` | — | **Required.** GPU backend on the target nodes. |
| `exclude_current_system` | bool | `false` | Skip the machine running this command; collect from `ssh_ids` only. |
| `skip_ssh_key_file` | bool | `false` | Assume SSH key auth is pre-configured (skips mlcflow key-file lookup). |
| `output_path` | str | `"."` | Output directory for the JSON file. Overridden by `report_dir` when set at the top level. |
| `node_config` | object | `null` | Optional function-based node groupings (Prefill/Decode/etc). Maps function names to lists of `{node_name, no_of_nodes}` entries. `node_name` is matched as a case-insensitive substring against the detected GPU model name. |
| `serving_node` | str | `null` | SSH target for the inference server (`user@host` or `user@host:port`). When set, the capture also SSHes into this node to extract serving configuration from the startup log. |
| `log_path` | str | `null` | Path to the vLLM server log **on the serving node**. Required when `serving_node` is set and serving config extraction is desired. |
| `endpoint_url` | str | `null` | Base URL of the running inference server. Passed to the mlcflow script, which probes it via HTTP to detect the serving framework (e.g. `"vLLM 0.9.0"`). |

### Capture Flow

```
capture_system_info(config, run_metadata_path=...)
  │
  └─ mlc.access("get-mlperf-multi-node-system-info", ...)
       │
       ├─ prehook: get,mlperf,single-node,system-info on local machine (node 0)
       │           skipped if exclude_current_system=true
       │
       ├─ preprocess(): for each ssh_id
       │    remote_run get,mlperf,single-node,system-info on remote node
       │    copy back: mlperf-system-info-single-node-{id}.json
       │
       ├─ preprocess(): if serving_node set
       │    remote_run get,mlperf,serving-config on serving node
       │    parse.py reads vLLM startup log from the top (local on serving node):
       │      tensor_parallel_size, pipeline_parallel_size,
       │      expert_parallel_size, max_num_seqs
       │      + framework name and version ("vLLM 0.9.0")
       │    copy back: serving_config.json
       │
       ├─ preprocess(): if endpoint_url set and framework not yet detected
       │    GET /version or /get_server_info → "vLLM 0.9.0" / "SGLang 0.4.2"
       │    sets MLC_MLPERF_SERVING_FRAMEWORK (HTTP probe takes priority over log)
       │
       └─ postprocess():
            merge per-node JSONs → mlperf-multi-node-system-info.json
            if serving_config.json present:
              patch run_metadata.yml config_summary with extracted values
              if serving framework not detected via HTTP: use serving_config.json framework field
```

**`run_metadata.yml` patching** only happens in the benchmark context. `capture_system_info` accepts an optional `run_metadata_path` argument; `finalize_benchmark` passes `ctx.report_dir / "run_metadata.yml"`, which has already been written before the capture call. The `config_summary` block fields (`tensor_parallel`, `pipeline_parallel`, `expert_parallel`, `batch`) are updated in-place; fields that could not be parsed remain `null`.

### Standalone Command (`sysinfo from-config`)

```bash
inference-endpoint sysinfo from-config -c examples/sysinfo_example.yaml
```

The YAML file has two top-level keys:

```yaml
report_dir: results/h100_sysinfo/   # output directory (optional)

system_info:
  ssh_ids:
    - root@ssh1:22    # prefill node 1
    - root@ssh2:22    # prefill node 2
    - root@ssh3:22    # decode node 1
    - root@ssh4:22    # decode node 2
    - root@ssh5:22    # decode node 3
    - root@ssh6:22    # decode node 4
    - root@ssh7:22    # decode node 5

  accelerator_backend: cuda
  exclude_current_system: true   # master node is orchestrator-only
  skip_ssh_key_file: false

  # serving_node: where the inference server process is running.
  # If multiple serving nodes exist, point to any one — all nodes are assumed
  # to run the same serving framework version.
  serving_node: root@ssh1:22
  log_path: /tmp/vllm.log        # path on serving_node where server output was redirected

  node_config:                   # optional: function-based node groupings
    Prefill:
      - node_name: NVIDIA H100   # case-insensitive substring of detected GPU model
        no_of_nodes: 2
    Decode:
      - node_name: NVIDIA H100
        no_of_nodes: 5
```

`report_dir` takes priority over `system_info.output_path` when both are set.

Output is written to `report_dir/mlperf-multi-node-system-info.json`.

### Integrated Benchmark Config

Add a `sys_info_capture` block to a benchmark YAML config. The `endpoint_url` field is auto-populated from `endpoint_config.endpoints[0]` if not explicitly set.

```yaml
sys_info_capture:
  ssh_ids:
    - root@ssh1:22
    - root@ssh2:22
    - root@ssh3:22
    - root@ssh4:22
    - root@ssh5:22
    - root@ssh6:22
    - root@ssh7:22
  accelerator_backend: cuda
  exclude_current_system: true
  skip_ssh_key_file: false
  serving_node: root@ssh1:22
  log_path: /tmp/vllm.log
  # endpoint_url is auto-populated from endpoint_config.endpoints[0] if not set
  node_config:
    Prefill:
      - node_name: NVIDIA H100
        no_of_nodes: 2
    Decode:
      - node_name: NVIDIA H100
        no_of_nodes: 5
```

### `node_config` Validation

When `node_config` is provided, the automations script enforces:
- Every `node_name` must match at least one probed node's GPU model string (case-insensitive substring). Unmatched names return an error.
- For each unique `node_name`, the total `no_of_nodes` across all function groups must not exceed the number of nodes of that type actually probed. Declaring more nodes than were SSHed into is an error.

### Error Handling

| Situation | Standalone (`sysinfo from-config`) | Integrated (benchmark) |
|-----------|-----------------------------------|------------------------|
| mlcflow script returns non-zero | `ExecutionError` propagates to CLI handler | Logged as `error` with retry hint; benchmark exits 0 |
| Unexpected exception | Propagates to CLI handler | Logged as `error` with exception type; benchmark exits 0 |
| SSH failure on a node | Logged as error inside script; other nodes continue | Same |
| Per-node JSON missing after SSH run | Logged as warning; node skipped | Same |
| `node_name` unmatched or count exceeds probed | `ExecutionError` | `ExecutionError` → logged as `error` |
| `serving_config.json` absent or unreadable | Logged as error; `run_metadata.yml` left unchanged | Same |

In the integrated path, `sys_info_capture` failures never abort the benchmark. `results.json` and `run_metadata.yml` are written before the capture call, so the benchmark output is complete regardless of capture outcome. The error log includes `report_dir` and a command to re-run capture manually.

---

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

The `eval` command is exposed in help output but still raises `CLIError` with a tracking issue
link. The benchmark
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
| `sys_info/`                 | Invokes mlcflow to collect hardware/software/serving info from nodes |
