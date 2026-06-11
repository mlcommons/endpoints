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
  +-- if system_info is configured:
        write run_metadata.json
        capture_system_info() → mlcflow (hardware + serving config)
        patch run_metadata.json with serving config values
```

## System Info Capture

> **Requires the `sysinfo` optional dependency.** Install it with:
>
> ```bash
> # uv (recommended)
> uv sync --extra sysinfo
> # or pass --extra sysinfo directly to uv run, e.g.:
> uv run --extra sysinfo inference-endpoint benchmark from-config --config config.yaml
>
> # pip (from repo root)
> pip install -e ".[sysinfo]"
> ```
>
> If `mlc-scripts` is not installed and `system_info` is configured, the benchmark still completes and results are written first; system info capture is then attempted, fails with an error log, and the process exits 0.

System info capture collects hardware/software details from one or more nodes and writes a structured JSON file for MLPerf inference submissions. It runs in two contexts:

- **Standalone** (`sysinfo from-config`): triggered manually, independent of any benchmark run. See [Standalone Command (`sysinfo from-config`)](#standalone-command-sysinfo-from-config) below for a full example.
- **Integrated** (`benchmark from-config`): triggered automatically after a benchmark run completes if `system_info` is present in the config. For example:

  ```yaml
  name: "llama3.1-8b-vllm-perf-c1000"
  version: "1.0"
  type: "online"

  model_params:
    name: "meta-llama/Llama-3.1-8B-Instruct"
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 128

  datasets:
    - name: cnn_dailymail::llama3_8b
      type: performance
      samples: 13368
      parser:
        input: prompt

  settings:
    runtime:
      min_duration_ms: 600000
      max_duration_ms: 3600000
      scheduler_random_seed: 137
      dataloader_random_seed: 111
      n_samples_to_issue: 13368

    load_pattern:
      type: "concurrency"
      target_concurrency: 1000

    client:
      num_workers: 4

  endpoint_config:
    endpoints:
      - "http://localhost:11001"
    api_key: null

  report_dir: sglang_perf_c1000

  system_info:
    system_name: H100x8_SGLang
    ssh_ids:
      - user@inference-node
    accelerator_backend: cuda
    exclude_current_system: true
    skip_ssh_key_file: false
    serving_node: user@inference-node
    endpoint_url: http://localhost:11001
    serving_framework: sglang
  ```

  then running:

  ```bash
  inference-endpoint benchmark from-config --config config.yaml
  ```

  System info capture runs automatically at the end of the benchmark.

Both paths call `sys_info/capture.py::capture_system_info()` and produce the same output JSON (as per endpoints spec). Both also patch `run_metadata.json` with serving configuration values extracted from the inference server's startup log, if the file is present in `report_dir`. In integrated execution mode it is always present (written by the benchmark before capture runs); in standalone execution mode it is patched only if it already exists there.

### Config Reference (`SysInfoCaptureConfig`)

| Field                    | Type                                             | Description                                                                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `system_name`            | str                                              | **Required.** Name of the system under test (e.g. `"H100x8_vLLM"`). Used as the MLPerf submission system identifier.                                                                                                         |
| `ssh_ids`                | `list[str]`                                      | **Required.** Nodes to collect hardware info from. Format: `user@host` or `user@host:port`.                                                                                                                                  |
| `accelerator_backend`    | `"cuda"` \| `"rocm"` \| `"xpu"` \| `"none"`      | GPU backend on the target nodes. Default: `"none"`.                                                                                                                                                                          |
| `exclude_current_system` | bool                                             | Skip the machine running this command; collect from `ssh_ids` only. Default: `false`.                                                                                                                                        |
| `skip_ssh_key_file`      | bool                                             | Assume SSH key auth is pre-configured (skips mlcflow key-file lookup). Default: `false`.                                                                                                                                     |
| `node_config`            | object                                           | Optional function-based node groupings (Prefill/Decode/etc). Maps function names to lists of `{node_name, no_of_nodes}` entries. `node_name` is matched as a case-insensitive substring against the detected GPU model name. |
| `serving_node`           | str                                              | SSH target for the inference server (`user@host` or `user@host:port`). When set, the capture SSHes into this node and reads `/tmp/serving.log` to extract serving config. Server stdout/stderr **must** be redirected there. |
| `endpoint_url`           | str                                              | Base URL of the running inference server. Probed via HTTP to detect the serving framework name and version (e.g. `"vLLM 0.9.0"`).                                                                                            |
| `serving_framework`      | `"auto"` \| `"vllm"` \| `"sglang"` \| `"trtllm"` | Serving engine type used for startup log parsing. Default: `"auto"` (detected from the endpoint).                                                                                                                            |

### Capture Flow

System info capture is powered by the [mlperf-automations](https://github.com/mlcommons/mlperf-automations) project and orchestrates the following steps:

1. **Hardware collection** — SSHes into each node listed in `ssh_ids` and collects CPU, memory, GPU, networking, and OS information. If `exclude_current_system` is false, the local machine is also probed. Results are merged into a single `system_desc.json`.

2. **Serving config extraction** _(if `serving_node` is set)_ — SSHes into the inference server node and reads its startup log to extract parallelism settings (`tensor_parallel`, `pipeline_parallel`, `expert_parallel`) and batch size.

3. **Framework detection** _(if `endpoint_url` is set)_ — probes the live inference server via HTTP to detect the framework name and version (e.g. `"vLLM 0.9.0"`, `"SGLang 0.4.2"`). HTTP detection takes priority over the log-based result.

4. **Output** — always writes `system_desc.json` to the configured report directory. If a `run_metadata.json` is present in `report_dir`, it is also patched with the extracted serving config values (fields that could not be parsed remain `null`).

### Standalone Command (`sysinfo from-config`)

```bash
inference-endpoint sysinfo from-config -c examples/sysinfo_example.yaml
```

```yaml
report_dir: results/h100_sysinfo/ # output directory

system_info:
  system_name: H100x7_vLLM
  ssh_ids:
    - root@ssh1:22 # prefill node 1
    - root@ssh2:22 # prefill node 2
    - root@ssh3:22 # decode node 1
    - root@ssh4:22 # decode node 2
    - root@ssh5:22 # decode node 3
    - root@ssh6:22 # decode node 4
    - root@ssh7:22 # decode node 5

  accelerator_backend: cuda
  exclude_current_system: true # master node is orchestrator-only
  skip_ssh_key_file: false

  # serving_node: where the inference server process is running.
  # Server stdout/stderr must be redirected to /tmp/serving.log on that node.
  # If multiple serving nodes exist, point to any one — all nodes are assumed
  # to run the same serving framework version.
  serving_node: root@ssh1:22

  node_config: # optional: function-based node groupings
    Prefill:
      - node_name: NVIDIA H100 # case-insensitive substring of detected GPU model
        no_of_nodes: 2
    Decode:
      - node_name: NVIDIA H100
        no_of_nodes: 5
```

Output is written to `report_dir/system_desc.json`.

### `node_config` Validation

When `node_config` is provided, the automations script enforces:

- Every `node_name` must match at least one probed node's GPU model string (case-insensitive substring). Unmatched names return an error.
- For each unique `node_name`, the total `no_of_nodes` across all function groups must not exceed the number of nodes of that type actually probed. Declaring more nodes than were SSHed into is an error.

### Error Handling

In standalone execution mode (`sysinfo from-config`), any capture failure propagates as an error and exits non-zero.

In integrated execution mode (triggered at the end of a benchmark run), `system_info` failures never abort the benchmark — `results.json` and `run_metadata.json` are written before capture runs, so benchmark output is always complete. Capture errors are logged but the process exits 0.

**Two warnings not to ignore:**

- **SSH failure on a node** — the error is logged in the MLC output but capture continues with the remaining nodes. The resulting `system_desc.json` will be missing that node's hardware info. Always verify `system_desc.json` looks complete before submission.
- **Serving config unavailable** — if the serving node is unreachable, `run_metadata.json` will have empty serving config fields (`tensor_parallel`, `pipeline_parallel`, `batch`, etc.). Check the MLC log and re-run `sysinfo from-config` manually if needed.

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

| Dependency                  | Role                                                                 |
| --------------------------- | -------------------------------------------------------------------- |
| `main.py`                   | App definition, logging setup, global error handling                 |
| `config/`                   | Defines CLI/YAML schema models and config loading                    |
| `dataset_manager/`          | Loads performance and accuracy datasets                              |
| `endpoint_client/`          | Sends requests to endpoint workers                                   |
| `load_generator/session.py` | Runs the benchmark session                                           |
| `metrics/`                  | Aggregates and reports benchmark results                             |
| `evaluation/`               | Scores collected accuracy datasets during benchmark finalization     |
| `sys_info/`                 | Invokes mlcflow to collect hardware/software/serving info from nodes |
