# Config — Design Spec

> Parses YAML and CLI configuration, exposes ruleset extension points, and produces the immutable `RuntimeSettings` object that drives downstream components.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · **config** · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`config/` translates user-provided configuration (CLI arguments or YAML files) into validated
config models and, from those, an immutable `RuntimeSettings` object that drives the rest of the
system. It also provides the extension point for competition rulesets, though full ruleset-driven
runtime construction is still incomplete in the current execution path.

## Responsibilities

- Validate and parse YAML configuration with Pydantic
- Expose YAML templates for common benchmark patterns
- Define ruleset extension points (MLCommons and future competitions)
- Produce an immutable `RuntimeSettings` from validated config

## Component Map

```
CLI args / YAML file
        │
        ▼
  BenchmarkConfig  (Pydantic — schema.py)
        │
        ▼
  RuntimeSettings.from_config()
        │
        +── optional future ruleset integration
        ▼
  RuntimeSettings  (frozen dataclass)  ←─── drives all downstream components
```

## Key Types

### `BenchmarkConfig` (Pydantic model)

Top-level YAML schema. Most nested fields have defaults, but the top-level config still requires
the benchmark `type` and `endpoint_config`.

Key nested models:

| Model            | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `LoadPattern`    | Pattern type + parameters (target QPS, concurrency) |
| `RuntimeConfig`  | Duration, sample count, RNG seeds                   |
| `ClientSettings` | Worker count and HTTP client settings               |
| `EndpointConfig` | Endpoint URLs, API key                              |
| `Dataset`        | Dataset path, type (performance / accuracy)         |

### `RuntimeSettings` (frozen dataclass)

Immutable snapshot of all parameters needed to execute a run.

| Field                | Type           | Source                                  |
| -------------------- | -------------- | --------------------------------------- |
| `load_pattern`       | `LoadPattern`  | config                                  |
| `n_samples_to_issue` | `int`          | calculated: QPS × duration, or explicit |
| `min_duration_ms`    | `int`          | runtime config                          |
| `max_duration_ms`    | `int`          | runtime config                          |
| `min_sample_count`   | `int`          | current default / future ruleset hook   |
| `metric_target`      | `Metric`       | primary target driving scheduler logic  |
| `reported_metrics`   | `list[Metric]` | metrics validated after the run         |
| `rng_sched`          | `Random`       | seeded from `scheduler_random_seed`     |
| `rng_sample_index`   | `Random`       | seeded from `dataloader_random_seed`    |

Once constructed, `RuntimeSettings` cannot be modified. All consumers receive the same instance.

### `BenchmarkSuiteRuleset` (abstract base)

Extension point for competition-specific constraints.

```python
class BenchmarkSuiteRuleset(ABC):
    version: str

    @abstractmethod
    def apply_user_config(self, *args, **kwargs) -> RuntimeSettings:
        ...
```

Implementations override `apply_user_config()` to enforce minimum durations, sample counts,
required metrics, and fixed RNG seeds. The MLCommons ruleset lives in `rulesets/mlcommons/`.
That interface exists today, but `RuntimeSettings.from_config()` still uses the default conversion
path even when a ruleset object is supplied.

Rulesets are registered in `ruleset_registry.py` by name string (e.g. `"mlperf-inference-v5.1"`).

## Key Enums

| Enum              | Values                                     |
| ----------------- | ------------------------------------------ |
| `APIType`         | `OPENAI`, `SGLANG`                         |
| `LoadPatternType` | `MAX_THROUGHPUT`, `POISSON`, `CONCURRENCY` |
| `DatasetType`     | `PERFORMANCE`, `ACCURACY`                  |
| `TestMode`        | `PERF`, `ACC`, `BOTH`                      |
| `TestType`        | `OFFLINE`, `ONLINE`, `EVAL`, `SUBMISSION`  |
| `StreamingMode`   | `AUTO`, `ON`, `OFF`                        |

## YAML Templates

Pre-built templates are stored in `config/templates/`:

| Template                    | Use case                     |
| --------------------------- | ---------------------------- |
| `offline_template.yaml`     | Max-throughput offline run   |
| `online_template.yaml`      | Poisson online run           |
| `concurrency_template.yaml` | Fixed-concurrency online run |
| `eval_template.yaml`        | Accuracy evaluation          |
| `submission_template.yaml`  | Official MLPerf submission   |

Generated via `inference-endpoint init <name>`. `concurrency_template.yaml` exists on disk, but the
current `init` command exposes only `offline`, `online`, `eval`, and `submission`.

## Design Decisions

**Pydantic for config, frozen dataclass for runtime**

Pydantic is used at the boundary (file parsing, CLI parsing) where untrusted input arrives and
validation error messages matter. `RuntimeSettings` is a frozen dataclass because it carries no
untrusted data and must not change after construction. Using Pydantic in the hot path would add
unnecessary overhead.

**Ruleset is the intended strategy object**

The ruleset pattern keeps competition-specific constraints out of the core benchmark logic. Adding
a new ruleset requires only implementing `BenchmarkSuiteRuleset` and registering the class — no
changes to `BenchmarkSession` or the CLI. The abstraction is present, but the live benchmark path
has not fully delegated runtime construction to rulesets yet.

**Reproducibility via explicit seeds**

`RuntimeSettings` contains two seeded `Random` instances: one for scheduler timing jitter
(`rng_sched`) and one for dataset sample ordering (`rng_sample_index`). These make the runtime
configuration reproducible in principle, but the original seed values are not currently persisted
to the report output.

## Integration Points

| Consumer                        | Usage                                                        |
| ------------------------------- | ------------------------------------------------------------ |
| `load_generator/session.py`     | Receives `RuntimeSettings` at construction                   |
| `load_generator/scheduler.py`   | Reads `load_pattern`, `n_samples_to_issue`, RNG seeds        |
| `endpoint_client/config.py`     | Reads `api_type`, `num_workers`, streaming mode              |
| `metrics/reporter.py`           | Reads `reported_metrics`, duration bounds                    |
| `commands/benchmark/cli.py`     | Defines benchmark subcommands and resolves CLI vs YAML input |
| `commands/benchmark/execute.py` | Runs the benchmark lifecycle from resolved configuration     |
