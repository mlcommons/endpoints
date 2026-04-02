# Plugins — Design Spec

> Reserved extension namespace for registering custom `HttpRequestAdapter`, `DatafileLoader`, and `BenchmarkSuiteRuleset` implementations without modifying core package code.

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · **plugins** · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`plugins/` is the extension point for adding custom adapters, dataset loaders, rulesets, or
other integrations without modifying core package code.

## Responsibilities

- Reserve a stable namespace for future plugin APIs
- Point readers to the concrete registries that exist today

## Current State

The plugin system is a reserved namespace. `plugins/__init__.py` currently contains only
placeholder documentation and does not yet expose a public registration interface. No built-in
plugins exist. When the first real plugin requirement arrives, the registration mechanism will
be implemented with a concrete use case.

## Design Decisions

**Namespace reservation over premature framework**

The `plugins/` directory signals that extensibility is a first-class concern, without committing
to a specific plugin discovery mechanism (e.g. `importlib.metadata` entry points, config-based
loading). When the first real plugin requirement arrives, the mechanism can be chosen with a
concrete use case.

## Integration Points

| Extensible component    | Current registration target                  |
| ----------------------- | -------------------------------------------- |
| `HttpRequestAdapter`    | `endpoint_client/config.py` adapter registry |
| `DatafileLoader`        | `dataset_manager/dataset.py` format registry |
| `BenchmarkSuiteRuleset` | `config/ruleset_registry.py`                 |
