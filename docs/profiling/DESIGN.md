# Profiling — Design Spec

> `line_profiler` integration with a zero-cost no-op decorator by default; activated via the `ENABLE_LINE_PROFILER=1` environment variable for line-level timing of hot-path functions.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · **profiling** · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`profiling/` integrates `line_profiler` into the benchmark run and provides a pytest plugin for
profiling during test execution. It is a developer tool with no effect on production runs unless
explicitly enabled.

## Responsibilities

- Wrap functions with `line_profiler.LineProfiler` for line-level timing
- Emit profiling output at the end of a benchmark or test run
- Provide a pytest plugin that activates profiling when `ENABLE_LINE_PROFILER=1` is set

## Files

| File                         | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `line_profiler.py`           | `profile()` decorator and `dump_stats()` utility |
| `pytest_profiling_plugin.py` | pytest plugin; hooks into test session lifecycle |

## Usage

```python
from inference_endpoint.profiling import profile

@profile
def hot_function(...):
    ...
```

When profiling is inactive (default), `@profile` is a no-op. When active, it wraps the function
with `LineProfiler` and accumulates timing across all calls.

In tests:

```bash
ENABLE_LINE_PROFILER=1 pytest tests/unit/...
```

## Design Decisions

**No-op decorator by default**

Importing `@profile` from `profiling/` is safe in production code. When profiling is not
enabled, the decorator returns the original function unchanged. This means profiling annotations
can remain in hot-path code without any runtime cost.

**`ENABLE_LINE_PROFILER` env var for selective activation**

Setting `ENABLE_LINE_PROFILER=1` activates profiling for the process in question. This avoids
permanently modifying the code; `@profile` annotations can remain in hot-path code without any
runtime cost when the env var is unset.

## Integration Points

| Consumer                              | Usage                                                         |
| ------------------------------------- | ------------------------------------------------------------- |
| `endpoint_client/`, `load_generator/` | `@profile` annotations on hot-path functions                  |
| pytest                                | Plugin registered via `pytest_plugins` in `tests/conftest.py` |
