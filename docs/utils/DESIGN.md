# Utils — Design Spec

> Shared helpers (logging setup, version, tokenizer utilities) and a standalone HTTP benchmarking tool. The core helper modules have no dependencies on other project subpackages.

**Component specs:** [async_utils](../async_utils/DESIGN.md) · [commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) · [core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) · [endpoint_client](../endpoint_client/DESIGN.md) · [evaluation](../evaluation/DESIGN.md) · [load_generator](../load_generator/DESIGN.md) · [metrics](../metrics/DESIGN.md) · [openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) · [profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) · [testing](../testing/DESIGN.md) · **utils**

---

## Overview

`utils/` contains shared utilities that do not belong to any specific component. The core of this
package is a set of stateless helper modules with no cross-component dependencies.
`benchmark_httpclient.py` is a standalone benchmarking tool that lives here for convenience but
does import from other `inference_endpoint` subpackages.

## Files

| File                      | Purpose                                                                   |
| ------------------------- | ------------------------------------------------------------------------- |
| `logging.py`              | Configures the root logger (format, level, handlers)                      |
| `version.py`              | Exposes package version from `inference_endpoint.__version__` and git SHA |
| `dataset_utils.py`        | Tokenizer inspection utilities (vocab stats, token length histograms)     |
| `benchmark_httpclient.py` | Standalone HTTP throughput benchmarking utility (imports internals)       |

## Design Decisions

**No cross-imports from `utils/` helper modules**

`logging.py` and `dataset_utils.py` stay lightweight and broadly reusable. `version.py` is also
small, but it intentionally imports `inference_endpoint.__version__` and shells out to `git` to
report build metadata. `benchmark_httpclient.py` is exempt entirely: it is a standalone tool, not
a reusable helper.

**`benchmark_httpclient.py` is a standalone tool**

This module benchmarks the raw HTTP client throughput independent of the load generator and
scheduler. It is useful for diagnosing whether performance bottlenecks are in the client layer
or in the scheduling/coordination layer. For broader tuning guidance, see
[CLIENT_PERFORMANCE_TUNING.md](../CLIENT_PERFORMANCE_TUNING.md). It can be run directly:

```bash
uv run python -m inference_endpoint.utils.benchmark_httpclient --endpoint URL --workers 4
```

## Integration Points

| Consumer           | Usage                                        |
| ------------------ | -------------------------------------------- |
| `main.py`          | Calls `setup_logging()` at startup           |
| `commands/info.py` | Imports `__version__` for the `info` command |
