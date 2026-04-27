# Profiling Guide

Line-by-line profiling for the HTTP client and worker processes using `line_profiler`.

## Quick Start

Enable profiling with the `ENABLE_LINE_PROFILER` environment variable:

```bash
# Profile integration tests
ENABLE_LINE_PROFILER=1 uv run pytest tests/integration/endpoint_client/

# Profile performance tests
ENABLE_LINE_PROFILER=1 uv run pytest tests/performance/

# Profile specific test
ENABLE_LINE_PROFILER=1 uv run pytest tests/integration/endpoint_client/test_http_client_core.py::test_basic_future_handling
```

Profiling statistics are automatically displayed after tests complete.

## Configuration

### Environment Variables

- **`ENABLE_LINE_PROFILER`**: Set to `1` to enable profiling (default: disabled)
- **`LINE_PROFILER_LOGFILE`**: Custom path for worker profile files (default: `/tmp/mlperf_client_profiles/profile`)

### Examples

```bash
# Use defaults (main to stderr, workers to /tmp/mlperf_client_profiles/)
ENABLE_LINE_PROFILER=1 uv run pytest tests/integration/

# Custom worker profile location
ENABLE_LINE_PROFILER=1 LINE_PROFILER_LOGFILE=/custom/path/profile uv run pytest tests/

# Profile your own script
ENABLE_LINE_PROFILER=1 uv run python my_script.py
```

## How It Works

The profiling system provides detailed line-by-line execution statistics:

1. **Main process** → writes stats to `stderr`
2. **Worker processes** → write stats to files (prevents interleaved output)
3. **pytest plugin** → aggregates and displays all results at test end

## Usage in Code

### Adding Profiling to Functions

Use the `@profile` decorator for both sync and async functions:

```python
from inference_endpoint.profiling import profile

@profile
async def my_async_function(data):
    """This function will be profiled when ENABLE_LINE_PROFILER=1."""
    result = await process_data(data)
    return result

@profile
def my_sync_function(data):
    """This function will also be profiled."""
    return compute_result(data)
```

**Important**:

- The decorator is a no-op when profiling is disabled (zero overhead).
- May need to explicitly call shutdown() if stats are not printing to stdout by default

```python
from inference_endpoint.profiling import is_enabled, shutdown

# Check if profiling is active
if is_enabled():
    print("Profiling is enabled for this process")

# Explicit shutdown for worker processes (called on signal handling)
def cleanup():
    shutdown()  # Writes stats to file/stderr before exit
```

### Main Process Profiles

Shows execution for the coordinating process:

- `HTTPEndpointClient.issue` - Query submission (fire-and-forget to worker via ZMQ)
- `HTTPEndpointClient.poll` / `recv` / `drain` - Response retrieval

### Worker Process Profiles

Shows execution for worker processes making HTTP requests:

- `Worker._run_main_loop` - Main worker event loop (request dispatch)
- `Worker._prepare_request` - Build `InFlightRequest` from `Query`
- `Worker._fire_request` - Send HTTP request and schedule response handling
- `Worker._handle_streaming_body` - SSE streaming response processing
- `Worker._handle_non_streaming_body` - Non-streaming response processing
- `Worker._iter_sse_lines` - SSE line parser / generator

### Known Issues

**Harmless shutdown warning** (can be ignored):

```
AttributeError: 'NoneType' object has no attribute 'monitoring'
```

This appears during Python interpreter shutdown when pytest tears down `sys.monitoring` before `line_profiler`'s C extension cleanup runs. It does **not** affect profiling accuracy or test results.

**Warning**:
line-profiler has limitations in profiling async-code (eg: it cannot tell how much time an await actually spent sleeping)
