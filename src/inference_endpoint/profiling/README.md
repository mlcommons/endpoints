# Profiling Guide

Profiling support for the HTTP client and worker processes using multiple profiling backends.

## Available Profilers

The system supports multiple profiling backends:

1. **line_profiler** - Line-by-line profiling with detailed per-line timing
2. **pyinstrument** - Statistical sampling profiler with low overhead
3. **yappi** - Multi-threaded deterministic profiler with thread/async support
4. **Event Loop Stats** - Event loop load monitoring (CPU vs I/O wait time)

Note: Only one code profiler (1-3) can be active at a time, but event loop stats (4) can run independently alongside any profiler.

## Quick Start

### line_profiler (Detailed Line-by-Line)

Best for: Understanding exactly which lines are slow

```bash
# Profile integration tests
ENABLE_LINE_PROFILER=1 pytest tests/integration/endpoint_client/

# Profile performance tests
ENABLE_LINE_PROFILER=1 pytest tests/performance/

# Profile specific test
ENABLE_LINE_PROFILER=1 pytest tests/integration/endpoint_client/test_http_client_core.py::test_basic_future_handling

# Profile your own script
ENABLE_LINE_PROFILER=1 python my_script.py
```

### pyinstrument (Low-Overhead Sampling)

Best for: Production-like profiling with minimal overhead

```bash
# Profile integration tests
ENABLE_PYINSTRUMENT=1 pytest tests/integration/endpoint_client/

# Profile performance tests
ENABLE_PYINSTRUMENT=1 pytest tests/performance/

# Profile your own script
ENABLE_PYINSTRUMENT=1 python my_script.py
```

### yappi (Multi-threaded Deterministic Profiler)

Best for: Multi-threaded applications, async code, and precise function-level timing

```bash
# Profile integration tests with wall-clock time
ENABLE_YAPPI=1 pytest tests/integration/endpoint_client/

# Profile performance tests with CPU time
ENABLE_YAPPI=1 pytest tests/performance/

# Profile your own script
ENABLE_YAPPI=1 python my_script.py

# Show all functions (not just @profile decorated)
ENABLE_YAPPI=1 python my_script.py
```

### Event Loop Stats (Asyncio Performance Analysis)

Best for: Understanding event loop bottlenecks, CPU vs I/O wait time, and async performance

```bash
# Monitor event loop performance in tests
ENABLE_LOOP_STATS=1 pytest tests/performance/

# Combine with code profiling
ENABLE_LOOP_STATS=1 ENABLE_YAPPI=1 pytest tests/performance/

# Profile your own script
ENABLE_LOOP_STATS=1 python my_script.py
```

Output includes:

- CPU busy time vs I/O wait time breakdown
- Event loop iterations per second
- Concurrency metrics (file descriptors, active tasks, ready queue depth)
- CPU usage (user vs system time)
- Context switches (voluntary = I/O waits, involuntary = preemptions)
- Network and disk I/O throughput
- Per-coroutine I/O wait attribution

Profiling statistics are automatically displayed/saved after tests complete.

## Configuration

### Environment Variables

#### line_profiler

- **`ENABLE_LINE_PROFILER`**: Set to `1` to enable (default: disabled)
- **`LINE_PROFILER_LOGFILE`**: Custom path for worker profile files (default: `/tmp/mlperf_client_profiles/profile`)

#### pyinstrument

- **`ENABLE_PYINSTRUMENT`**: Set to `1` to enable (default: disabled)
- **`PYINSTRUMENT_OUTPUT_DIR`**: Custom output directory (default: `pyinstrument_profiles/`)

#### yappi

- **`ENABLE_YAPPI`**: Set to `1` to enable (default: disabled)
- **`YAPPI_OUTPUT_DIR`**: Custom output directory (default: `yappi_profiles/`)

#### Event Loop Stats

- **`ENABLE_LOOP_STATS`**: Set to `1` to enable (default: disabled)
- **`EVENT_LOOP_STATS_LOGFILE`**: Custom path for worker stats files (default: `/tmp/mlperf_event_loop_stats/stats`)

## Usage in Code

### Adding Profiling to Functions

Use the `@profile` decorator for both sync and async functions:

```python
from inference_endpoint.profiling import profile

@profile
async def my_async_function(data):
    """This function will be profiled when any profiler is enabled."""
    result = await process_data(data)
    return result

@profile
def my_sync_function(data):
    """This function will also be profiled."""
    return compute_result(data)
```

**Important**:

- The decorator is a no-op when profiling is disabled (zero overhead)
- May need to explicitly call profiler_shutdown() if stats are not printing to stdout by default

```python
from inference_endpoint.profiling import is_enabled, profiler_shutdown

# Check if profiling is active
if is_enabled():
  print("Profiling is enabled for this process")

# Explicit profiler_shutdown for worker processes (called on signal handling)
def cleanup():
  profiler_shutdown()  # Writes stats to file/stderr before exit
```

### Main Process Profiles

Shows execution for the coordinating process:

- `HTTPEndpointClient.issue_query` - Query submission
- `HTTPEndpointClient._send_to_worker` - ZMQ message passing
- `FuturesHttpClient._handle_responses` - Response processing

### Worker Process Profiles

Shows execution for worker processes making HTTP requests:

- `Worker._main_loop` - Main worker loop
- `Worker._make_http_request` - HTTP request execution
- `Worker._handle_streaming_request` - Streaming responses
- `Worker._handle_non_streaming_request` - Non-streaming responses
- `Worker._parse_sse_line` - SSE parsing logic

### Known Issues

**Harmless shutdown warning** (can be ignored):

```
AttributeError: 'NoneType' object has no attribute 'monitoring'
```

This appears during Python interpreter shutdown when pytest tears down `sys.monitoring` before `line_profiler`'s C extension cleanup runs. It does **not** affect profiling accuracy or test results.

**Warning**:
line-profiler has limitations in profiling async-code (eg: it cannot tell how much time an await actually spent sleeping)
