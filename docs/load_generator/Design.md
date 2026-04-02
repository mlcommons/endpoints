# Load Generator — Design Spec

> Central orchestrator for a benchmark run: controls what samples to issue, when to issue them via pluggable schedulers, and routes completion events to the metrics recorder.

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · **load_generator** · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`load_generator/` is the central orchestrator for a benchmark run. It controls **what** to send
(dataset samples), **when** to send them (load pattern), and **how** to observe the results
(event hooks feeding the metrics recorder).

## Responsibilities

- Manage the full benchmark session lifecycle (start → run → drain → report)
- Implement timing strategies: max throughput, Poisson, fixed concurrency
- Emit structured events for every sample state transition
- Coordinate graceful shutdown with in-flight drain

## Component Map

```
BenchmarkSession                        ← top-level owner; runs on background thread
    └── SchedulerBasedLoadGenerator     ← iterates (sample_index, delay_ns) pairs
            ├── Scheduler               ← determines timing
            │     ├── MaxThroughputScheduler    (offline: all at t=0)
            │     ├── PoissonDistributionScheduler (online: exp inter-arrival)
            │     └── ConcurrencyScheduler       (online: fixed in-flight count)
            └── SampleIssuer (ABC)      ← sends the query; implemented by endpoint_client/
```

## Public Interface

### `BenchmarkSession`

```python
@classmethod
def start(
    cls,
    runtime_settings: RuntimeSettings,
    dataset: Dataset,
    sample_issuer: SampleIssuer,
    scheduler: Scheduler,
    *args,
    accuracy_datasets: list[Dataset] | None = None,
    load_generator_cls: type[LoadGenerator] = SchedulerBasedLoadGenerator,
    name: str | None = None,
    max_shutdown_timeout_s: float | None = None,
    report_dir: os.PathLike | None = None,
    tokenizer_override: AutoTokenizer | None = None,
    dump_events_log: bool = False,
) -> "BenchmarkSession"

def wait_for_test_end(self, timeout: float | None = None) -> bool
def stop(self) -> None
```

`start()` spawns the run thread immediately. `wait_for_test_end()` blocks the caller until the
session finishes or the timeout expires. `stop()` signals early termination.

### `SampleIssuer` (abstract base class — implemented externally)

```python
def start() -> None
def issue(sample: Sample) -> None
def shutdown() -> None
```

`SampleIssuer` is an `ABC`, not a structural protocol. `start()` and `shutdown()` have default
no-op implementations; subclasses must implement `issue()`. `issue()` must be non-blocking;
responses are delivered asynchronously via `SampleEventHandler`.

### `Scheduler` (base class)

```python
def __iter__(self) -> Iterator[tuple[int, int]]
# yields (sample_index, delay_ns)
```

Subclasses register themselves via `__init_subclass__(load_pattern=LoadPatternType.X)` and are
looked up at construction time.

## Data Flow

```
BenchmarkSession._run_test()
  │
  ├─ for (index, delay_ns) in SchedulerBasedLoadGenerator:
  │     busy_wait(delay_ns)
  │     sample = load_sample_data(index)
  │     SampleIssuer.issue(sample)           → async, fire-and-forget
  │
  └─ wait_for_drain()                        ← blocks until all in-flight complete
        │
        └─ SampleEventHandler routes completions:
              FIRST_CHUNK → recorder.record_event(SampleEvent.FIRST_CHUNK)
              COMPLETE    → recorder.record_event(SampleEvent.COMPLETE)
```

## Design Decisions

**Busy-wait for timing precision**

`SchedulerBasedLoadGenerator` uses a busy-wait loop (`while time.monotonic_ns() < target_ns`) for
inter-sample delays rather than `asyncio.sleep()` or `time.sleep()`. This achieves sub-millisecond
timing accuracy at high QPS without introducing event-loop latency. The trade-off is elevated CPU
usage on the scheduling thread during the run.

**Thread-based session, not async**

`BenchmarkSession._run_test()` runs on a `threading.Thread`, not a coroutine. The scheduler loop
is blocking by design — it must not yield to the event loop, which could introduce scheduling jitter.
The async event loop is owned by `HTTPEndpointClient`, not the load generator.

**`SampleEventHandler` singleton with registered hooks**

All sample-level events (FIRST_CHUNK, COMPLETE, etc.) route through a single global
`_SampleEventHandler`. Hooks are registered before the run starts and remain constant for its
duration. This eliminates per-sample dispatch overhead at runtime.

**`ConcurrencyScheduler` coordination via `threading.Condition`**

The concurrency scheduler blocks issuance when in-flight count reaches the target, then wakes
via a Condition notified by the COMPLETE hook. This provides back-pressure without polling.

## Event Types

| Event                       | Enum type      | Meaning                                 |
| --------------------------- | -------------- | --------------------------------------- |
| `TEST_STARTED`              | `SessionEvent` | Run begins                              |
| `STOP_PERFORMANCE_TRACKING` | `SessionEvent` | Performance issuance phase has ended    |
| `LOADGEN_STOP`              | `SessionEvent` | Load generator finished issuing samples |
| `TEST_ENDED`                | `SessionEvent` | Run complete                            |
| `LOADGEN_ISSUE_CALLED`      | `SessionEvent` | `issue()` called                        |
| `LOADGEN_DATA_LOAD`         | `SessionEvent` | Sample payload loaded from dataset      |
| `HTTP_REQUEST_ISSUED`       | `SampleEvent`  | Request sent to endpoint                |
| `HTTP_RESPONSE_COMPLETED`   | `SampleEvent`  | Endpoint HTTP response fully received   |
| `FIRST_CHUNK`               | `SampleEvent`  | First SSE chunk received                |
| `NON_FIRST_CHUNK`           | `SampleEvent`  | Subsequent SSE chunk                    |
| `COMPLETE`                  | `SampleEvent`  | Final result received                   |

## Integration Points

| Dependency                   | Role                                                       |
| ---------------------------- | ---------------------------------------------------------- |
| `core/types.py`              | `Query`, `QueryResult`, `StreamChunk`                      |
| `endpoint_client/`           | Implements `SampleIssuer`                                  |
| `metrics/recorder.py`        | Receives all events via `SampleEventHandler`               |
| `config/runtime_settings.py` | `RuntimeSettings` drives duration, sample count, RNG seeds |
| `dataset_manager/`           | Provides `Dataset` for sample data                         |
