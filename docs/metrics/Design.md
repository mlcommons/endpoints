# Metrics — Design Spec

> Records per-sample timing events to SQLite during a run (write path), then aggregates them into QPS, latency percentiles, TTFT, and TPOT after the run (read path).

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · [dataset_manager](../dataset_manager/Design.md) · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · **metrics** · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`metrics/` records benchmark events during a run and aggregates them into performance metrics
afterwards. It is split into two parts with a clean boundary: `EventRecorder` writes; nothing
else does. `MetricsReporter` reads.

## Responsibilities

- Persist every timing event to SQLite during the run (write path)
- Aggregate events into QPS, latency percentiles, TTFT, and TPOT after the run (read path)
- Validate results against metric targets from the active ruleset
- Produce human-readable console output and machine-readable JSON reports

## Component Map

```
SampleEventHandler  ──► EventRecorder (SQLite, queue-backed)
                                │
                                ▼
                        MetricsReporter
                                │
                    ┌───────────┴──────────┐
                    ▼                      ▼
              console output          JSON report
```

## Public Interface

### `EventRecorder`

Only one `EventRecorder` may be actively writing at a time per process. The live instance is
accessible via the class variable `EventRecorder.LIVE`.

```python
class EventRecorder:
    LIVE: "EventRecorder | None"   # class variable; set on construction, cleared on close

    @classmethod
    def record_event(
        cls,
        ev_type: Event,
        timestamp_ns: int,
        sample_uuid: str = "",
        force_commit: bool = False,
        assert_active: bool = True,
        data: Any = None,
    ) -> bool

    def wait_for_writes(self, force_commit: bool = True) -> None
    # Blocks until the background writer thread has flushed all queued events

    @staticmethod
    def db_path(session_id: str) -> Path
```

`record_event()` is a **classmethod** and is **non-blocking**: events are placed on a queue and
written by a background thread. Returns `True` if recorded, `False` if no recorder is active
(when `assert_active=False`).

### `MetricsReporter`

```python
class MetricsReporter:
    def __init__(
        self,
        connection_name: os.PathLike,
        client_type: str = "duckdb",
    ) -> None

    def create_report(
        self,
        tokenizer: Tokenizer | None = None,
        tpot_reporting_mode: TPOTReportingMode = TPOTReportingMode.REQUEST_WEIGHTED,
    ) -> Report

    def dump_to_json(self, json_path: Path) -> None
```

`create_report()` executes SQL aggregation over the events database and returns a `Report`
object. The optional `tokenizer` enables output sequence length (OSL) computation. `Report`
itself exposes `display(fn=print, summary_only=False)` for console output.

### Metric Types (`metric.py`)

```python
class Throughput(Metric):
    REL_TOL = 0.1                  # ±10% relative tolerance
    def __init__(self, target_qps: float): ...   # stored as self.target

class QueryLatency(Metric):
    REL_TOL = 0.1
    def __init__(self, target_latency_ms: float | None = None,
                 target_qps: float | None = None): ...

class TTFT(Metric):
    def __init__(self, max_ttft_latency_ms: float): ...  # hard ceiling

class TPOT(Metric):
    def __init__(self, max_tpot_latency_ms: float): ...  # hard ceiling
```

Each metric exposes `is_valid(measurement) -> bool`. The target value is stored as
`self.target` on the base `Metric` class.

## Data Flow

### Write Path (during run)

```
SampleEventHandler.query_result_complete(result)
  → EventRecorder.record_event(
        SampleEvent.COMPLETE,
        time.monotonic_ns(),
        sample_uuid=result.id,
        data={...},
    )
  → queue.put(EventRow(...))               # non-blocking
  → background thread: INSERT INTO events
```

### Read Path (after run)

```
MetricsReporter.create_report()
  → SELECT / GROUP BY on events table (DuckDB)
  → compute percentiles (p50, p90, p99, p999)
  → compute TTFT  = time from LOADGEN_ISSUE_CALLED to FIRST_CHUNK
  → compute TPOT  = (COMPLETE.ts - FIRST_CHUNK.ts) / output_tokens
  → compute tracked duration from TEST_STARTED / STOP_PERFORMANCE_TRACKING windows
  → compute QPS   = tracked completed samples / tracked duration
  → validate each metric against RuntimeSettings.reported_metrics
```

## Design Decisions

**SQLite as the event store**

SQLite gives durable, queryable storage with no external dependencies. The write path uses a
single background writer thread (SQLite's WAL mode is single-writer) to avoid contention.
Aggregation uses DuckDB for columnar SQL performance over the file written by SQLite.

**Queue-backed writes to decouple hot path**

The `record_event()` call must not block the load generator thread. Events are placed on a
`queue.Queue` and consumed by a dedicated writer thread. The queue is unbounded; back-pressure
is not a concern because write throughput (SQLite) exceeds event rate in all tested scenarios.

**Singleton enforcement**

Only one `EventRecorder` may exist per process. The singleton is enforced at construction time
with a class-level flag. This prevents double-counting if code accidentally constructs a second
recorder.

**TPOT calculation modes**

TPOT can be weighted by request (each request contributes equally) or by output token count
(each token contributes equally). The default is request-weighted. The `TPOTReportingMode` enum
controls this at report time without re-running the benchmark.

## Integration Points

| Component                    | Role                                                      |
| ---------------------------- | --------------------------------------------------------- |
| `load_generator/sample.py`   | Calls `record_event()` for every state transition         |
| `load_generator/session.py`  | Calls `create_report()` at run end; saves output          |
| `config/runtime_settings.py` | `reported_metrics` list drives which metrics are computed |
| `config/ruleset_base.py`     | Provides `Metric` targets for validation                  |
