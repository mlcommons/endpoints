# Metrics Aggregator Service — Design Document

## Overview

The metrics aggregator receives `EventRecord` messages from a ZMQ SUB socket,
computes per-sample metrics in real time, and stores them in a `KVStore`.
Each metric is backed by its own mmap file on `/dev/shm` for lock-free
cross-process reads.

```
EventRecord (ZMQ SUB) → Event dispatch → SampleRow update → Trigger fires
                                                                  │
                                                                  ▼
                                                         kv_store.update(metric_name, value)
                                                                  │
                                                                  ▼
                                                         Per-metric mmap file on /dev/shm
```

## Module Layout

```
metrics_aggregator/
├── __init__.py
├── __main__.py          # CLI entry point
├── aggregator.py        # MetricsAggregatorService (thin event router)
├── kv_store.py          # KVStore ABC, BasicKVStore (mmap), BasicKVStoreReader
├── metrics_table.py     # SampleRow, TrackedBlock, MetricsTable, EmitTrigger, triggers
└── token_metrics.py     # TokenizePool (thread-pool tokenizer)
```

## Architecture

### Component Roles

| Component                    | Responsibility                                                                                                                                                                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MetricsAggregatorService** | Thin event router. Receives EventRecord batches, dispatches session events to `MetricsTable.handle_session_event()` and sample events to `MetricsTable.set_field()`. Manages counter keys (n_issued, n_completed, n_failed, duration_ns) directly on the KVStore. |
| **KVStore**                  | Abstract key-value store. Keys are either "counter" (single float) or "series" (append-only list). `BasicKVStore` backs each key with an individual mmap file on `/dev/shm`.                                                                                      |
| **MetricsTable**             | Owns sample rows, session state, trigger registry, tracked blocks. When `add_trigger()` is called, creates the metric key in the KVStore and wires the store onto the trigger.                                                                                    |
| **EmitTrigger**              | ABC for metric computations. Each trigger has a `metric_name` and `kv_store` reference (set by MetricsTable). `fire()` computes a value and calls `self.kv_store.update(self.metric_name, value)`.                                                                |
| **SampleRow**                | Pure data container (`msgspec.Struct, gc=False`). Holds per-sample timestamps.                                                                                                                                                                                    |
| **TrackedBlock**             | Per-tracking-window duration state.                                                                                                                                                                                                                               |

### KVStore Design

Each key maps to a `KVItem` backed by an individual file in `/dev/shm`:

- **Counter** (`_CounterItem`): 8-byte file, stores a single float64.
  Used for `n_samples_issued`, `n_samples_completed`, `n_samples_failed`, `duration_ns`.

- **Series** (`_SeriesItem`): Append-only file with layout:
  `[count: 8B uint64] [v0: 8B float64] [v1: 8B float64] ...`
  Used for `ttft_ns`, `sample_latency_ns`, `isl`, `osl`, `tpot_ns`, `chunk_delta_ns`.

Write protocol (single writer, aggregator process):

1. Write float64 at offset `HEADER + count * 8`
2. Update count at offset 0 (aligned 8-byte write, atomic on x86-64 TSO)

Read protocol (any process, via `BasicKVStoreReader`):

1. Read count from offset 0
2. Read values[last_read_idx : count] (only new values)
3. Compute rollup stats lazily (count, total, min, max, sum_sq) with incremental progress

```python
class KVStore(ABC):
    def create_key(self, key, type: "series" | "counter"): ...
    def update(self, key, value): ...
    def get(self, key): ...
    def snapshot(self): ...
    def close(self): ...
```

### Trigger System

Triggers are registered on `MetricsTable` at aggregator construction time.
Each trigger directly updates its metric in the KVStore — no emitter middleman:

```python
table = MetricsTable(kv_store)
table.add_trigger("recv_first_ns", TtftTrigger())
table.add_trigger("complete_ns", SampleLatencyTrigger())
table.add_trigger("complete_ns", OslTrigger(tokenize_pool, loop))
```

`add_trigger()`:

1. Calls `kv_store.create_key(trigger.metric_name, "series")`
2. Sets `trigger.kv_store = kv_store`
3. Registers the trigger on the field

When a trigger fires, it calls `self.kv_store.update(self.metric_name, value)`.
Sample UUID is not relevant — the KVStore only stores metric values.

### Session Counters

The aggregator manages counter keys directly (not via triggers):

```python
# In aggregator.__init__:
kv_store.create_key("n_samples_issued", "counter")
kv_store.create_key("n_samples_completed", "counter")
kv_store.create_key("n_samples_failed", "counter")
kv_store.create_key("duration_ns", "counter")

# In aggregator.process():
if ev == SampleEventType.ISSUED:
    self._n_issued += 1
    store.update("n_samples_issued", self._n_issued)
```

## Subscribed Events

### Session Events

| Event                        | Effect                                                    |
| ---------------------------- | --------------------------------------------------------- |
| `STARTED`                    | Records `session_started_ns` on MetricsTable              |
| `START_PERFORMANCE_TRACKING` | Sets `is_tracking = True`, opens a new `TrackedBlock`     |
| `STOP_PERFORMANCE_TRACKING`  | Sets `is_tracking = False`, updates `duration_ns` counter |
| `ENDED`                      | Triggers shutdown: drain in-flight tasks, finalize        |

### Sample Events

| Event            | Field Set                       | Trigger(s) Fired                                                     |
| ---------------- | ------------------------------- | -------------------------------------------------------------------- |
| `ISSUED`         | `issued_ns`                     | `IslTrigger`                                                         |
| `RECV_FIRST`     | `recv_first_ns`, `last_recv_ns` | `TtftTrigger`, `ChunkDeltaTrigger` (skips: pre-change is None)       |
| `RECV_NON_FIRST` | `last_recv_ns`                  | `ChunkDeltaTrigger`                                                  |
| `COMPLETE`       | `complete_ns`                   | `SampleLatencyTrigger`, `OslTrigger`, `TpotTrigger` (streaming only) |

### Error Events

`ErrorEventType.*` events increment the `n_samples_failed` counter.

## Metrics Computed

### Timing Metrics (sync triggers)

| Metric              | Trigger                | Formula                                   |
| ------------------- | ---------------------- | ----------------------------------------- |
| `ttft_ns`           | `TtftTrigger`          | `recv_first_ns - issued_ns`               |
| `chunk_delta_ns`    | `ChunkDeltaTrigger`    | `current_recv_ns - previous_last_recv_ns` |
| `sample_latency_ns` | `SampleLatencyTrigger` | `complete_ns - issued_ns`                 |

### Token Metrics (async triggers)

| Metric    | Trigger       | Source                                                                |
| --------- | ------------- | --------------------------------------------------------------------- |
| `isl`     | `IslTrigger`  | `len(token_ids)` (sync) or `token_count(text)` (async)                |
| `osl`     | `OslTrigger`  | `token_count(full_output_text)`                                       |
| `tpot_ns` | `TpotTrigger` | `(complete_ns - recv_first_ns) / token_count(text_after_first_chunk)` |

## Performance Tracking

See the existing TrackedBlock documentation — this has not changed.
Tracking windows, block duration, and aggregate QPS computation
remain identical to the previous design.

## SampleRow

```python
class SampleRow(msgspec.Struct, gc=False):
    sample_uuid: str
    tracked_block_idx: int = -1
    issued_ns: int | None = None
    recv_first_ns: int | None = None
    last_recv_ns: int | None = None
    complete_ns: int | None = None
```

## Lifecycle

### Startup

```
python -m inference_endpoint.async_utils.services.metrics_aggregator \
    --socket-dir /tmp/socket_dir \
    --socket-name ev_pub_abc123 \
    --tokenizer gpt2 \
    --streaming \
    --readiness-path svc_ready_1a2b3c4d \
    --readiness-id 1
```

1. Auto-generate metrics directory: `/dev/shm/metrics_<uuid>`
2. Create `BasicKVStore` in that directory
3. Create `TokenizePool` (if `--tokenizer` provided)
4. Create `MetricsAggregatorService` — constructs `MetricsTable(kv_store)`,
   registers triggers (which create series keys), creates counter keys
5. `aggregator.start()` — registers ZMQ socket reader
6. Signal readiness via `send_ready_signal()` (if `--readiness-path` provided)
7. `await shutdown_event.wait()`

### Shutdown

1. `ENDED` received → set shutdown flag
2. `await table.drain_tasks()` → wait for async triggers
3. Update `duration_ns` counter with final tracked duration
4. Close KVStore (closes all mmap files)
5. `finally`: `kv_store.unlink()` removes `/dev/shm/metrics_<uuid>`
6. `shutdown_event.set()` → unblock main coroutine

## Future: Prometheus

The `KVStore` ABC allows swapping `BasicKVStore` with a Prometheus-backed
implementation. Counters map to Prometheus gauges, series map to histograms
or summaries. The trigger and aggregator code would not change.
