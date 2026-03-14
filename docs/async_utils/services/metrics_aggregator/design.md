# Metrics Aggregator Service — Design Document

## Overview

The metrics aggregator is a ZMQ subscriber service that consumes `EventRecord` messages
from the pub/sub event bus, computes per-sample metrics in real time, and pushes them
to a `MetricEmitter` backend (currently JSONL; future: Prometheus PushGateway).

It runs as an independent subprocess with its own event loop, connected to the same
ZMQ PUB socket as the EventLoggerService.

```
                        ZMQ PUB (ipc://)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       EventLogger   MetricsAggregator   (future subscribers)
       (JSONL/SQL)   (real-time metrics)
```

## Module Layout

```
metrics_aggregator/
├── __init__.py
├── __main__.py          # CLI entry point
├── aggregator.py        # MetricsAggregatorService (ZmqEventRecordSubscriber)
├── emitter.py           # MetricEmitter ABC, JsonlMetricEmitter
├── metrics_table.py     # SampleRow, MetricsTable
└── token_metrics.py     # TokenizePool (thread-pool tokenizer)
```

## Subscribed Events

### Session Events

| Event                                         | Effect                                               |
| --------------------------------------------- | ---------------------------------------------------- |
| `SessionEventType.STARTED`                    | Records session start timestamp                      |
| `SessionEventType.START_PERFORMANCE_TRACKING` | Sets `is_tracking = True`                            |
| `SessionEventType.STOP_PERFORMANCE_TRACKING`  | Sets `is_tracking = False`                           |
| `SessionEventType.ENDED`                      | Flushes emitter, closes subscriber, signals shutdown |

### Sample Events

| Event              | Stored Field                                        | Metric Emitted                        | Formula                                |
| ------------------ | --------------------------------------------------- | ------------------------------------- | -------------------------------------- |
| `ISSUED`           | `issued_ns`                                         | `isl` (async)                         | `token_count(prompt_text)`             |
| `RECV_FIRST`       | `recv_first_ns`, `last_recv_ns`, `first_chunk_text` | `ttft_ns`                             | `recv_first_ns - issued_ns`            |
| `RECV_NON_FIRST`   | `last_recv_ns` (updated)                            | `chunk_delta_ns`                      | `timestamp - last_recv_ns`             |
| `CLIENT_SEND`      | `client_send_ns`                                    | —                                     | —                                      |
| `CLIENT_RESP_DONE` | `client_resp_done_ns`                               | `request_duration_ns`                 | `client_resp_done_ns - client_send_ns` |
| `COMPLETE`         | `complete_ns`                                       | `sample_latency_ns`, `osl`, `tpot_ns` | see below                              |

Ignored sample events: `TRANSPORT_SENT`, `TRANSPORT_RECV` (infrastructure-level, not
relevant for user-facing metrics).

## Performance Tracking Window

The `is_tracking` flag gates which samples are tracked:

```
  STARTED                                                    ENDED
    │                                                          │
    ▼                                                          ▼
────┬─────────┬─────────────────────────────┬──────────────────┬──
    │         │  ◄── samples issued here    │                  │
    │   START_PERF_TRACKING          STOP_PERF_TRACKING        │
    │         │      are tracked            │                  │
    │         │                             │                  │
```

- A sample is tracked **if and only if** its `ISSUED` event arrives while `is_tracking` is `True`.
- Once tracked, a sample continues to receive events and emit metrics regardless of
  later `STOP_PERFORMANCE_TRACKING` events. Only new ISSUEs are blocked.
- This allows warmup queries (before START) and cooldown queries (after STOP) to be
  excluded from reported metrics while still draining in-flight samples cleanly.

## Data Model: SampleRow

Each tracked sample gets a `SampleRow` — a `msgspec.Struct` with `gc=False` that
stores raw `int | None` nanosecond timestamps and accumulated text:

```
SampleRow
├── sample_uuid: str
├── issued_ns: int | None           ← set on ISSUED
├── complete_ns: int | None          ← set on COMPLETE
├── recv_first_ns: int | None        ← set on RECV_FIRST
├── last_recv_ns: int | None         ← set on RECV_FIRST, updated on each RECV_NON_FIRST
├── client_send_ns: int | None       ← set on CLIENT_SEND
├── client_resp_done_ns: int | None  ← set on CLIENT_RESP_DONE
├── prompt_text: str | None          ← from ISSUED event data (for ISL tokenization)
├── first_chunk_text: str | None     ← from RECV_FIRST event data (for TPOT denominator)
└── output_chunks: list[str]         ← accumulated from RECV_FIRST/RECV_NON_FIRST data
```

Metric formulas are simple methods on the row:

```python
def ttft_ns(self) -> int | None:           # recv_first_ns - issued_ns
def sample_latency_ns(self) -> int | None: # complete_ns - issued_ns
def request_duration_ns(self) -> int | None: # client_resp_done_ns - client_send_ns
def output_text(self) -> str:              # "".join(output_chunks)
```

Rows are created on ISSUED and removed on COMPLETE.

### Design Rationale: Why Not a Declarative Field System

An earlier iteration used `_MetricField` structs with `delta_start_field_prio` lists
to declaratively describe which field pairs produce which metrics. This was abandoned
because:

1. The formulas are few and fixed — a declarative DSL adds indirection without flexibility.
2. String-based field lookups at runtime obscure the actual data flow.
3. The metric emission logic was coupled into the data storage layer (`set_field` both
   stored a timestamp and emitted a metric), making it hard to test or reason about.
4. Special cases (`mutable` flag for `recv_non_first`, `msgspec.UNSET` sentinels)
   added complexity for what is fundamentally `int | None`.

The current design keeps data storage (SampleRow) separate from metric emission
(aggregator event handlers). Each handler is 5-15 lines, reads top-to-bottom, and
is independently testable.

## Metrics Computed

### Timing Metrics (emitted immediately on triggering event)

| Metric                | Emitted On       | Formula                                | Notes                                                                                                      |
| --------------------- | ---------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ttft_ns`             | RECV_FIRST       | `recv_first_ns - issued_ns`            | Time to first token. Streaming only.                                                                       |
| `sample_latency_ns`   | COMPLETE         | `complete_ns - issued_ns`              | End-to-end latency from issue to completion.                                                               |
| `request_duration_ns` | CLIENT_RESP_DONE | `client_resp_done_ns - client_send_ns` | HTTP-level request time (inside worker process).                                                           |
| `chunk_delta_ns`      | RECV_NON_FIRST   | `timestamp - last_recv_ns`             | Inter-token arrival time. `last_recv_ns` starts at `recv_first_ns` and advances with each non-first chunk. |

### Token Metrics (require tokenization, may be async)

| Metric    | Emitted On           | Formula                                                      | Notes                                                                                                                                                                                                                                   |
| --------- | -------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `isl`     | ISSUED (async)       | `token_count(prompt_text)`                                   | Input sequence length. Scheduled as async task to not block event processing.                                                                                                                                                           |
| `osl`     | COMPLETE (awaited)   | `token_count(output_text)`                                   | Output sequence length. Output text is from accumulated chunks (streaming) or COMPLETE data (non-streaming).                                                                                                                            |
| `tpot_ns` | COMPLETE (after OSL) | `(complete_ns - recv_first_ns) / (osl - first_chunk_tokens)` | Time per output token after the first chunk. The first chunk may contain multiple tokens, so `first_chunk_text` is tokenized separately for the denominator. Only emitted for streaming responses where `osl - first_chunk_tokens > 0`. |

## Event Dispatch Flow

```
process(records: list[EventRecord])
│
├── for each record:
│   ├── Session events → update is_tracking / session state
│   │
│   └── Sample events (if sample_uuid non-empty):
│       ├── ISSUED
│       │   ├── if not is_tracking: skip
│       │   ├── create SampleRow in MetricsTable
│       │   ├── store issued_ns
│       │   ├── store prompt_text from record.data (if str)
│       │   └── schedule ISL tokenization (async, fire-and-forget)
│       │
│       ├── RECV_FIRST
│       │   ├── lookup row (skip if not tracked)
│       │   ├── store recv_first_ns, last_recv_ns
│       │   ├── emit ttft_ns
│       │   └── append record.data to output_chunks
│       │
│       ├── RECV_NON_FIRST
│       │   ├── lookup row (skip if not tracked)
│       │   ├── emit chunk_delta_ns (from last_recv_ns)
│       │   ├── update last_recv_ns
│       │   └── append record.data to output_chunks
│       │
│       ├── CLIENT_SEND
│       │   └── store client_send_ns
│       │
│       ├── CLIENT_RESP_DONE
│       │   ├── store client_resp_done_ns
│       │   └── emit request_duration_ns
│       │
│       └── COMPLETE
│           ├── store complete_ns
│           ├── emit sample_latency_ns
│           ├── await OSL tokenization → emit osl
│           ├── if streaming and osl > first_chunk_tokens → emit tpot_ns
│           └── remove row from MetricsTable
│
└── if ENDED seen: flush emitter, close subscriber, signal shutdown
```

## MetricEmitter

The `MetricEmitter` ABC defines:

```python
class MetricEmitter(ABC):
    def emit(self, sample_uuid: str, metric_name: str, value: int | float) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

### JsonlMetricEmitter (current implementation)

Writes one JSON line per metric:

```json
{"sample_uuid":"a1b2c3...","metric_name":"ttft_ns","value":1500,"timestamp_ns":98765432100}
{"sample_uuid":"a1b2c3...","metric_name":"sample_latency_ns","value":4000,"timestamp_ns":98765436100}
```

Uses `msgspec.json.Encoder` for serialization. Supports a configurable `flush_interval`
(flush to disk every N records). `timestamp_ns` is the wall-clock time when the metric
was emitted (not the event timestamp).

### Future: PrometheusEmitter

Would push to Prometheus PushGateway. The `emit()` signature supports this —
`metric_name` maps to a Prometheus metric, `sample_uuid` becomes a label,
`value` is the observation. Histograms/summaries can be built by accumulating
values per metric name.

## TokenizePool

Thread-pool wrapper around HuggingFace `AutoTokenizer` for ISL/OSL/TPOT computation.

### Architecture

```
                          TokenizePool
                         ┌─────────────┐
                         │ ThreadPool  │
    token_count("text")──►  Executor   │
         (blocking)      │  ┌───────┐  │
                         │  │Thread1│──► thread-local AutoTokenizer
                         │  │Thread2│──► thread-local AutoTokenizer
                         │  │  ...  │   │
                         │  └───────┘  │
                         └─────────────┘
```

### Thread-Safety Analysis

- **`ThreadPoolExecutor.submit()`** is internally synchronized — safe to call from
  any thread.
- **Thread-local tokenizer instances** (`threading.local()`) mean zero shared mutable
  state during tokenization. Each worker thread lazily loads its own
  `AutoTokenizer.from_pretrained()` on first use.
- **HuggingFace tokenizers** (Rust backend via `tokenizers` crate) release the GIL
  during the core tokenization work, so multiple threads actually run in parallel.
- **Blocking vs async**: `tokenize()` and `token_count()` block the calling thread
  on `future.result()`. In async context, use `token_count_async()` which wraps the
  call in `loop.run_in_executor(None, ...)` to avoid blocking the event loop.

### Why `run_in_executor` for async?

The `token_count_async` method uses a double-hop: `event loop executor → TokenizePool executor`.
This seems redundant but is necessary because:

1. The aggregator's `process()` runs as an async task on the event loop.
2. Calling `pool.token_count()` directly would block the loop (the `future.result()`
   inside `token_count()` is a synchronous wait).
3. `run_in_executor` offloads the blocking call to a thread, freeing the loop to
   continue processing events.

The inner `ThreadPoolExecutor` in `TokenizePool` still provides the thread-local
tokenizer isolation. The outer executor just prevents the blocking wait from starving
the event loop.

## ISL Tracking: How the Prompt Gets to the Aggregator

### Current Design

The `ISSUED` event's `data` field carries the prompt text as a `str`. The aggregator
tokenizes this to compute ISL.

`EventRecord.data` is typed as `OUTPUT_TYPE | ErrorData | None` where
`OUTPUT_TYPE = str | TextModelOutput`. A raw prompt string fits naturally.

### Where to Publish

The ISSUED event is published in the load generator when `issue_sample()` is called.
At that point, `sample.data` contains the post-transform dataset row. The publisher
extracts the prompt:

```python
# In the load generator, when issuing a sample:
prompt_text = sample.data.get("prompt", "")
publisher.publish(EventRecord(
    event_type=SampleEventType.ISSUED,
    sample_uuid=sample.uuid,
    data=prompt_text or None,
))
```

### Adapter Considerations

The prompt text available at ISSUED time is **post-transform** — dataset transforms
have already been applied by this point. This matters because:

| Adapter                 | Transform Pipeline                            | `sample.data` at ISSUED             | ISL Source                              |
| ----------------------- | --------------------------------------------- | ----------------------------------- | --------------------------------------- |
| OpenAI / OpenAI-Msgspec | `ColumnFilter → AddStaticColumns`             | `{"prompt": "...", "model": "..."}` | Tokenize `prompt` string                |
| SGLang                  | `Harmonize → ColumnFilter → AddStaticColumns` | `{"input_tokens": [int, ...]}`      | `len(input_tokens)` — already tokenized |

For SGLang, the `Harmonize` transform pre-tokenizes the prompt into `input_tokens`
(a list of integer token IDs). The original text is discarded. Options:

1. **Short-term**: If `input_tokens` is present, ISL = `len(input_tokens)`. The
   publisher can check for this and encode the count as a string in `data`
   (e.g., the aggregator checks `data.startswith("__ISL:")`) or use a new event type.
2. **Better**: Preserve the original prompt text in `sample.data["prompt"]` alongside
   `input_tokens` during the Harmonize transform. The publisher always sends `prompt`.
3. **Best (future)**: A `DATA_READY` event with richer metadata, or widen
   `EventRecord.data` to support a metadata dict. Requires core type changes.

Recommendation: option 2 for now — modify `Harmonize` to keep `prompt` in the output
row alongside `input_tokens`.

## Lifecycle

### Startup

```python
python -m inference_endpoint.async_utils.services.metrics_aggregator \
    --metrics-dir /tmp/metrics \
    --socket-address ipc:///tmp/events.sock \
    --tokenizer gpt2 \
    --tokenizer-workers 2
```

1. Create `TokenizePool` (if `--tokenizer` provided)
2. Create `JsonlMetricEmitter` writing to `<metrics-dir>/metrics.jsonl`
3. Create `MetricsAggregatorService` connected to the ZMQ PUB socket
4. `aggregator.start()` adds the ZMQ socket reader to the event loop
5. `await shutdown_event.wait()` blocks until ENDED is received

### Shutdown

On `SessionEventType.ENDED`:

1. `_finalize()` flushes the emitter
2. `close()` closes the emitter file and removes the ZMQ socket reader
3. `shutdown_event.set()` unblocks the main coroutine
4. `TokenizePool.close()` shuts down worker threads (in `finally` block)

### Graceful Drain

Events received in the same batch as ENDED are processed (the `_shutdown_received`
flag is checked at the top of the loop, so events before ENDED in the batch still
get handled). Events in subsequent batches are dropped.

In-flight samples that never receive COMPLETE will be abandoned (their rows stay in
the table but are never emitted). This is expected — if the session ends, those
samples didn't complete.

## Output Format

### JSONL Example (streaming sample)

```json
{"sample_uuid":"a1b2c3d4","metric_name":"isl","value":42,"timestamp_ns":100000000}
{"sample_uuid":"a1b2c3d4","metric_name":"ttft_ns","value":1500000,"timestamp_ns":100001500}
{"sample_uuid":"a1b2c3d4","metric_name":"chunk_delta_ns","value":500000,"timestamp_ns":100002000}
{"sample_uuid":"a1b2c3d4","metric_name":"chunk_delta_ns","value":600000,"timestamp_ns":100002600}
{"sample_uuid":"a1b2c3d4","metric_name":"request_duration_ns","value":3800000,"timestamp_ns":100003800}
{"sample_uuid":"a1b2c3d4","metric_name":"sample_latency_ns","value":4000000,"timestamp_ns":100004000}
{"sample_uuid":"a1b2c3d4","metric_name":"osl","value":28,"timestamp_ns":100004001}
{"sample_uuid":"a1b2c3d4","metric_name":"tpot_ns","value":92592.6,"timestamp_ns":100004001}
```

### JSONL Example (non-streaming sample)

```json
{"sample_uuid":"e5f6a7b8","metric_name":"isl","value":15,"timestamp_ns":200000000}
{"sample_uuid":"e5f6a7b8","metric_name":"request_duration_ns","value":2500000,"timestamp_ns":200002500}
{"sample_uuid":"e5f6a7b8","metric_name":"sample_latency_ns","value":3000000,"timestamp_ns":200003000}
{"sample_uuid":"e5f6a7b8","metric_name":"osl","value":50,"timestamp_ns":200003001}
```

Note: no `ttft_ns`, `chunk_delta_ns`, or `tpot_ns` for non-streaming — these require
`RECV_FIRST` which only occurs in streaming mode.

## Not Yet Wired

The EventRecord pub/sub infrastructure is ready, but actual `publish(EventRecord(...))`
calls for sample events have not been connected in the load generator or worker
processes. What needs to happen:

1. **Load generator** (`load_generator.py` / `session.py`): Publish `ISSUED` with
   prompt text, `START/STOP_PERFORMANCE_TRACKING`, `STARTED`, `ENDED`.
2. **Worker** (`worker.py`): Publish `CLIENT_SEND`, `CLIENT_RESP_DONE`,
   `RECV_FIRST`, `RECV_NON_FIRST`, `COMPLETE` with response data.
3. **Session orchestrator**: Spawn the metrics aggregator subprocess alongside
   the event logger subprocess, passing the same ZMQ socket address.
