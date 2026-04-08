# Warmup Strategy Design

**Date:** 2026-04-07
**Status:** Draft
**Author:** rkaleem + Claude

## Problem

LLM inference endpoints exhibit cold-start artifacts that contaminate benchmark measurements. Without warmup, the first N requests see artificially inflated TTFT due to:

- CUDA graph compilation and JIT kernel specialization for new sequence length buckets
- KV cache page table initialization (PagedAttention block allocation)
- Batch scheduler stabilization (preemption thresholds, dynamic batching windows)
- Client-side connection pool ramp-up and worker process JIT (Python 3.12 specializing interpreter)

Empirical analysis of a 13-concurrency-level sweep on a B200 server (GPT OSS 120B) shows:

| Concurrency | Cold-start TTFT | Steady-state TTFT | Ratio |
| ----------- | --------------- | ----------------- | ----- |
| 16          | 155 ms          | 66 ms             | 2.4x  |
| 64          | 417 ms          | 66 ms             | 6.3x  |
| 256         | 9,081 ms        | 210 ms            | 43x   |
| 1024        | 21,872 ms       | 89,946 ms\*       | N/A   |

\*c=1024 is oversaturated; steady-state TTFT is dominated by queue depth, not cold start.

The cold-start effect is primarily a prefill/TTFT problem. TPOT (decode performance) is stable from the first token at all concurrency levels.

## Goals

1. Measure steady-state inference performance by warming both server and client before measurement begins
2. Avoid prefix cache contamination (warmup dataset must have different content from performance dataset)
3. Maintain continuous server load during the transition from warmup to measurement (no batch drain gap)
4. Cleanly separate warmup and performance metrics via event timestamps in the existing SQLite events DB

## Non-Goals

- Synthetic warmup dataset generation (future work)
- Convergence-based warmup termination (adaptive stopping when metrics stabilize)
- Warmup for accuracy evaluation datasets

## Design

### Strategy: Hybrid Overlap with Two-Phase Single-Session

The warmup and performance phases share a single `BenchmarkSession` and `EventRecorder`. The warmup phase issues requests from a user-provided warmup dataset. The performance phase begins issuing immediately after all warmup samples are issued (no drain wait). A `START_PERFORMANCE_TRACKING` event is recorded when the last warmup request completes. Only performance samples issued after that timestamp are included in reported metrics.

### Execution Timeline

```
Phase:     Warmup Issuance        Performance Issuance
           |-------- W -------->|------------ P ------------------------------>|
                                 ^                                              ^
                                 warmup issuance done                    perf issuance done
                                 perf issuance begins immediately        STOP_PERFORMANCE_TRACKING

                    |--overlap--|
                    warmup responses still completing
                    perf requests issued but not yet counted

                                * last warmup response completes *
                                --> START_PERFORMANCE_TRACKING event
                                --> perf samples issued AFTER this count
```

### Why Not Drain-Then-Measure?

Data from the concurrency sweep proves drain is counterproductive:

At c=256, draining all warmup requests takes ~82 seconds. During drain, batch occupancy drops from 256 toward 0. When performance measurement begins, the batch scheduler must refill from empty, recreating the exact cold-start artifact warmup was meant to prevent.

The hybrid approach keeps the batch scheduler loaded throughout the transition. Performance requests backfill slots as warmup requests complete, maintaining steady-state batch occupancy.

### Components

#### 1. New Session Event: `START_PERFORMANCE_TRACKING`

Add to `SessionEvent` enum in `load_generator/events.py`:

```python
class SessionEvent(Event):
    ...
    START_PERFORMANCE_TRACKING = "start_performance_tracking"
```

Recorded by the warmup completion hook when the last warmup sample completes.

#### 2. Warmup Completion Hook

A `SampleEvent.COMPLETE` hook registered before warmup begins. Tracks outstanding warmup sample UUIDs. When the count reaches zero, records `START_PERFORMANCE_TRACKING`.

```
WarmupTracker:
    outstanding_uuids: set[str]    # populated during warmup issuance
    _active: bool = True

    on_complete(result):
        if not self._active:
            return  # already fired, ignore subsequent calls
        outstanding_uuids.discard(result.id)
        if len(outstanding_uuids) == 0:
            self._active = False
            EventRecorder.record_event(START_PERFORMANCE_TRACKING, now)
```

The tracker deactivates itself via `_active` flag rather than removing the hook from `SampleEventHandler`. This avoids accidentally clearing other `COMPLETE` hooks (e.g., progress bar, concurrency scheduler slot release). The cost is one extra bool check per completion for the remainder of the run — negligible.

Thread safety: The hook fires on the response handler's thread (via `SampleEventHandler.query_result_complete`). UUID discards come from a single call site. The `_active` flag is set once (True -> False) and only read thereafter, so no race condition.

#### 3. Warmup LoadGenerator

Constructed identically to the performance `SchedulerBasedLoadGenerator`, but:

- Uses the warmup `Dataset` instance
- Uses `RuntimeSettings` with `n_samples_to_issue = warmup_samples` (the user-configured count)
- Uses the same `Scheduler` class and load pattern as the performance phase
- Shares the same `SampleIssuer` (same HTTP client, same workers)

Sample wrapping: If `warmup_samples > warmup_dataset.num_samples()`, the scheduler wraps around (reuses samples). Repeated prompts during warmup are acceptable since warmup metrics are not reported and the goal is system priming, not content diversity.

#### 4. Session Orchestration Changes

`BenchmarkSession._run_test()` gains a warmup phase:

```
_run_test(warmup_generator, perf_generator, ...):
    with event_recorder:
        TEST_STARTED

        if warmup_generator:
            # Register warmup completion hook
            tracker = WarmupTracker(...)
            SampleEventHandler.register_hook(COMPLETE, tracker.on_complete)

            # Issue all warmup samples, collecting UUIDs
            for issued in warmup_generator:
                tracker.outstanding_uuids.add(issued.sample.uuid)

            # Do NOT wait for warmup completion — fall through immediately
            # START_PERFORMANCE_TRACKING fires asynchronously when last
            # warmup response arrives (via tracker.on_complete hook)

        # Issue performance samples (overlaps with warmup tail)
        for _ in perf_generator:
            pass

        STOP_PERFORMANCE_TRACKING
        ...  # wait for inflight, report, etc.
```

Note: There is a timing subtlety — some warmup samples may complete _before_ their UUID is added to `outstanding_uuids` (since `issue_sample()` is non-blocking and the response could arrive before the loop iteration advances). To handle this, `WarmupTracker` should pre-populate UUIDs before issuance, or use a counter instead of a set. The implementation plan will address this.

If no warmup is configured, the flow is identical to today (no `START_PERFORMANCE_TRACKING` event, reporter behaves as current).

#### 5. MetricsReporter Changes

Add a `start_performance_tracking_timestamp_ns` property (symmetric to existing `stop_performance_tracking_timestamp_ns`):

```python
@property
def start_performance_tracking_timestamp_ns(self) -> float:
    """Returns timestamp of START_PERFORMANCE_TRACKING, or -inf if not found."""
    ...
```

Update SQL queries that filter performance samples. Currently they use:

```sql
HAVING MAX(CASE WHEN event_type = 'loadgen_issue_called' THEN timestamp_ns END) < {stop_ts}
```

Add a lower bound:

```sql
HAVING MAX(CASE WHEN event_type = 'loadgen_issue_called' THEN timestamp_ns END) >= {start_ts}
   AND MAX(CASE WHEN event_type = 'loadgen_issue_called' THEN timestamp_ns END) < {stop_ts}
```

Affected methods:

- `derive_sample_latency()`
- `get_sample_statuses()`
- `get_error_count()`
- `derive_ttft()`
- `derive_tpot()`
- `derive_output_sequence_lengths()`
- `duration_ns` (start time becomes `max(TEST_STARTED, START_PERFORMANCE_TRACKING)`)

When `START_PERFORMANCE_TRACKING` is absent, `start_ts = -inf` and the queries behave identically to today. This maintains backward compatibility with existing event databases.

#### 6. Configuration

Add warmup fields to the config schema (`config/schema.py`):

```python
class WarmupConfig(pydantic.BaseModel):
    dataset: DatasetReference | None = None
    samples: int | None = None  # default: target_concurrency if concurrency mode, else 64
```

Add `warmup: WarmupConfig | None` to `BenchmarkConfig`.

CLI surface:

```
--warmup-dataset <path>       # path to warmup dataset (same format as --dataset)
--warmup-samples <int>        # number of warmup samples to issue
```

YAML surface:

```yaml
warmup:
  dataset: warmup_data.jsonl
  samples: 256
```

Default behavior when warmup dataset is provided but samples is omitted:

- `concurrency` mode: `samples = target_concurrency` (fill all batch slots)
- `poisson` mode: `samples = 64` (reasonable default for steady-state priming)
- `max_throughput` mode: `samples = 64`

When no warmup dataset is provided, warmup is skipped entirely.

### Edge Cases

| Scenario                                                 | Behavior                                                                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| All warmup requests complete before perf issuance starts | `START_PERFORMANCE_TRACKING` fires before first perf sample. All perf samples count. Degenerates to drain-then-measure with zero gap. |
| Warmup samples error out                                 | Count as complete for warmup countdown. Goal is system priming, not correctness.                                                      |
| No warmup configured                                     | No `START_PERFORMANCE_TRACKING` event. Reporter uses `-inf` as start bound. Identical to current behavior.                            |
| `warmup_samples` > warmup dataset size                   | Wrap around via scheduler's `WithoutReplacementSampleOrder` cycling.                                                                  |
| Warmup dataset ISL/OSL doesn't match perf                | User's responsibility. Documentation will advise matching distributions.                                                              |
| Warmup completion takes very long (>10% of run)          | Accepted tradeoff. Future work could add `warmup_drain_timeout_s` to cap wait.                                                        |
| Early stop requested during warmup                       | `stop_requested` flag is checked per-sample in the load generator. Warmup terminates, perf phase begins (or is skipped).              |

### Data Flow

```
                                    +------------------+
                                    |  WarmupConfig     |
                                    |  (dataset, count) |
                                    +--------+---------+
                                             |
                            +----------------+----------------+
                            |                                 |
                   +--------v---------+             +---------v--------+
                   | Warmup Dataset   |             | Perf Dataset     |
                   | (user-provided)  |             | (existing)       |
                   +--------+---------+             +---------+--------+
                            |                                 |
                   +--------v---------+             +---------v--------+
                   | Warmup Scheduler |             | Perf Scheduler   |
                   | (same pattern)   |             | (same pattern)   |
                   +--------+---------+             +---------+--------+
                            |                                 |
                   +--------v---------+             +---------v--------+
                   | Warmup LoadGen   |             | Perf LoadGen     |
                   +--------+---------+             +---------+--------+
                            |                                 |
                            |        +------------------+     |
                            +------->| SampleIssuer     |<----+
                                     | (shared)         |
                                     +--------+---------+
                                              |
                                     +--------v---------+
                                     | HTTPEndpointClient|
                                     | (shared workers) |
                                     +--------+---------+
                                              |
                                     +--------v---------+
                                     | EventRecorder    |
                                     | (single SQLite)  |
                                     +------------------+

Events timeline:
  TEST_STARTED
  ... warmup LOADGEN_ISSUE_CALLED events ...
  ... perf LOADGEN_ISSUE_CALLED events (overlap) ...
  START_PERFORMANCE_TRACKING  (when last warmup COMPLETE arrives)
  ... more perf LOADGEN_ISSUE_CALLED events ...
  STOP_PERFORMANCE_TRACKING
  ... wait for inflight ...
  TEST_ENDED
```

### Metrics Impact (from empirical analysis)

Steady-state metrics (middle 80% of measurement window) are unaffected by warmup strategy choice, confirming that the benefit is purely in excluding cold-start contamination:

| Concurrency | Metric   | No Warmup  | Hybrid      | Samples Excluded |
| ----------- | -------- | ---------- | ----------- | ---------------- |
| 16          | TTFT p50 | 40.5 ms    | 40.7 ms     | 18 (1.8%)        |
| 64          | TTFT p50 | 65.9 ms    | 66.0 ms     | 104 (4.3%)       |
| 256         | TTFT p50 | 193.1 ms   | 192.1 ms    | 609 (12.5%)      |
| 512         | TTFT p50 | 2749.5 ms  | 2749.6 ms   | 1225 (19.5%)     |
| 1024        | TTFT p50 | 93998.7 ms | 103918.4 ms | 1555 (26.7%)     |

At high concurrency (512+), the no-warmup TTFT p50 is artificially lowered by the cold-start samples having shorter output sequences (they get prefilled while the batch is underfull, so they see lower TTFT than steady state). The hybrid approach gives the true steady-state measurement.

### Files Modified

| File                            | Change                                                            |
| ------------------------------- | ----------------------------------------------------------------- |
| `load_generator/events.py`      | Add `START_PERFORMANCE_TRACKING` to `SessionEvent`                |
| `load_generator/session.py`     | Add warmup phase to `_run_test()`, accept warmup generator        |
| `load_generator/sample.py`      | No changes (existing hook system sufficient)                      |
| `metrics/reporter.py`           | Add `start_performance_tracking_timestamp_ns`, update SQL queries |
| `config/schema.py`              | Add `WarmupConfig`, add `warmup` field to `BenchmarkConfig`       |
| `commands/benchmark/execute.py` | Load warmup dataset, create warmup scheduler/generator            |
| `commands/benchmark/cli.py`     | Add `--warmup-dataset`, `--warmup-samples` flags                  |

### New Files

| File                                                  | Purpose                                                 |
| ----------------------------------------------------- | ------------------------------------------------------- |
| `load_generator/warmup.py`                            | `WarmupTracker` class (completion hook + UUID tracking) |
| `tests/unit/load_generator/test_warmup.py`            | Unit tests for warmup tracker and session orchestration |
| `tests/integration/commands/test_warmup_benchmark.py` | Integration test: warmup + perf with echo server        |

### Testing Strategy

1. **Unit: WarmupTracker** — verify UUID tracking, `START_PERFORMANCE_TRACKING` emission on last complete, error sample handling
2. **Unit: Reporter filtering** — verify SQL queries correctly filter by `[start_ts, stop_ts)` window, backward compatibility when no start event exists
3. **Unit: Session orchestration** — verify warmup generator runs before perf generator, no drain gap
4. **Integration: Echo server** — full benchmark with warmup dataset, verify warmup samples excluded from report, verify perf metrics are correct
5. **Integration: Edge cases** — warmup completes before perf starts, warmup errors, no warmup configured

### Future Work

- **Synthetic warmup dataset generation**: Given perf dataset ISL/OSL distribution, generate random-token prompts matching the length profile
- **Convergence-based termination**: Monitor rolling TTFT variance during warmup, stop when below threshold
- **Warmup drain timeout**: `warmup_drain_timeout_s` to cap the overlap window and fire `START_PERFORMANCE_TRACKING` even if warmup samples remain in-flight
- **Warmup metrics reporting**: Optional secondary report section showing warmup phase metrics for debugging
