# Metrics Aggregator Service — Design Document

## Overview

The metrics aggregator is a **subprocess** (`python -m
inference_endpoint.async_utils.services.metrics_aggregator`) that subscribes to
the EventRecord pub/sub stream, folds per-sample events into a
`MetricsRegistry` (counters + HDR-histogram series + raw values), and publishes
`MetricsSnapshot` frames over an IPC PUB socket at a fixed cadence. At
end-of-run it atomically writes `final_snapshot.json`, which is the **primary**
source for `Report`; the terminal pub/sub frame is only a TUI "run finished"
signal.

This document covers the service's lifecycle and, in depth, the **token
metrics pipeline** — how ISL/OSL/TPOT tokenization keeps pace with
high-completion-rate runs.

## Module Layout

| File               | Purpose                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| `__main__.py`      | Subprocess entry: argparse, affinity expansion, lifecycle wiring, SIGTERM |
| `aggregator.py`    | `MetricsAggregatorService` — event router, session state, drain           |
| `registry.py`      | `MetricsRegistry`, `CounterSampler`, `SeriesSampler`                      |
| `snapshot.py`      | `MetricsSnapshot` wire schema, `SessionState`, msgpack codec              |
| `publisher.py`     | `MetricsPublisher` — tick task + atomic final-snapshot write              |
| `subscriber.py`    | `MetricsSnapshotSubscriber` — main-process consumer                       |
| `metrics_table.py` | In-flight sample rows + trigger dispatch (TTFT/TPOT/ISL/OSL)              |
| `token_metrics.py` | `BatchTokenizer` (sharded batch tokenization) + `TokenBatchQueue`         |

## Lifecycle

```
INITIALIZE ──STARTED──► LIVE ──ENDED──► DRAINING ──► COMPLETE
                                                └──► INTERRUPTED  (SIGTERM/SIGINT)
```

- **LIVE**: the publisher tick task emits a snapshot every
  `--publish-interval` seconds (default 0.25 s).
- **DRAINING**: entered on `ENDED`; the buffered tokenizations are flushed,
  bounded by the `--drain-timeout` budget (default 60 s; `0` = unlimited).
- The ENDED path runs inside a finalization boundary: whatever the drain does
  — finish, time out, or fail — `publish_final` and the shutdown signal always
  run. A tokenizer failure can degrade the snapshot (see the
  `n_pending_tasks` contract below) but can never hang the subprocess.
- **INTERRUPTED**: a signal handler writes a best-effort partial final
  snapshot so `Report` can distinguish a killed run from a clean one.

## Token Metrics Pipeline

ISL, OSL, and TPOT all require running the HF tokenizer over prompt or
completion text. With streaming on, each completed sample needs up to three
tokenizer passes, so at high completion rates tokenization is the service's
dominant CPU cost — and a per-event dispatch model cannot keep up: work
arriving faster than it drains accumulates an unbounded backlog that must be
paid at end-of-run. The pipeline is therefore built around two ideas:
**defer-to-flush batching** and **process-sharded batch encoding**.

### Defer-to-flush (`TokenBatchQueue`)

Token triggers do no work at event time. `fire()` appends
`(text, on_count)` — or `(message_parts, on_count)` for chat-template items —
to a buffer, an O(1) operation with no event-loop tasks. The buffer is cleared
in batches at exactly two points:

1. **Every publish tick** — the publisher awaits a `pre_publish` hook before
   composing each snapshot, so live ISL/OSL/TPOT reflect recently completed
   samples. A failure here is swallowed by the tick (live publishing never
   stops).
2. **End-of-run** — `flush_remaining(timeout)` drains everything still
   buffered, bounded by the drain budget.

`flush()` serializes under an asyncio lock and detaches the buffer up front,
so enqueues that race a flush land in the next one. Failure isolation is
layered: the plain-text phase and the chat-template phase fail independently
(they run on separate executors, so a dead text shard must not drop message
items), a raising recorder callback is logged without aborting the rest of
the batch, and the first error is re-raised only after both phases ran.
`flush_remaining` never raises — a timeout or tokenizer failure becomes a
logged, non-zero pending count.

### Sharded batch encoding (`BatchTokenizer`)

A flush hands the whole buffer to `count_texts_async`, which splits it into
contiguous chunks and fans them out across worker **processes**, one pinned to
each block of `CORES_PER_WORKER` (8) cores. Why this shape:

- Each worker runs the raw `tokenizers` backend's `encode_batch_fast` — Rust,
  rayon-parallel, no Python-per-text cost. Batching amortizes the
  submit/result overhead over thousands of texts.
- A single BPE rayon pool is memory-bound and saturates at ~8 cores; more
  threads oversubscribe and, on multi-socket parts, cross the NUMA boundary.
  Sharding across processes pinned to disjoint 8-core blocks (affinity set
  **before** the backend loads, so each rayon pool sizes itself to its block
  and stays NUMA-local) is how the whole machine is used.
- Workers are spawn-context processes with module-level entry points (pickled
  by name), warmed in parallel at construction so N tokenizer loads do not
  serialize, and they ignore SIGINT — Ctrl-C goes to the whole process group,
  and worker lifetime must stay under the parent drain's control.

`--tokenizer-workers` controls the shard count: `-1` (default) auto-fits one
shard per 8-core block of the process affinity mask, an explicit count is
clamped to that capacity, and `0` disables sharding. Every fallback to the
in-process path (no fast Rust backend, affinity unavailable, fewer than two
blocks) is logged with its reason — a missing "shards" INFO line should never
be the only signal that the batch path is running single-threaded.

Chat-template items (tool-call outputs) take a separate in-process thread:
they are rare relative to the batched flush, and `apply_chat_template` is
Python/Jinja — sharding buys nothing. A template baseline (the empty
assistant-message frame) is computed once and subtracted so only the payload
is counted.

### CPU affinity: the tokenizer stage is post-run

The benchmark parent pins itself to the loadgen cores before launching
services, and subprocesses inherit that narrow mask. The tokenizer's heavy
work happens **after** the run (the end-of-run flush), so the run-time core
partition does not apply to it: at startup the service calls
`expand_to_all_online_cpus()` (see `endpoint_client/cpu_affinity.py`) to reset
its mask to every online CPU — the kernel still clamps to the cgroup/Slurm
cpuset — and shards size to the full machine. Mid-run tick flushes are small
batches; the drain is where the core count pays.

### The `n_pending_tasks` contract

`TokenBatchQueue.pending` counts enqueued-but-not-yet-recorded items and is
surfaced on every snapshot as `n_pending_tasks`. In the **final** snapshot:

- `state == complete && n_pending_tasks == 0` — clean run, token series exact.
- `state == complete && n_pending_tasks > 0` — **incomplete drain**: the
  end-of-run flush ran out of budget or the tokenizer failed; token-derived
  series are missing exactly that many samples. `Report` renders a warning.

Items dropped by a failed flush are intentionally _not_ removed from the
pending count — under-reporting an incomplete drain would silently rebadge it
as a clean run.

### Data flow

```
COMPLETE event ─► TokenTrigger.fire ─► queue.enqueue(text, on_count)   [O(1)]
                                            │
        publish tick (0.25 s) ──────────────┤  flush()
        ENDED drain (budgeted) ─────────────┘    │
                                                 ├─► chunks ─► N pinned worker procs
                                                 │             (encode_batch_fast)
                                                 └─► on_count(n) ─► registry.record()
```

## CLI Interface

| Flag                             | Default  | Purpose                                             |
| -------------------------------- | -------- | --------------------------------------------------- |
| `--socket-dir` / `--socket-name` | required | EventRecord SUB socket                              |
| `--metrics-socket`               | required | Snapshot PUB socket name                            |
| `--metrics-output-dir`           | required | Directory for `final_snapshot.json`                 |
| `--publish-interval`             | 0.25     | Live snapshot cadence (seconds)                     |
| `--drain-timeout`                | 60.0     | End-of-run tokenize budget (`0` = unlimited)        |
| `--tokenizer`                    | none     | HF name or local path; unset disables token metrics |
| `--tokenizer-workers`            | -1       | Shard processes (`-1` auto, `0` in-process)         |
| `--streaming`                    | off      | Register TTFT/chunk-delta/TPOT triggers             |

## References

- [docs/async_utils/services/DESIGN.md](../DESIGN.md) — the EventRecord
  pub/sub system this service subscribes to.
- [docs/PERF_ARCHITECTURE.md](../../../PERF_ARCHITECTURE.md) — CPU pinning
  strategy for the loadgen/worker hot path.
- AGENTS.md "Metrics Aggregator subprocess" — the condensed contract summary
  for AI agents.
