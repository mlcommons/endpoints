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

| File               | Purpose                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| `__main__.py`      | Subprocess entry: argparse, strict tokenizer startup, lifecycle wiring, SIGTERM |
| `aggregator.py`    | `MetricsAggregatorService` — event router, session state, drain                 |
| `registry.py`      | `MetricsRegistry`, `CounterSampler`, `SeriesSampler`                            |
| `snapshot.py`      | `MetricsSnapshot` wire schema, `SessionState`, msgpack codec                    |
| `publisher.py`     | `MetricsPublisher` — tick task + atomic final-snapshot write                    |
| `subscriber.py`    | `MetricsSnapshotSubscriber` — main-process consumer                             |
| `metrics_table.py` | In-flight sample rows + trigger dispatch (TTFT/TPOT/ISL/OSL)                    |
| `token_metrics.py` | `BatchTokenizer` (sharded batch tokenization) + `TokenBatchQueue`               |

## Lifecycle

```
INITIALIZE ──STARTED──► LIVE ──ENDED──► DRAINING ──► COMPLETE
                                                └──► INTERRUPTED  (SIGTERM/SIGINT)
```

- **LIVE**: the publisher tick task emits a snapshot every
  `--publish-interval` seconds (default 0.25 s).
- **DRAINING**: entered on `ENDED`; the buffered tokenizations are flushed,
  bounded by the `--drain-timeout` budget (default 300 s; `0` = unlimited).
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

1. **The queue's own live loop** — `start_live(interval)` flushes
   periodically (at the publish cadence) through the tokenizer's **in-process
   live lane**: a small thread pool of `--tokenizer-workers` threads
   (default 2) whose rayon pool is capped to the same width, taking at most
   `_LIVE_FLUSH_MAX_ITEMS` per flush so the queue lock is never held for a
   long encode. Live flushes never touch the shard processes; they run inside
   the aggregator process, wherever the parent placed it.
   `--tokenizer-workers 0` disables mid-run tokenization entirely. Failures
   are logged once and never stop the loop — failed or cancelled live items
   are **re-queued** so the drain retries them.
2. **End-of-run** — `flush_remaining(timeout)` stops the live loop and drains
   everything still buffered through **every** shard, bounded by the drain
   budget. The publisher knows nothing about tokenization — it only reads
   `(state, n_pending_tasks)`.

`flush()` serializes under an asyncio lock and detaches the buffer up front,
so enqueues that race a flush land in the next one. Failure isolation is
layered: the plain-text phase and the chat-template phase fail independently
(in drain mode they run on separate executors, so a dead text shard must not
drop message items), a raising recorder callback is logged without aborting
the rest of the batch, and the first error is re-raised only after both
phases ran. Live-mode failures and cancellations re-queue the detached items
(a mid-run hiccup never loses samples); drain-mode failures are terminal —
the items stay counted in `pending`. `flush_remaining` never raises — a
timeout or tokenizer failure becomes a logged, non-zero pending count.

### Sharded batch encoding (`BatchTokenizer`)

The end-of-run drain hands the whole buffer to `count_texts_async`, which splits it into
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
  serialize (the warmup wait is bounded — a hung load is a startup error, not
  a wedge), and they ignore SIGINT — Ctrl-C goes to the whole process group,
  and worker lifetime must stay under the parent drain's control.

The shard pool has no CLI knob: it always auto-sizes to one shard per
8-core block of the allowed CPU universe (always at least one).
`--tokenizer-workers` sizes the **live** in-process thread lane instead
(default 2; `0` = no mid-run tokenization). There is no implicit fallback: an
environment that cannot shard — no fast Rust backend, a failed or over-budget
warmup — is a startup error, because a silent in-process slow path cannot
keep up with completions and would surface much later as an incomplete drain.
Platforms without a CPU-affinity API (e.g. macOS) still shard at full speed,
just unpinned: blocks are sized from the online CPU count and each worker
caps its rayon pool to the block size instead of pinning.

Chat-template items (tool-call outputs) take a separate in-process thread:
they are rare relative to the batched flush, and `apply_chat_template` is
Python/Jinja — sharding buys nothing. A template baseline (the empty
assistant-message frame) is computed once and subtracted so only the payload
is counted.

### CPU affinity: the tokenizer stage is post-run

The benchmark parent pins itself to the loadgen cores (the fastest
perf-ranked physical cores) before launching services, and subprocesses
inherit that narrow mask. The tokenizer's heavy work happens **after** the
run, so the run-time core partition does not apply to it — but the aggregator
itself must not move: `_setup_shards` probes the full allowed universe via
`expand_to_all_online_cpus()` (see `endpoint_client/cpu_affinity.py`; the
kernel still clamps to the cgroup/Slurm cpuset) **and then restores the
inherited mask**, so the event loop, the publisher, and the live tokenizer
threads stay exactly where the parent placed them. Only the drain-phase shard
children, which pin themselves to their own 8-core blocks, span the whole
machine — and they are idle until `ENDED`.

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
   live loop (0.25 s) ── flush(live) ───────┤─► in-process thread pool
                                            │   (rayon capped to --tokenizer-workers)
   ENDED drain (budgeted) ── flush() ───────┘─► chunks ─► N pinned worker procs
                                                │          (encode_batch_fast)
                                                └─► on_count(n) ─► registry.record()
```

## CLI Interface

| Flag                             | Default  | Purpose                                             |
| -------------------------------- | -------- | --------------------------------------------------- |
| `--socket-dir` / `--socket-name` | required | EventRecord SUB socket                              |
| `--metrics-socket`               | required | Snapshot PUB socket name                            |
| `--metrics-output-dir`           | required | Directory for `final_snapshot.json`                 |
| `--publish-interval`             | 0.25     | Live snapshot cadence (seconds)                     |
| `--drain-timeout`                | 300.0    | End-of-run tokenize budget (`0` = unlimited)        |
| `--tokenizer`                    | none     | HF name or local path; unset disables token metrics |
| `--tokenizer-workers`            | 2        | Live in-process threads (`0` = defer all to drain)  |
| `--streaming`                    | off      | Register TTFT/chunk-delta/TPOT triggers             |

## References

- [docs/async_utils/services/DESIGN.md](../DESIGN.md) — the EventRecord
  pub/sub system this service subscribes to.
- [docs/PERF_ARCHITECTURE.md](../../../PERF_ARCHITECTURE.md) — CPU pinning
  strategy for the loadgen/worker hot path.
- AGENTS.md "Metrics Aggregator subprocess" — the condensed contract summary
  for AI agents.
