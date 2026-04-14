# Review Council — Batched ZMQ Publisher

Reviewed by: **Claude** (Codex produced no structured findings) | Depth: **thorough** | Mode: **--no-post**

## 🔴 Must Fix (high)

| #   | File        | Line | Category | Summary                                                                                                     |
| --- | ----------- | ---- | -------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | `pubsub.py` | 234  | bug      | Batch messages bypass ZMQ topic filtering — topic-filtered subscribers receive all event types from batches |
| 2   | `pubsub.py` | 183  | bug      | `close()` sets `is_closed=True` before flushing — concurrent `publish()` calls silently dropped             |

### Detail

**#1 — Batch bypasses topic filtering**

When a subscriber has specific topic filters (e.g., `topics=["session"]`), it also subscribes to `BATCH_TOPIC` (line 234). Batches contain records of all event types. `receive()` yields every payload without checking if the record's event type matches the subscription. A `"session"`-only subscriber would receive `"sample"` records from batches.

Not a production bug today (all subscribers use `topics=None`), but violates the filtering contract. Fix: either filter after unpack, or raise an error when topic filters are combined with a batching publisher.

**#2 — Close ordering issue**

`close()` sets `is_closed = True` at line 183, then calls `_flush_batch()` at line 187. The base class `publish()` checks `is_closed` and returns immediately. Any `publish()` between these lines is silently dropped. If `_flush_batch()` fails, the buffer is lost (already replaced with `[]` inside `_flush_batch`). Fix: flush before setting `is_closed`.

---

## 🟡 Should Fix (medium)

| #   | File         | Line | Category       | Summary                                                                                                                                 |
| --- | ------------ | ---- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 3   | `pubsub.py`  | 271  | error-handling | Corrupt batch silently drops all records (potentially thousands) with only a warning log                                                |
| 4   | `pubsub.py`  | 267  | bug            | Batch topic detection `raw[:TOPIC_FRAME_SIZE] == BATCH_TOPIC` could collide with future event categories starting with "batch"          |
| 5   | `session.py` | 479  | performance    | `flush()` on session events synchronously encodes/sends up to 999 buffered records — session event latency proportional to buffer size  |
| 6   | `execute.py` | 417  | design         | `EventPublisherService` hides loop selection behind `LoopManager` singleton; `send_threshold=1000` not configurable from benchmark path |

### Detail

**#3** — A corrupt batch drops the entire message (could be 1000 records) with just `logger.warning`. For a system that guarantees delivery (`SNDHWM=0`, `LINGER=-1`), this is a data integrity gap. Consider logging the raw data size and raising severity.

**#4** — `BATCH_TOPIC` is not registered via `EventTypeMeta`, so the collision check in `EventTypeMeta.__new__` won't catch a future category named "batch". Add a guard in the metaclass.

**#5** — At 50k QPS, the buffer may have 999 records when a session event triggers `flush()`. Encoding and sending ~70KB synchronously delays the session event. Acceptable for now but worth documenting.

**#6** — The old explicit `ZmqEventRecordPublisher(name, ctx, loop=loop)` was replaced with `EventPublisherService(ctx)` which implicitly uses `LoopManager().default_loop`. Functionally equivalent but less transparent.

---

## 🔵 Consider (low)

| #   | File                      | Line | Category     | Summary                                                                                                        |
| --- | ------------------------- | ---- | ------------ | -------------------------------------------------------------------------------------------------------------- |
| 7   | `event_publisher.py`      | 36   | api-contract | `send_threshold` parameter not documented in docstring                                                         |
| 8   | `session.py`              | 131  | api-contract | `flush()` added to `EventPublisher` protocol — breaking change for any external implementations                |
| 9   | `test_event_publisher.py` | 226  | testing      | No dedicated test for: threshold-triggered auto-flush, mixed event types in batch, batch + single interleaving |

---

## Summary

The batching design is sound — 19x throughput improvement, 29% smaller wire size, ordering preserved via insertion-order list encoding. The main issues are:

1. **Topic filtering incompatibility** (#1) — needs either subscriber-side filtering or a documented restriction
2. **Close ordering** (#2) — simple fix: flush before `is_closed = True`
3. **Silent batch drop** (#3) — should at least log estimated record count

Generated with [Claude Code](https://claude.com/claude-code)
