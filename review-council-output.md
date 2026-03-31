# Review Council — PR "Make Loadgen Async" (Round 4)

Reviewed by: **Claude** (Codex produced no structured findings) | Depth: **thorough** | Mode: **--no-post**

Focus areas: data flow correctness after metrics/report fixes, remaining bugs, testing gaps.

## 🔴 Must Fix (critical/high)

| # | File | Line | Category | Summary |
|---|------|------|----------|---------|
| 1 | `session.py` | 437-451 | bug | Double-decrement of `inflight` for streaming queries — `StreamChunk(is_complete=True)` and subsequent `QueryResult` both decrement. |

### Detail

**#1 — Double-decrement of inflight for streaming queries**

For streaming responses, the worker sends intermediate `StreamChunk` messages, then a final `StreamChunk(is_complete=True)`, then a `QueryResult` from `get_final_output()`. Both the terminal `StreamChunk` (line 445) and the `QueryResult` (line 415) decrement `phase_issuer.inflight` and call `on_query_complete()`. This causes:

- `inflight` goes negative
- Premature `drain_event.set()` (drain completes before all responses are in)
- Double `on_query_complete()` on ConcurrencyStrategy releases the semaphore twice, corrupting concurrency control
- Double `on_sample_complete` callback fires for the same query

Fix: Remove the inflight tracking from the `StreamChunk(is_complete=True)` path. The `QueryResult` that follows will handle it. Or track completed query_ids and skip duplicates.

---

## 🟡 Should Fix (medium)

| # | File | Line | Category | Summary |
|---|------|------|----------|---------|
| 2 | `kv_store.py` | 288 | api-contract | `_SeriesItem.append()` uses `type(value) != self._dtype` — rejects numpy int64, bool subclasses. |
| 3 | `kv_store.py` | 296 | concurrency | ARM mmap ordering: `msync()` is not a memory barrier — reader may see stale values on ARM. |
| 4 | `session.py` | 424 | bug | If a custom accumulator doesn't set `first_chunk` metadata, all chunks are `RECV_NON_FIRST` — TTFT/TPOT silently dropped. |
| 5 | `execute.py` | 371-394 | data-integrity | `_setup_kv_reader` hardcodes metric key strings — should use `MetricCounterKey`/`MetricSeriesKey` enums. |
| 6 | `execute.py` | 510 | error-handling | SIGINT during cleanup (`launcher.wait_for_exit`) may hang up to 10s if services don't process ENDED fast enough. |

### Detail

**#2 — Strict type identity check rejects valid subtypes**

`type(value) != self._dtype` uses identity, not isinstance. `numpy.int64`, `bool` (subclass of int), etc. are rejected. Use `isinstance(value, self._dtype)` or coerce with `self._dtype(value)`.

**#3 — ARM mmap ordering (known limitation)**

The code documents this (lines 299-311) and uses `msync()` as mitigation, but `msync` is a filesystem flush, not a memory barrier. On ARM (aarch64 clusters, Apple Silicon dev machines), a reader process could observe the incremented count before the value is visible. In practice, the reader runs after the aggregator exits (post-fix from this PR), so this is mitigated. But concurrent reads during the benchmark would be unsafe on ARM.

**#4 — Missing RECV_FIRST event for non-OpenAI accumulators**

If a custom or third-party accumulator doesn't set `metadata["first_chunk"]`, the session defaults to `False` and all chunks are published as `RECV_NON_FIRST`. The MetricsAggregator never sees `RECV_FIRST`, so TTFT and TPOT are silently zero. Consider logging a warning when COMPLETE arrives without a prior RECV_FIRST for a streaming sample.

**#5 — Hardcoded metric key strings**

`_setup_kv_reader` uses string literals (`"tracked_samples_issued"`, `"ttft_ns"`, etc.) that must match enum values. If an enum value is renamed, the reader silently reads 0/empty. Import and iterate over the enums directly.

**#6 — SIGINT during cleanup**

`launcher.wait_for_exit(10.0)` blocks for up to 10 seconds. If the user hits Ctrl+C during this, the default SIGINT handler (removed at line 525) is no longer installed — a second SIGINT would raise `KeyboardInterrupt` unhandled inside the `finally` block. This is unlikely but could leave orphaned service processes.

---

## 🔵 Consider (low)

| # | File | Line | Category | Summary |
|---|------|------|----------|---------|
| 7 | `session.py` | 462 | design | `max_duration_ms=0` means "no limit" (same as None) — could be confusing. |
| 8 | `session.py` | 353 | design | No drain timeout — documented as intentional, but transport message loss causes indefinite hang. |
| 9 | `scoring.py` | 124 | error-handling | KeyError on missing dataset name gives no context (available keys). |
| 10 | `execute.py` | 423 | performance | macOS fallback to `tempfile.gettempdir()` (no `/dev/shm`) means mmap files on disk with msync overhead. |
| 11 | `strategy.py` | 281 | design | `match` statement in factory (cold path — acceptable per AGENTS.md). |

---

## Testing Coverage Notes

The new tests (19 added) cover strategies and session well. Remaining gaps:

| Scenario | Coverage |
|----------|----------|
| Streaming + session (StreamChunk flow) | Partial — stale chunk tested but double-decrement not caught |
| ARM mmap reader correctness | Not tested (x86 test env) |
| Custom accumulator without `first_chunk` metadata | Not tested |
| SIGINT during drain | Not tested |
| Concurrent KVStore reader + writer | Not tested (reader runs after writer exits) |

---

## Progress Since Round 3

Issues fixed since the last review:

- ✅ **#1 (Round 3)** Report snapshot race — moved report build after `launcher.wait_for_exit()`
- ✅ **#3 (Round 3)** PromptData drops token_ids — now passes `input_tokens`/`token_ids` to PromptData
- ✅ **#5 (Round 3)** 60s drain timeout — removed, now waits indefinitely
- ✅ **#6 (Round 3)** sample_idx_map name collision — uses full dataset name
- ✅ **ReadyCheck ENOTSOCK** — socket no longer closed on TimeoutError
- ✅ **MetricSeriesKey filename** — enum resolved to `.value` in EmitTrigger
- ✅ **Endpoint URL validation** — rejects URLs without `http://`/`https://` scheme
- ✅ **report.txt** — human-readable report written to report directory
- ✅ **Crash recovery** — `_salvage_tmpfs` copies logs on interrupt before cleanup

New issue found: **#1 (Round 4) double-decrement** is critical and should be fixed before merge.

Generated with [Claude Code](https://claude.com/claude-code)
