# PR #255 Follow-Up Items

Unresolved PR review comments and review council findings deferred from the async loadgen PR.

## Medium Priority

### 1. Use enums for KV metric key registration
- **Source**: arekay-nv (`execute.py:385`), viraatc (`execute.py:378`), review council round 4 #5
- **Issue**: `_setup_kv_reader` hardcodes metric key strings (`"tracked_samples_issued"`, `"ttft_ns"`, etc.). If an enum value is renamed, the reader silently reads 0/empty.
- **Fix**: Import `MetricCounterKey` and `MetricSeriesKey` enums and iterate over them. Consider a typed tuple `(key, type, dtype, streaming_only)` for each metric.

### 2. Rethink PhaseType enum
- **Source**: arekay-nv (`session.py:57`)
- **Issue**: Performance and accuracy phases differ only in whether metrics are tracked. Saturation just lacks a drain barrier. The enum conflates scheduling behavior with metric tracking.
- **Suggestion**: Expose tracking and drain as independent knobs with sensible defaults per phase type.

### 3. Support multiple performance datasets
- **Source**: arekay-nv (`execute.py:346`)
- **Issue**: `_build_phases` creates a single performance phase. Configs already support multiple perf datasets but only the first is used.
- **Fix**: Iterate over all performance datasets and create a phase for each.

### 4. Rename report field for clarity
- **Source**: arekay-nv (`execute.py:650`)
- **Issue**: `n_samples_issued` in the results JSON is ambiguous — it refers to perf-phase-only tracked samples, not total.
- **Fix**: Rename to `perf_samples_issued` or `tracked_samples_issued`.

### 5. Consolidate teardown logic
- **Source**: arekay-nv (`execute.py:495`)
- **Issue**: Error handling in the `except` block duplicates cleanup that also happens in `finally`. The `except` block calls `pbar.close()`, `publisher.close()`, `launcher.kill_all()` — some of which also run in `finally`.
- **Fix**: Move all teardown to the `finally` block, remove duplicates from `except`.

### 6. Tmpfs directory ownership
- **Source**: arekay-nv (`execute.py:580`)
- **Issue**: `_write_scoring_artifacts` calls `shutil.rmtree(tmpfs_dir)` but didn't create the directory. Directory removal should be handled by the same code that created it.
- **Fix**: Move tmpfs cleanup to `run_benchmark`'s `finally` block (partially done — `_salvage_tmpfs` + cleanup already there, but `_write_scoring_artifacts` also cleans up).

### 7. Log stop-check reason
- **Source**: arekay-nv (`session.py:481`)
- **Issue**: When `_make_stop_check` returns True, the caller doesn't know why (all samples issued vs max_duration reached vs stop requested). Makes debugging long runs harder.
- **Fix**: Log the reason in the stop check closure or return an enum.

### 8. Warn on missing tokenizer
- **Source**: arekay-nv (`execute.py:296`)
- **Issue**: If the HuggingFace model name has a typo, `_check_tokenizer_exists` returns False and TPOT/OSL metrics silently fail later (no tokenizer available in the aggregator).
- **Fix**: Log a warning when tokenizer is not found, suggesting the user check the model name.

## Low Priority

### 9. Accuracy-only benchmark support
- **Source**: viraatc (`execute.py:342`)
- **Tracked in**: `docs/accuracy_only_support.md`

### 10. Use ZMQ context scope in workers
- **Source**: viraatc (`execute.py:535`)
- **Status**: Addressed for the main process (`ManagedZMQContext.scoped()`). Worker processes already use `scoped()` via `_ZmqWorkerConnector.connect()`.

## Known Limitations (documented, not actionable now)

- **Report counter asymmetry**: `total_samples_failed` includes all phases; `tracked_*` counters are perf-only. No `tracked_samples_failed` counter exists yet.
- **Shared RNG across phases**: Accuracy phases reuse the same `random.Random` instances as the perf phase. Ordering depends on how many draws perf consumed.
- **Double SIGINT leaks resources**: Second Ctrl+C during `loop.run_until_complete` raises `KeyboardInterrupt` bypassing the `finally` block.
- **`load_sample()` before ISSUED timestamp**: Dataset reads happen synchronously before recording the ISSUED timestamp. Datasets must be pre-loaded into memory for accurate timing.
- **ARM mmap ordering**: On ARM with tmpfs, `msync()` is a no-op and cannot provide write ordering. An on-disk metrics directory is required for correct cross-process reads. See `fs_check.py`.
