# Report Design

## Overview

The report module provides benchmark result summarization, display, and
serialization. It reads from the KVStore (via `BasicKVStoreReader`) and
produces a `Report` with rollup statistics, percentiles, and histograms.

## Architecture

```
BasicKVStoreReader.snapshot()
        │
        ▼
  build_report(reader)
        │
        ├── counters → n_issued, n_completed, n_failed, duration_ns
        │
        └── for each series metric:
              SeriesStats.values → compute_summary() → dict
        │
        ▼
     Report
        ├── .display(fn)     → human-readable output
        ├── .to_json(path)   → JSON serialization
        ├── .qps             → computed from n_completed / duration
        └── .tps             → computed from osl total / duration
```

## Design Principles

**No SQL, no UUID tracking, no deduplication.**

The old `MetricsReporter` queried SQLite via duckdb and built `RollupQueryTable`
objects with UUID-indexed rows, repeat counts, and numpy sorted arrays. None of
this complexity is needed when the input is a `list[float]` from the KVStore.

The entire rollup is a single function: `compute_summary(values) → dict`.
It calls numpy for percentiles and histograms. No classes, no state.

## Components

### `compute_summary(values, percentiles, n_histogram_buckets) → dict`

Takes a `list[float]`, returns a dict with:

- `total`, `min`, `max`, `avg`, `std_dev`, `median`
- `percentiles`: dict of `{str(p): float}` for each requested percentile
- `histogram`: `{"buckets": [(lo, hi), ...], "counts": [int, ...]}`

Empty input returns zeros with empty histogram/percentiles.

### `Report` (frozen dataclass)

Fields:

- `version`, `git_sha`, `test_started_at`
- `n_samples_issued`, `n_samples_completed`, `n_samples_failed`
- `duration_ns`
- `ttft`, `tpot`, `latency`, `output_sequence_lengths` — each a summary dict

Properties:

- `qps`: `n_samples_completed / (duration_ns / 1e9)`, or None
- `tps`: `osl_total / (duration_ns / 1e9)`, or None

Methods:

- `display(fn, summary_only, newline)` — human-readable output with histograms
- `to_json(save_to)` — JSON serialization with QPS/TPS included

### `build_report(reader) → Report`

Reads a `BasicKVStoreReader` snapshot and constructs a `Report`. Works
identically for live metrics (mid-test) and final reports (post-drain) —
the caller decides when to call it.

Counter keys read: `n_samples_issued`, `n_samples_completed`,
`n_samples_failed`, `duration_ns`, `test_started_at`.

Series keys summarized: `ttft_ns`, `tpot_ns`, `sample_latency_ns`, `osl`.

## What Was Removed

From the old `metrics/reporter.py`:

- `MetricsReporter` — SQLite/duckdb query engine (replaced by KVStore)
- `RollupQueryTable` — UUID-indexed rollup table (replaced by `compute_summary`)
- `MetricRow` — per-row accessor (not needed)
- `TPOTReportingMode` — niche enum (can be re-added if needed)
- `SampleUUIDNotFoundError` — UUIDs not relevant in KVStore
- `output_sequence_from_data` — SQL event data parser (not needed)
- `dump_to_json` / `dump_all_to_csv` — event log export (handled by EventLoggerService)
