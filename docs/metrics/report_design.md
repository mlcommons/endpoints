# Report Design

Builder-level reference for `metrics/report.py`. For where the report sits in the run pipeline,
see [metrics/DESIGN.md](DESIGN.md); for the snapshot it consumes, see the
[metrics aggregator spec](../async_utils/services/metrics_aggregator/DESIGN.md).

## Overview

`report.py` provides benchmark result summarization, display, and serialization. Its input is the
terminal metrics-aggregator snapshot in **dict** form (`snapshot_to_dict`, as persisted to
`final_snapshot.json`); its output is a `Report` carrying rollup statistics, percentiles, and
histograms, rendered to console and to `result_summary.json`.

## Architecture

```
final_snapshot.json  (dict form of MetricsSnapshot)
        │
        ▼  json.loads(...)  → dict
  Report.from_snapshot(snap, run_config=..., use_legacy_loadgen_qps_metrics=...)
        │
        ├── counters  → n_samples_issued/completed/failed, tracked_duration_ns,
        │                legacy_loadgen_window_duration_ns, finish-reason counts
        │
        ├── for each series (ttft, tpot, latency, osl):
        │       _series_to_metric_dict(stat) → rollup dict
        │
        └── derive qps / tps once (window chosen by run config)
        │
        ▼
     Report (frozen msgspec.Struct)
        ├── .display(fn)   → human-readable output with histograms
        └── .to_json(path) → result_summary.json (QPS/TPS + run_config included)
```

## Design Principles

**The dict snapshot is the sole input.** `from_snapshot` consumes the `snapshot_to_dict` shape
directly and never decodes the wire `MetricsSnapshot` Struct (which is `array_like=True` for compact
msgpack). A consumer feeds `json.loads(path.read_bytes())` straight in. This lets a report be
rebuilt offline from any persisted `final_snapshot.json`.

**Honest incompleteness over crashes.** Every counter and series read uses `.get(...)` with a safe
default, so a truncated or partial (INTERRUPTED) snapshot produces an honest _empty_ rollup and a
`complete=False` report rather than raising. Missing `state` defaults to `"interrupted"`.

**Self-describing artifacts.** Derived QPS/TPS, the window that produced them
(`legacy_loadgen_window_duration_ns`), and `run_config` are all serialized, so a valid run is fully
identified by its own `result_summary.json`.

## Components

### `_series_to_metric_dict(stat) → dict`

Converts one series-stat dict from the snapshot into the rollup shape `display()` expects:

- `total`, `min`, `max`, `avg`, `std_dev`, `median`
- `percentiles`: `{str(p): float}` for each requested percentile
- `histogram`: `{"buckets": [(lo, hi), ...], "counts": [int, ...]}`
- `early_stopping_percentiles` (optional): MLPerf early-stopping estimate map, placed right after
  `percentiles`; present only when early stopping is enabled (COMPLETE snapshots for ttft/tpot/latency)

`avg`/`std_dev`/`median` are derived from the cheap rollups + percentiles (integer-exact variance
form for ns series). `median` falls back to `(min + max) / 2` only for hand-crafted dicts that omit
p50. A zero-count series returns `{}` (or an all-null early-stopping map if the feature is enabled).

### `Report` (frozen `msgspec.Struct`)

Fields: `version`, `git_sha`, `test_started_at`, `n_samples_issued/completed/failed`,
`duration_ns`, `state`, `complete`, the four rollup dicts (`ttft`, `tpot`, `latency`,
`output_sequence_lengths`), `legacy_loadgen_window_duration_ns`, `qps`, `tps`,
`finish_reason_counts`, `run_config`, and `accuracy`.

- `complete` = `state == "complete" and n_pending_tasks == 0` — `False` marks partial async metrics
  (drain timeout, interrupt, or a live-tick fallback when no final snapshot was found).
- `qps`/`tps` are computed once in `from_snapshot` (see below) rather than as live properties, so
  the serialized report is self-complete.
- `accuracy` is empty from `from_snapshot`; per-dataset entries are attached after scoring.

Methods: `display(fn, ...)` for console output with histograms; `to_json(save_to)` for
`result_summary.json`.

### `Report.from_snapshot(snap, *, run_config=None, use_legacy_loadgen_qps_metrics=True) → Report`

Splits the snapshot's `metrics` list into counters and series, then builds the `Report`.

Counter keys read: `tracked_samples_issued`, `tracked_samples_completed`,
`tracked_samples_failed`, `tracked_duration_ns`, `legacy_loadgen_window_duration_ns`, and the
`tracked_finish_reason_*` counters. Series read (snapshot key → report field, via
`SERIES_TO_SUMMARY_FIELD`): `ttft_ns` → `ttft`, `tpot_ns` → `tpot`, `sample_latency_ns` →
`latency`, and `osl` → `output_sequence_lengths`.

**QPS/TPS window selection.** The snapshot always carries both a native window
(`tracked_duration_ns`) and the MLPerf LoadGen "completed" window
(`legacy_loadgen_window_duration_ns`, poisson only), so it stays reinterpretable either way. The
legacy window drives the headline QPS/TPS only when it is enabled
(`use_legacy_loadgen_qps_metrics`), available, and there are ≥2 completions
(`QPS = (completed - 1) / window`). Otherwise both QPS and TPS fall back to the native window so
they always share one window, and `legacy_loadgen_window_duration_ns` is left `None` so the
serialized report records which view it holds. `tps` is `None` when no OSL was recorded.
