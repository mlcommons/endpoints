# Metrics — Design Spec

> Builds the final benchmark `Report` from the metrics-aggregator's rollup
> snapshot and renders it (console + `result_summary.json`). Also holds the
> metric target types, the MLPerf early-stopping percentile math, and the
> standardized run-artifact plots.

**Component specs:** [async_utils](../async_utils/DESIGN.md) ·
[commands](../commands/DESIGN.md) · [config](../config/DESIGN.md) ·
[core](../core/DESIGN.md) · [dataset_manager](../dataset_manager/DESIGN.md) ·
[endpoint_client](../endpoint_client/DESIGN.md) ·
[evaluation](../evaluation/DESIGN.md) ·
[load_generator](../load_generator/DESIGN.md) · **metrics** ·
[openai](../openai/DESIGN.md) · [plugins](../plugins/DESIGN.md) ·
[profiling](../profiling/DESIGN.md) · [sglang](../sglang/DESIGN.md) ·
[testing](../testing/DESIGN.md) · [utils](../utils/DESIGN.md)

---

## Overview

`metrics/` is the **reporting** side of the pipeline: it consumes the rollup
snapshot produced by the metrics-aggregator subprocess and turns it into a
human-readable console report and a machine-readable `result_summary.json`.

It does **not** record or aggregate events itself. During a run, events are
published over ZMQ by the load generator and folded into per-sample rollups by a
separate
[`MetricsAggregatorService`](../async_utils/services/metrics_aggregator/DESIGN.md)
subprocess, which publishes `MetricsSnapshot` messages and atomically writes the
terminal snapshot to `final_snapshot.json`. `metrics/` reads that snapshot and
builds the `Report`.

## Responsibilities

- Turn a metrics-aggregator snapshot dict into a `Report` (`report.py`,
  `Report.from_snapshot`)
- Derive headline QPS/TPS and E2E average interactivity once, so the serialized
  report is self-complete
- Render the report to console (`display`) and to `result_summary.json`
  (`to_json`)
- Define the metric **target** types used by rulesets (`metric.py`)
- Compute MLPerf LoadGen early-stopping percentile estimates
  (`early_stopping.py`)
- Produce standardized run-artifact plots (`results_plots.py`)

## Data Flow

```
load_generator  ──►  EventPublisher (events PUB)
                          │
                          ▼
        MetricsAggregatorService (subprocess)
          folds events → MetricsRegistry
          publishes MetricsSnapshot (metrics PUB)
          atomically writes final_snapshot.json  ◄── primary Report source
                          │
                          ▼
        main process (commands/benchmark/execute.py)
          json.loads(final_snapshot.json)  → dict
                          │
                          ▼
        Report.from_snapshot(dict, run_config=..., token_count_source=...)
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Report.display(fn)      Report.to_json(result_summary.json)
```

The snapshot is consumed as a **dict** (`json.loads`), not decoded back into the
wire `MetricsSnapshot` Struct — the wire form is `array_like=True` for compact
msgpack, and the dict form (`snapshot_to_dict`) is the canonical consumer
contract. If `final_snapshot.json` is missing (e.g. the aggregator was SIGKILLed
before its signal handler ran), the caller falls back to the subscriber's last
live snapshot and the resulting `Report.complete` is `False`.

## Files

| File                   | Purpose                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `report.py`            | `Report` Struct + `Report.from_snapshot(dict)`; console `display()` and `to_json()` serializer                     |
| `metric.py`            | Metric target types (`Throughput`, `QueryLatency`, `TTFT`, `TPOT`) used by rulesets                                |
| `early_stopping.py`    | MLPerf LoadGen early-stopping percentile estimates (pure math); see [docs/early_stopping.md](../early_stopping.md) |
| `results_plots.py`     | Standardized run-artifact plots (matplotlib-guarded); CLI: `scripts/plot_results.py`                               |
| `token_count_basis.py` | Usage classification rule, effective basis, and the run's basis (summary, else `config.yaml`) for post-run tools   |

## Public Interface

### `Report` (`report.py`)

`Report` is a frozen `msgspec.Struct`. It is built by `from_snapshot` and is the
single object that `display()` and `to_json()` render.

```python
class Report(msgspec.Struct, frozen=True):
    version: str
    git_sha: str | None
    test_started_at: int
    n_samples_issued: int
    n_samples_completed: int
    n_samples_failed: int
    duration_ns: int | None
    state: str            # terminal SessionState string ("complete" / "interrupted")
    complete: bool        # per the primary basis: server = state == "complete", valid
                          # accounting, no drops AND n_samples_missing_usage == 0;
                          # client = same, with n_pending_tasks == 0 instead of the
                          # usage check
    ttft: dict            # per-metric rollup dicts (min/max/mean/percentiles/...)
    tpot: dict
    latency: dict
    output_sequence_lengths: dict
    legacy_loadgen_window_duration_ns: int | None = None
    qps: float | None = None
    tps: float | None = None
    e2e_avg_interactivity: float | None = None
    finish_reason_counts: dict[str, int] = {}
    run_config: dict | None = None          # config (load pattern, warmup, RNG seeds)
    accuracy: list[dict] = []               # per-dataset accuracy, attached post-scoring
    token_count_source: TokenCountSource    # primary basis of tpot/ISL/OSL/tps: SERVER or CLIENT
    n_samples_missing_usage: int = 0        # successful tracked samples with no usable server usage
    n_samples_missing_usage_empty_output: int = 0   # ... of which had empty output
    client_token_tasks_pending: int = 0     # tokenizer tasks pending at the final snapshot
    other_basis: OtherBasisMetrics          # the other basis's ISL/OSL/TPOT blocks and TPS
    osl_paired: OslPaired | None = None     # OSL totals/mean diff/ratio over paired samples

    @classmethod
    def from_snapshot(
        cls, snap: dict, *, token_count_source: TokenCountSource,
        run_config: dict | None = None,
        use_legacy_loadgen_qps_metrics: bool = True,
    ) -> "Report": ...

    def incomplete_reason(self) -> str | None: ...   # cause of complete == False

    def to_json(self, save_to: os.PathLike | None = None) -> bytes: ...
    def display(self, ...) -> None: ...
```

**`from_snapshot`** reads counters and per-series rollups out of the snapshot
dict, all via `.get(...)` with safe defaults so a partial/malformed snapshot
yields an honest _incomplete_ report instead of crashing (missing `state`
defaults to `"interrupted"`). It derives headline throughput once:

- The snapshot always carries **both** `tracked_duration_ns` (endpoints-native
  full-run window) and `legacy_loadgen_window_duration_ns` (MLPerf LoadGen
  "completed" window, poisson only). It is therefore config-agnostic and
  reinterpretable either way.
- Which window drives the headline QPS/TPS is decided by the run config
  (`use_legacy_loadgen_qps_metrics`), not by the snapshot. The legacy window is
  used only when it is enabled, available, and there are ≥2 completions
  (`QPS = (completed-1)/window`); otherwise both QPS and TPS fall back to the
  native window so they always share one window, and
  `legacy_loadgen_window_duration_ns` is left `None` so the serialized report
  honestly records which view it holds.
- `tps` is additionally `None` when no OSL was recorded (every output empty, or
  a client primary has no tokenizer configured). OSL is captured on `COMPLETE`
  regardless of streaming, so non-streaming runs still get a TPS. With a server
  primary the OSL is the server's `completion_tokens`, so `tps` needs no
  tokenizer. `other_basis.tps` is the other basis's TPS over the same window; it
  is `None` whenever that basis is partial (for a client other basis,
  `client_token_tasks_pending > 0`).
- For failure-free runs, `e2e_avg_interactivity` is total output tokens (on the
  primary basis) divided by summed end-to-end sample latency in seconds. It is
  `None`/`N/A` when no OSL or positive sample-latency total is available, or
  when any tracked request failed. Failed responses contribute to
  `sample_latency_ns` but not OSL, so reporting the ratio in that case would
  combine different sample populations.

**Token-count basis.** The snapshot does not record where its token counts came
from, so the caller passes the run's effective `token_count_source` as a
required keyword (the primary basis). The headline `tpot`,
`input_sequence_lengths`, `output_sequence_lengths`, `tps`,
`e2e_avg_interactivity`, early-stopping maps and steady-state block are on the
primary basis; the other basis fills `other_basis`, serialized flat under
`_<basis>`-suffixed names (`tpot_server` beside a client primary, `tpot_client`
beside a server primary), and `osl_paired` compares the two over samples counted
on both. `complete` follows the primary basis: with a server primary it is
`False` when any successful tracked sample reported no usage, and with a client
primary when tokenization is still pending. `incomplete_reason()` names the
cause (missing usage, a client-basis tokenizer drain, or an incomplete state).
The other basis falling short leaves `complete` `True` and marks its blocks
`"partial": true`. See
[report_design.md](report_design.md#server-vs-client-token-counts) for the
systematic differences between the two bases.

**Accuracy** is not part of the metrics snapshot; `from_snapshot` leaves
`accuracy` empty and the finalizer attaches per-dataset accuracy entries after
scoring. `run_config` is supplied by the caller (it is config, not a measured
metric).

### Metric target types (`metric.py`)

`Throughput`, `QueryLatency`, `TTFT`, and `TPOT` are the metric **targets** a
ruleset validates a run against — not the measured rollups (those live in the
snapshot / `Report`). Each stores its target on the base `Metric` and exposes
`is_valid(measurement) -> bool`; `Throughput`/`QueryLatency` apply a relative
tolerance, while `TTFT`/`TPOT` are hard latency ceilings.

## Design Decisions

**Reporting is decoupled from aggregation.** Event recording, tokenization, and
rollup live in the aggregator subprocess; `metrics/` only consumes the finished
snapshot. This keeps the hot path (the aggregator) isolated from report
formatting and lets the report be rebuilt offline from a persisted
`final_snapshot.json`.

**The dict snapshot is the contract.** `from_snapshot` takes the
`snapshot_to_dict` shape and never decodes the wire Struct — see the data-flow
note above. `result_summary.json` is self-complete (carries derived QPS/TPS, E2E
average interactivity, and `run_config`) so a valid run is fully described by
its own artifact.

**Honest incompleteness over crashes.** Every snapshot read defaults safely;
client-primary drain-timeout, server-primary missing-usage and interrupted runs
produce a `Report` with `complete=False` and an explicit indicator in
`display()` rather than a partial success that looks clean.

## Integration Points

| Component                                                                    | Role                                                                                                             |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| [`metrics_aggregator`](../async_utils/services/metrics_aggregator/DESIGN.md) | Produces the `MetricsSnapshot` / `final_snapshot.json` that `from_snapshot` consumes                             |
| `commands/benchmark/execute.py`                                              | Launches the aggregator subprocess, reads `final_snapshot.json`, builds/attaches the `Report`, attaches accuracy |
| `config/` (`use_legacy_loadgen_qps_metrics`)                                 | Selects which duration window drives headline QPS/TPS                                                            |
| `scripts/plot_results.py`                                                    | CLI entry point over `results_plots.py`                                                                          |
