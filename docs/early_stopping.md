# Early stopping — design

Optional, default-off feature that adds an MLPerf-LoadGen-style **early-stopping percentile
estimate** to tail-latency metrics (TTFT / TPOT / total latency). It reports a *conservative,
confidence-backed* percentile alongside the empirical one, so a run with too few samples to trust its
raw p99/p90 is surfaced honestly instead of silently under-reporting the tail.

It is an **estimate only** — no target-latency pass/fail, no dynamic mid-run halt. See the study /
gap analysis in the companion `endpoints-early-stopping` repo.

## What it computes

For target percentile `p`, confidence `c` (default 0.99), tolerance `d` (default 0.0), over the `n`
ascending-sorted latencies of a series:

```
estimate = sorted[n - t],  t = max{ i : n >= find_min_passing(i, p, d, c) + i }
```

This is LoadGen's SingleStream estimate (`results.cc:162-226`): a value the true p-percentile is `<=`
at confidence `c`, always `>=` the empirical percentile. Below the floor
`find_min_passing(1, p, d, c) + 1` (662 for p99) the estimate is `None` (too few samples). The
binomial math lives in `metrics/early_stopping.py` (fast `betai`, no scipy; validated against LoadGen
values, e.g. `h_min(t=0,p99)=459`).

## Layering (who owns what)

```
config/schema.py            EarlyStoppingConfig (Pydantic, enabled=False)  <- the YAML flag
  -> commands/benchmark/execute.py   passes it to the aggregator subprocess as CLI args
    -> metrics_aggregator/__main__.py   parses args -> EarlyStoppingSpec -> MetricsRegistry
      -> metrics_aggregator/registry.py   SeriesSampler.build_stat(exact=True) computes the estimate
                                          (COMPLETE path only; the raw sorted array lives here)
        -> metrics_aggregator/snapshot.py   SeriesStat.early_stopping (optional trailing field)
          -> metrics/report.py   surfaces the block into result_summary.json + the text report
metrics/early_stopping.py   pure math (find_min_passing / es_percentile_estimate) — no I/O, unit-tested
```

**Why the aggregator computes it:** the estimate needs the full sorted raw latency array, which only
exists in the aggregator subprocess (`SeriesSampler._raw`). Only the summarized `SeriesStat` crosses
to the main process, so the estimate must be produced before that boundary — in `build_stat`, on the
COMPLETE (exact) path. Hot path is untouched; this is cold-path work at run end.

**Target metrics:** `ES_TARGET_METRICS = {ttft_ns, tpot_ns, sample_latency_ns}`. Only these
tail-latency series get an estimate; counters / ISL / OSL do not.

## Config (YAML)

```yaml
settings:
  early_stopping:
    enabled: false        # default off
    percentile: 0.99      # target: 0.99 for Server/interactive p99, 0.90 for SingleStream/T2V
    confidence: 0.99      # MLPerf default
    tolerance: 0.0        # MLPerf default (d=0)
```

## Output (`result_summary.json`)

Each of `ttft`/`tpot`/`latency` gains an `early_stopping` block (present only when enabled):

```json
"tpot": {
  "percentiles": { "99.0": 71908203.4, ... },
  "early_stopping": {
    "percentile": 0.99, "confidence": 0.99, "tolerance": 0.0,
    "n": 40000, "estimate": 71997480.0, "empirical": 71908203.4,
    "sufficient": true, "min_queries": 662, "discarded": 353
  }
}
```
`sufficient=false` with `estimate=null` means the run had fewer than `min_queries` samples to claim
the percentile at the requested confidence. The text report renders one extra line per metric.

## Explicitly out of scope

- No `target_latency` pass/fail gate (LoadGen's Server ES path).
- No dynamic mid-run halt — evaluated once at COMPLETE.
- Offline/max_throughput scenarios are not tail-latency-bounded, so ES is a no-op there even if enabled.
