# Early stopping — design

On-by-default feature that adds an MLPerf-LoadGen-style **early-stopping percentile estimate** to
tail-latency metrics (TTFT / TPOT / total latency). It reports a _conservative, confidence-backed_
percentile alongside the empirical one, so a run with too few samples to trust its raw p99/p90 is
surfaced honestly instead of silently under-reporting the tail. It is on by default because it is
non-invasive — computed once at run COMPLETE from data the aggregator already keeps (the exact path
sorts that array once and shares it between the percentile grid and the estimates), with the hot
path untouched and the output field purely additive.

It is an **estimate only** — no target-latency pass/fail, no dynamic mid-run halt. The gap analysis
motivating it: at large n the estimate merely certifies the empirical percentile (a small
conservative shift), while on short runs — where the empirical tail is optimistically noisy — the
estimate stays honest, which is exactly the regime edge/T2V workloads live in.

## What it computes

For each target percentile `p` — every entry of the series' own report percentile grid at or above
the median (`es_targets_from_grid`; the default grid yields p50/p75/p80/p90/p95/p97/p99/p99.9) —
at confidence `c = 0.99`, over the `n` ascending-sorted latencies of a series
(percentiles use the grid convention — 0-100 — at every surface; only the LoadGen-parity
kernel keeps LoadGen's fraction domain, converted exactly once and unrounded, the same `/100`
`np.percentile` applies internally):

```
estimate = sorted[n - t],  t = max{ i : n >= find_min_passing(i, p, d, c) + i }
```

This is LoadGen's SingleStream estimate (`results.cc:162-226`): a value the true p-percentile is `<=`
at confidence `c`, always `>=` the empirical percentile. Below the floor
`find_min_passing(1, p, d, c) + 1` (662 for p99, 64 for p90) the estimate is `None` (too few
samples). The binomial math lives in `metrics/early_stopping.py` (fast `betai`, no scipy; validated
against LoadGen values, e.g. `h_min(t=0,p99)=459`).

**Confidence and tolerance are algorithm constants, not configuration** (`CONFIDENCE = 0.99`,
`TOLERANCE = 0.0` in `metrics/early_stopping.py`). LoadGen hardcodes both (`results.cc:157-158`);
lowering `c` or raising `d` weakens the certified claim (`d > 0` certifies percentile `p − d`).
The pure math keeps defaulted arguments for parity tests only.

**The percentile targets are not a separate list either** — they derive from the series' report
percentile grid, filtered to `≥ p50` (`ES_MIN_PERCENTILE = 50.0`, grid convention): the estimate is a _tail_
certification (a conservative upper confidence bound), so below-median grid entries are skipped.
One source of truth: whatever percentiles a series reports, ES covers — every scenario's gate
percentile (p99 Server, p90 SingleStream/T2V) is always included, with nothing to tune.

Each estimate is a _marginal_ `c`-confidence statement per percentile. Reporting several at once is
fine for diagnostics, but a joint gate across all of them holds at lower than `c` confidence
(multiple testing) — compliance gates should use the single scenario percentile.

## Layering (who owns what)

```
config/schema.py            EarlyStoppingConfig (Pydantic, enabled=True)   <- the YAML opt-out
  -> commands/benchmark/execute.py   passes it to the aggregator subprocess as CLI args
    -> metrics_aggregator/__main__.py   parses args -> EarlyStoppingSpec -> MetricsRegistry
      -> metrics_aggregator/registry.py   SeriesSampler.build_stat(exact=True) computes the estimate
                                          (COMPLETE path only; the raw sorted array lives here)
        -> metrics_aggregator/snapshot.py   SeriesStat.early_stopping_percentiles (trailing field)
          -> metrics/report.py   surfaces the block into result_summary.json + the text report
metrics/early_stopping.py   pure math (find_min_passing / es_percentile_estimate) — no I/O, unit-tested
```

**Why the aggregator computes it:** the estimate needs the full sorted raw latency array, which only
exists in the aggregator subprocess (`SeriesSampler._raw`). Only the summarized `SeriesStat` crosses
to the main process, so the estimate must be produced before that boundary — in `build_stat`, on the
COMPLETE (exact) path. Hot path is untouched; this is cold-path work at run end.

**Target metrics:** any series registered with `register_series(..., tail_latency=True)` — today
`ttft_ns`, `tpot_ns`, and `sample_latency_ns` (see the aggregator's registration block). The flag
lives at the metric's definition site, so a new latency metric opts in where it is declared;
counters / ISL / OSL simply don't set it.

## Config (YAML)

```yaml
settings:
  early_stopping:
    enabled: true # default; the single opt-out (or --no-early-stopping on the CLI)
```

The only knob is the opt-out — for consumers that strictly validate the `result_summary.json`
schema. Everything else is a constant (see above), so there is nothing to tune per config and no
way to accidentally weaken the statistical claim.

## Output (`result_summary.json`)

Each of `ttft`/`tpot`/`latency` gains an `early_stopping_percentiles` map (COMPLETE snapshots only, for the series the run recorded; absent when opted out) — keys mirror the `percentiles` grid, values are the conservative estimate or `null`
when the run has too few samples to certify that percentile:

```json
"tpot": {
  "percentiles": { "99.0": 71908203.4, "99.9": 72592022.9, ... },
  "early_stopping_percentiles": { "50.0": 68903121.0, ..., "99.0": 71943332.4, "99.9": null }
}
```

A `null` value means the run had fewer samples than that percentile's floor at confidence 0.99.
An enabled target series that recorded nothing still emits the map (all `null`) rather than
silently looking feature-off. The rich per-percentile detail (empirical value, n, `min_queries`,
discard count) is INFO-logged by the aggregator at run end and reproducible offline via
`scripts/early_stopping_estimate_from_events.py`. The text report renders the map per metric
(one line per percentile; `N/A` = insufficient samples).

## Explicitly out of scope

- No `target_latency` pass/fail gate (LoadGen's Server ES path).
- No dynamic mid-run halt — evaluated once at COMPLETE.
- No scenario gating: offline/max_throughput runs with the flag enabled still compute the estimates
  (for `latency`, plus `ttft`/`tpot` when streaming); they are simply not gating metrics there, since
  throughput-bound scenarios have no tail-latency constraint to certify.
