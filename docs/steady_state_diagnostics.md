# `steady_state_diagnostics`

Use this CLI to inspect steady-state and drift for a benchmark run's
`events.jsonl`. The implementation is
`src/inference_endpoint/metrics/steady_state_diagnostics.py`. The full methodology
lives in [`steady-state-detection.md`](steady-state-detection.md); this page is the
operator's quick reference.

**The rules live elsewhere.** MLCommons
[§4.4 Reporting Basis (Steady-State Window)][rules] defines the official result:
when a steady-state window supersedes the whole-run `total`, how coverage
`status` is classified, and which load patterns are in scope. This document
describes the tool and its defaults. Where the two disagree, the rules win.

[rules]: https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md#44-reporting-basis-steady-state-window

## Two ways to get a verdict

**During a run** (opt-in): `--steady-state` /
`settings.steady_state.enabled: true`. The metrics aggregator collects the same
super-pass series while the run is active. It reuses the token counts from the
TPOT trigger and computes the verdict after token drain finishes.

The verdict lands on `Report.steady_state`, then reaches `result_summary.json`,
`report.txt`, and the console summary.

Live detection is off by default. It can judge these load patterns:

| Load pattern      | Steady state  | Why                                                                                 |
| ----------------- | ------------- | ----------------------------------------------------------------------------------- |
| `concurrency`     | supported     |                                                                                     |
| `poisson`         | supported     |                                                                                     |
| `offline`         | not supported | every query is issued at t=0, so the min-duration gate has no issue span to measure |
| `agentic`         | not supported | its steady-state metric is per-trajectory NATL, not what this collector produces    |
| anything unmapped | not supported | fails closed rather than inheriting a profile nobody validated                      |

A live verdict also requires:

- a performance phase, not accuracy-only mode;
- streaming, because TPOT is the gated metric and only exists when streaming;
- a resolvable tokenizer.

If any requirement is missing, collection stays off.

Detection is best-effort. It runs inside the aggregator's finalize path. If it
fails, the performance report is still written.

**After a run** (this CLI): replays `events.jsonl` through the same collector and
the same `compute_steady_state_metrics`. The CLI is the debug and audit path. It
is also the only path that prints the drift tables.

## What it does

1. Buckets performance-tracked samples into **super-passes** by issue order
   (`--superpass-size` samples each, default `--dataset-size`).
2. Reconstructs per-sample **TTFT** (`recv_first − issued`) and **TPOT**
   (`(complete − recv_first) / tokens(output-after-first-chunk)`). TPOT needs the
   `--tokenizer`, so the tokenizer is required.
3. Finds the **first steady plateau**. The window grows from the start while the
   gated metric, **TPOT** p50 and p90, stays trend-steady
   (Mann–Kendall + Hamed–Rao) and within a CoV bound.
4. Treats **TTFT** as a diagnostic and drift warning, not as a hard gate. At high
   concurrency, TTFT tail variance comes from prefill time, tracking dataset ISL
   skew, and queueing. It is not decode unsteadiness. See §5.5.
5. Splits staircase jumps into separate plateaus. The reported steady state is the
   first plateau that clears the min-duration gate. Earlier plateaus that are too
   brief to certify are skipped. Later plateaus are usually degradation.
6. Selects by estimator precision, following MSER. It does not select by
   throughput.
7. Summarizes the selected window with TTFT/TPOT histograms and percentiles, plus
   **per-user and system TPS** with batch-means confidence intervals.
8. Flags a level shift toward the end of the run as an `anomaly` when a
   multi-plateau run has a Pettitt change-point.

## Requirements

- An installed `inference-endpoint` environment.
- Network access to fetch the model's tokenizer (or a local tokenizer dir).

## Run

**Zero-config**: point it at a run directory. It auto-detects everything from the
sidecar `config.yaml`: model to tokenizer, and load pattern to profile.

```bash
uv run python -m inference_endpoint.metrics.steady_state_diagnostics <run_dir>/
```

> **Runs without the `session.phase_start` event** do not announce the super-pass
> size, so auto-detection cannot find one. Pass `--superpass-size <N>` as one full
> dataset pass. Everything else still resolves from `config.yaml`.

**One flag**: for a bare `events.jsonl` with no sidecar, `--model` drives the
built-in model-to-tokenizer registry and the workload profile:

```bash
uv run python -m inference_endpoint.metrics.steady_state_diagnostics events.jsonl \
  --model kimi-k3
```

The run config and **workload profile** auto-resolve the tokenizer, super-pass
size, CoV bounds, and metric. Profiles are `concurrency`, `poisson`, `offline`,
and `agentic`.

These optional overrides are still available:

```
--model <name>            # else auto-detected from config
--tokenizer <hf-id-or-dir># else from the model registry / config
--dataset-size <N>        # else from the performance phase_start event
--profile {concurrency,poisson,offline,agentic}   # else from load_pattern
--superpass-size N  --warmup auto  --warmup-band 0.05
--warmup-driver tpot_p50  --cov-bounds 0.03,0.05,0.08  --trend-gate mk_hamed_rao
--tokenize-batch-size N  --trust-remote-code  --json out.json
--no-min-duration         # downgrade the min-duration gate from reject to warning
```

**Model registry** (extend as needed):

- `kimi-k3`, `kimi-k2` → Moonshot (trust-remote-code)
- `gpt-oss` → `openai/gpt-oss-120b`
- `deepseek-r1`/`dsr1` → `deepseek-ai/DeepSeek-R1`

**Profiles:**

- `concurrency` and `poisson` use the standard issue-time window and **TPOT
  gate**. TTFT is diagnostic, not a gate.
- `offline` uses the same gate, window, and throughput span. It is not supported
  because every query is issued at t=0, so the span collapses.
- `agentic` computes **NATL**, a per-trajectory throughput metric, over
  trajectory super-passes. It prints a prominent **NOT-YET-SUPPORTED** banner.
  Agentic steady-state is experimental and must not be used for submissions.

## Interpreting the output

### Headline — `STEADY STATE`

```
=== STEADY STATE (headline) ===
  window: super-passes 0..3 (post-warmup), 23519 samples
  TPS per-user:    302.3 tok/s/user  CI [302.1, 302.5]
  TPS system:    40960.9 tok/s        CI [39399.3, 40606.8]
  TTFT p50 86.26ms  p90 156.10ms  p99 248.51ms  mean 97.41ms
  TPOT p50 3.29ms   p90 3.44ms    p99 3.56ms    mean 3.31ms
```

- **window** — the steady plateau, as **post-warmup** super-pass indices `lo..hi`, plus
  the pooled sample count it was measured over.
- **TPS per-user** = `1 / mean(TPOT)` — output tokens/s for a single stream
  (interactivity). **TPS system** = total output tokens ÷ window wall-clock (aggregate
  throughput). Each `CI` is a 95% batch-means interval (super-passes as batches), so it
  reflects per-super-pass variability, not a naïve iid interval.
- **TTFT / TPOT** — percentiles and mean over the pooled raw samples of the window.

If no window qualifies:

```
  not found: no admissible steady plateau
```

means no contiguous run of super-passes was steady enough. The run is drifting or
too short. The per-window diagnostics below show which metric failed CoV or trend.

A second "not found" form comes from the **min-duration gate**. See
docs/steady-state-detection.md §5.5. In this case, each admissible plateau is real
but too brief in wall time to certify.

```
  not found: all 6 admissible plateau(s) too short: longest 15s < 600s required (floor-dominated); pass --no-min-duration to override
```

A window can clear the ≥4-super-pass floor and still last only seconds at high
throughput. For example, a `c16k`-scale run's 4 super-passes are about the
concurrency. That is too short to reveal a minutes-scale hiccup.

The required duration is:

```
max(precision, relaxation, floor)
```

The terms are:

- `floor`: 600s. This binds for clean, fast runs.
- `relaxation`: 5·p90-latency. This binds for long-tail workloads.
- `precision`: k·τ. This binds for noisy metrics.

By default, the min-duration gate is **part of window selection**. The first
plateau that clears `min_duration` is reported. Earlier too-short plateaus are
skipped, and the headline shows
`note: skipped N earlier plateau(s) below min-duration`.

The run reports `not found` only when no plateau qualifies. `--no-min-duration`
turns the gate into a warning and reports the first plateau with
`WARNING: Window too short`.

The full breakdown is in `--json`: `steady_state.short_window`,
`window.plateau_index`, and `skipped_short`.

### `ANOMALY` line

```
  ANOMALY: level shift at super-pass 6, TPOT +100.0% toward end of run (likely degradation)
```

A second, materially different plateau was detected after the first and confirmed
by a Pettitt change-point. The headline steady result remains the **first**
plateau. This line says the run degraded later, for example from KV-cache
eviction or an unhealthy worker. `delta_pct` is signed (+ = TPOT rose = worse).

### `WARNING` line

```
  WARNING: ttft_p90 drifting UP over the rest of the run -- the window is a local
  plateau; global steady state is questionable
```

The reported window is locally steady, but a **watched metric (TPOT or TTFT)**
keeps climbing over the super-passes **after** it. The trend test covers the
whole tail, not just the selected window.

TTFT is a soft warning, not a hard gate. A slow TTFT climb can show queue
saturation building across a long high-concurrency run. The warning surfaces
that drift without fragmenting the selected window.

Treat the steady number as a best-effort local plateau, not a clean whole-run
steady state. `drifting_up` in `--json` lists the affected metrics. It is
distinct from `anomaly`, which is a discrete step.

### Diagnostics (below the headline)

Per `--window-size`, the CLI prints a **CoV steadiness** table and a
**whole-run trend** summary. The CoV table covers each gated and diagnostic
metric against each CoV bound over the trailing window. The trend summary covers
each metric across the algorithms.

The full rolling drift scan, including every window position, is only in
`--json`.

### `--json`

Full structured result: `steady_state` (window, `ttft`/`tpot` summaries + histograms,
`tps`, `anomaly` with every plateau), plus `trajectories`, `cov`, and `drift` (the
rolling scan) for deeper analysis.

## Caveats

- **TPOT parity.** Token counts use plain tokenization of the output. The live
  aggregator uses the chat-template path for reasoning and tool-call outputs, so
  absolute TPOT ms can differ for reasoning models. CoV and the trend tests are
  scale-invariant, so the steady/drift **verdicts** are unaffected. Only the
  absolute TPOT magnitude can differ.
- **Window sizes < 4** are useless for drift because the trend test needs ≥ 4
  points. They still contribute to the CoV table.
- Window indices are **post-warmup relative**. Add the resolved warmup, shown as
  `warmup N [auto|fixed]` in the header, to get absolute super-pass numbers.
- **Auto warmup** (default) crops leading super-passes that are still ramping
  toward the steady TPOT level. The band is around the back-half median. On
  long-ramp runs, such as high-concurrency and long-output runs, this can be
  large. A DeepSeek-R1 c28k run can crop about 24 super-passes. This moves the
  window off the ramp shoulder and onto the true plateau. Use a fixed
  `--warmup N` to override.
