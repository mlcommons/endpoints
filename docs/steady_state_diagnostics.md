# `steady_state_diagnostics`

Post-hoc **steady-state / drift diagnostics** for a benchmark run's `events.jsonl`,
at `src/inference_endpoint/metrics/steady_state_diagnostics.py`. The full methodology
lives in [`steady-state-detection.md`](steady-state-detection.md); this is the
operator's quick reference.

**The rules live elsewhere.** What counts as an official result — when the
steady-state window supersedes the whole-run `total`, the coverage `status`
taxonomy, and which load patterns are in scope — is set by
[§4.4 Reporting Basis (Steady-State Window)][rules] in the MLCommons
`endpoints_policies` repo. This document describes the tool and its defaults.
Where the two disagree, the rules win.

[rules]: https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md#44-reporting-basis-steady-state-window

## Two ways to get a verdict

**During a run** (opt-in): `--steady-state` / `settings.steady_state.enabled: true`.
The metrics aggregator accumulates the same super-pass series as the run happens --
reusing the token counts its TPOT trigger already produced -- and computes the
verdict once the token drain finishes. It lands on `Report.steady_state`, so it
reaches `result_summary.json`, `report.txt`, and the console summary.

Off by default. What it can judge:

| Load pattern      | Steady state  | Why                                                                                 |
| ----------------- | ------------- | ----------------------------------------------------------------------------------- |
| `concurrency`     | supported     |                                                                                     |
| `poisson`         | supported     |                                                                                     |
| `offline`         | not supported | every query is issued at t=0, so the min-duration gate has no issue span to measure |
| `agentic`         | not supported | its steady-state metric is per-trajectory NATL, not what this collector produces    |
| anything unmapped | not supported | fails closed rather than inheriting a profile nobody validated                      |

A run also needs a performance phase (not accuracy-only), streaming enabled
(TPOT is the gated metric, and it only exists when streaming), and a resolvable
tokenizer. Any of these missing and collection is silently off.

Detection is best-effort and cannot cost a run its result: it runs inside the
aggregator's own finalize, and a failure anywhere in it leaves the performance
report exactly as it would have been.

**After a run** (this CLI): replays `events.jsonl` through the same collector and
the same `compute_steady_state_metrics`, so both reach the same verdict. This is
the debug/audit path, and it is the only one that prints the drift tables.

## What it does

1. Buckets performance-tracked samples into **super-passes** by issue order
   (`--superpass-size` samples each, default `--dataset-size`).
2. Reconstructs per-sample **TTFT** (`recv_first − issued`) and **TPOT**
   (`(complete − recv_first) / tokens(output-after-first-chunk)`); TPOT needs the
   `--tokenizer`, so it is required.
3. Finds the **first steady plateau**: grow a window from the start while it stays
   _admissible_ — the gated metric (**TPOT** p50 & p90) is trend-steady
   (Mann–Kendall + Hamed–Rao) **and** within a CoV bound. **TTFT is not a hard gate** —
   at high concurrency its tail variance (prefill time tracking dataset ISL skew + queue)
   is structural, not decode un-steadiness, so it is a diagnostic and a drift _warning_
   only (see §5.5). A staircase jump breaks the window, segmenting the run into plateaus.
   The **first plateau that clears the min-duration gate is the reported steady state**
   (usually the literal first; earlier plateaus too brief to certify are skipped). Later
   plateaus are usually degradation. Selection follows MSER: pick by estimator precision,
   never by throughput.
4. Summarizes that window (TTFT/TPOT histograms + percentiles, **per-user & system
   TPS** with batch-means confidence intervals) and **flags a level shift** toward the
   end of the run (multi-plateau + Pettitt change-point) as an `anomaly`, rather than
   hiding it.

## Requirements

- An installed `inference-endpoint` environment.
- Network access to fetch the model's tokenizer (or a local tokenizer dir).

## Run

**Zero-config** — point it at a run directory and it auto-detects everything from the
sidecar `config.yaml` (model → tokenizer, load pattern → profile):

```bash
uv run python -m inference_endpoint.metrics.steady_state_diagnostics <run_dir>/
```

**One flag** — a bare `events.jsonl` with no sidecar; `--model` drives the built-in
model→tokenizer registry and the workload profile:

```bash
uv run python -m inference_endpoint.metrics.steady_state_diagnostics events.jsonl \
  --model kimi-k3
```

Everything is auto-resolved from the run's config + a **workload profile**
(`concurrency` / `poisson` / `offline` / `agentic`), which sets the tokenizer, super-pass
size, CoV bounds, and metric. All of the following remain available as **optional
overrides**:

```
--model <name>            # else auto-detected from config
--tokenizer <hf-id-or-dir># else from the model registry / config
--dataset-size <N>        # else from the performance phase_start event
--profile {concurrency,poisson,offline,agentic}   # else from load_pattern
--superpass-size N  --window-sizes 4,6,8  --warmup auto  --warmup-band 0.05
--warmup-driver tpot_p50  --cov-bounds 0.03,0.05,0.08  --trend-gate mk_hamed_rao
--tokenize-batch-size N  --trust-remote-code  --json out.json
--no-min-duration         # downgrade the min-duration gate from reject to warning
```

**Model registry** (extend as needed): `kimi-k3`, `kimi-k2` → Moonshot (trust-remote-code);
`gpt-oss` → `openai/gpt-oss-120b`; `deepseek-r1`/`dsr1` → `deepseek-ai/DeepSeek-R1`.

**Profiles:** `concurrency` and `poisson` share the standard issue-time window + **TPOT
gate** (TTFT is diagnostic, not a gate); `offline` uses the same gate but its
window/throughput are completion-based (partial support); `agentic` computes **NATL**
(per-trajectory throughput) over trajectory
super-passes and prints a prominent **NOT-YET-SUPPORTED** banner — agentic steady-state is
experimental and must not be used for submissions.

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

means no contiguous run of super-passes was steady enough — the run drifts or is too
short. The per-window diagnostics below show why (which metric failed CoV/trend).

A second "not found" form is the **min-duration gate** (docs/steady-state-detection.md
§5.5): every admissible plateau is genuine but too brief in wall-time to certify.

```
  not found: all 6 admissible plateau(s) too short: longest 15s < 600s required (floor-dominated); pass --no-min-duration to override
```

A window that clears the ≥4-super-pass floor can still be only seconds long at high
throughput (a `c16k`-scale run's 4 super-passes ≈ the concurrency), which cannot reveal
a minutes-scale hiccup. The required duration is `max(precision, relaxation, floor)`:
`floor` (600s) binds for clean fast runs, `relaxation` (5·p90-latency) for long-tail
workloads, `precision` (k·τ) for noisy metrics.

By default the gate is **part of window selection**: the first plateau that clears
`min_duration` is reported, **skipping** earlier too-short plateaus (the headline then
shows a `note: skipped N earlier plateau(s) below min-duration` line). Only when no
plateau qualifies does the run report `not found`. `--no-min-duration` disables this and
reports the first plateau with a `WARNING: Window too short` line instead. Full breakdown
is in `--json` (`steady_state.short_window`, plus `window.plateau_index` /
`skipped_short`).

### `ANOMALY` line

```
  ANOMALY: level shift at super-pass 6, TPOT +100.0% toward end of run (likely degradation)
```

A second, materially different plateau was detected after the first and confirmed by a
Pettitt change-point. The headline steady result is still the **first** plateau; this
line says the run degraded later (e.g. KV-cache eviction, a sick worker). `delta_pct`
is signed (+ = TPOT rose = worse).

### `WARNING` line

```
  WARNING: ttft_p90 drifting UP over the rest of the run -- the window is a local
  plateau; global steady state is questionable
```

The reported window is locally steady, but a **watched metric (TPOT or TTFT)** keeps
climbing over the super-passes **after** it (the trend test over the whole tail, not just
the window). This is TTFT's primary role now that it no longer hard-gates: a slow TTFT
creep — e.g. queue saturation building several-fold across a long high-concurrency run —
is surfaced here as a soft warning instead of fragmenting the window. Treat the steady
number as a best-effort local plateau, not a clean whole-run steady state. (`drifting_up`
in `--json` lists the affected metrics; distinct from `anomaly`, a discrete step.)

### Diagnostics (below the headline)

Per `--window-size`: a **CoV steadiness** table (per gated/diagnostic metric, PASS/`fail`/`n/a`
against each CoV bound over the trailing window) and a **whole-run trend** summary per
metric across the algorithms. The full rolling drift scan (every window position) is
only in `--json`.

### `--json`

Full structured result: `steady_state` (window, `ttft`/`tpot` summaries + histograms,
`tps`, `anomaly` with every plateau), plus `trajectories`, `cov`, and `drift` (the
rolling scan) for deeper analysis.

## Caveats

- **TPOT parity.** Token counts use plain tokenization of the output; the live
  aggregator uses the chat-template path for reasoning/tool-call outputs, so absolute
  TPOT ms can differ for reasoning models. CoV and the trend tests are scale-invariant,
  so the steady/drift **verdicts** are unaffected — only the absolute TPOT magnitude.
- **Window sizes < 4** are useless for drift (the trend test needs ≥ 4 points); they
  still contribute to the CoV table.
- Window indices are **post-warmup relative** (add the resolved warmup — shown as
  `warmup N [auto|fixed]` in the header — for absolute super-pass numbers).
- **Auto warmup** (default) crops leading super-passes still ramping toward the steady
  TPOT level (band around the back-half median). On long-ramp runs (high-concurrency,
  long-output) this can be large — e.g. ~24 super-passes on a DeepSeek-R1 c28k run —
  which is correct: it moves the window off the ramp shoulder onto the true plateau. Use
  a fixed `--warmup N` to override.
