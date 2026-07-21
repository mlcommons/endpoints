# Steady-State Windowing for Concurrency-Mode Benchmarks

**Status:** Design
**Date:** 2026-07-21
**Scope:** `concurrency` load pattern only. Offline (post-process) computation from `events.jsonl`.

## Problem

Concurrency-mode runs report artificially deflated throughput and inflated tail
latency because the measured duration includes two non-steady regions:

- **Ramp-up** — at `t=0` the client must issue enough samples to fill the
  target in-flight concurrency `N`. Until the pipe is full, offered load is below
  steady state. During this fill-burst against a finite-rate server, requests
  queue and their time-to-first-token (TTFT) inflates dramatically. The effect
  grows with `N`.
- **Ramp-down (drain)** — after issuance stops, the remaining in-flight requests
  drain while in-flight concurrency decays below `N`. With a non-homogeneous
  dataset (wildly varying input/output sequence lengths), the highest-OSL
  requests trail for a long time. Throughput deflates because the wall-clock
  denominator keeps ticking while concurrency is below `N`.

### Observed magnitude (validation)

From `endpoints-launch` GB300 DeepSeek-R1, `point_28080`
(`target_concurrency = 28080`, dataset `deepseek_r1_eval` = 4388 samples,
658200 samples issued = 150 dataset passes):

| statistic | TTFT    |
| --------- | ------- |
| median    | 0.36 s  |
| p90       | 15.4 s  |
| p99       | 82.7 s  |
| p99.9     | 99.1 s  |
| max       | 101.8 s |

p99 TTFT is 82 s against a 0.36 s median (~230×). The tail is dominated by the
ramp-up fill-burst queue, not steady-state behaviour. This is the deflation the
design targets. (Time-series validation of the _fix_ is not possible for this
run — its `events.jsonl` was discarded and the `v0.1.0` summary schema carries
only a single `duration_ns` with no load-generator-window split — but the tail
shape is fully consistent with the ramp-up hypothesis.)

## Constraints

1. **Unified window.** Latency metrics and throughput metrics MUST be computed
   over the _same_ set of samples. They cannot diverge on membership.
2. **Dataset-boundary measurement.** The dataset is non-homogeneous; a partial
   pass samples a biased subset of the ISL/OSL mix. The measured region MUST be
   an integer number of full dataset passes. No partial passes.
3. **Concurrency mode only.** In-flight is held at `N` by construction, so the
   ramp boundaries are knowable. Poisson / fixed-QPS is out of scope (in-flight
   floats per Little's law; "fill `N`" is undefined).
4. **MLPerf.** Compliance is desired but malleable — this issuance mechanism is
   new to MLPerf and policy can be proposed to accommodate it. Prefer designs
   that keep run length deterministic and reproducible.

## Key insight — issue-time window

Define the measurement window on **issue time**, not completion time, with a
single membership set for both metric families:

- **Measured set** = every request _issued_ during the steady region.
- **Throughput denominator** = the issue-time span of that region
  (`first_issue → last_issue`). In concurrency mode, in-flight == `N` across this
  entire span by construction, so issue-rate == completion-rate (steady state).
  The drain lives _after_ `last_issue`, so it never enters the denominator.
- **Latency / tokens** = the full lifetime of those same requests, including the
  high-OSL ones that complete during drain.

Consequences:

- Same samples for latency and throughput → satisfies the unified-window
  constraint.
- The drain is excluded from the throughput denominator **for free**, while its
  completions still count in the numerator and in the latency percentiles. **No
  end-crop is needed.** Only the ramp-up start needs cropping.
- The "p99 TTFT explodes" symptom is not a windowing problem once ramp-up is
  cropped — it is a _sample-count_ problem, answered by the existing MLPerf
  early-stopping validity gate (`metrics/early_stopping.py`).

## The super-pass unit

When `dataset_size < N`, a single dataset pass does not contain `N` distinct
in-flight samples, so per-pass windows are not representative. Define:

```
S = ceil(N / dataset_size)   # passes needed to hold N distinct samples in flight
```

`S` is the **super-pass** — the atomic unit for every windowing decision:

- **Warmup crop** = drop the first `S` passes (the fill region). Deterministic;
  no detection required.
- **Minimum measured region** = ≥ 1 super-pass.
- Any convergence window (rule B, below) is measured in super-passes, so each
  window always encompasses `N`.

For `point_28080`: `S = ceil(28080 / 4388) = 7` passes/super-pass; 150 passes ≈
21 super-passes total; warmup crop drops 1 super-pass.

## Architecture

The computation is **cold-path only**. It re-ingests the durable `events.jsonl`
after a run, mirroring the existing
`scripts/early_stopping_estimate_from_events.py` pattern. No hot-path change; no
dependency on live snapshots (which can lag behind under load).

```
events.jsonl ──▶ super-pass bucketer ──▶ per-super-pass series
                                              │
                                              ▼
                                     stopping rule (A or B)
                                              │
                                              ▼
                              steady region (integer super-passes)
                                              │
                                              ▼
                      windowing core ──▶ steady-state metrics
                                              │
                                              ▼
                      report: { total, steady_state }  (steady_state = official)
```

### Component 1 — Super-pass bucketer

Reads `events.jsonl`, reconstructs each request's issue timestamp and dataset
pass index, and buckets requests into super-passes of `S = ceil(N/dataset_size)`
passes. `N` and `dataset_size` are read from the run's resolved `config.yaml`.
Emits an ordered series of per-super-pass rollups (TTFT / TPOT / latency /
throughput / token counts, plus first/last issue timestamps).

### Component 2 — Windowing core (always on)

Given a chosen steady region (a contiguous integer number of super-passes):

- Measured set = requests issued within the region's issue-time span.
- Throughput denominator = `first_issue → last_issue` of the region.
- Latency/token percentiles = full lifetime of the measured set.

Always drops the first `S` passes (one super-pass) as warmup.

### Component 3 — Stopping rules (pure functions over the series)

Both consume the per-super-pass series and return the steady region. Both are
gated by the early-stopping validity check on the accumulated measured set.

- **Rule A — fixed budget.** Stop after `K` super-passes (config). Deterministic
  length, MLPerf-reproducible, simplest. Cost: may over-run.
- **Rule B — adaptive convergence.** Stop when a trailing window of super-passes
  converges. **Convergence metric = coefficient of variation (CoV) of key
  scalar percentiles** (p50/p99 TTFT, p50/p99 TPOT) across the trailing window,
  below a configured bound (Kalibera–Jones style). Cost: data-dependent run
  length.

  **Why CoV and not KL divergence:** the HDR histogram bucket edges are dynamic
  per snapshot (log-spaced over each snapshot's observed `[min, max]`), so KL
  divergence across snapshots compares mismatched buckets and is invalid (see
  AGENTS.md, "Histogram bucket edges are dynamic per snapshot"). Scalar
  percentiles are bucket-independent and robust.

### Component 4 — A/B experiment harness

Because both rules are pure functions over one recorded series, A and B are
compared **offline on the same data** — identical arrivals, only the stop
decision differs (a literal A/B test, no confound from separate live runs).

- **Ground truth** = the full-series asymptote (the steady-state number computed
  over all available super-passes).
- **Score** each rule by `|estimate − asymptote|` (per metric) and by
  super-passes consumed. "Better" = closer to the asymptote with fewer
  super-passes.

### Component 5 — Report

The report grows a `steady_state` block alongside the existing `total` block.
**`steady_state` is the official result.** The divergence between `total` and
`steady_state` is itself surfaced as a signal — a large gap indicates the run
had too much ramp relative to its length (or was too short).

## Production mechanism

Offline replay is the production path, not just the experiment path. Live
snapshots can lag under load; `events.jsonl` is the durable source of truth. The
same post-process script that runs the A/B experiment computes the official
steady-state result.

## Testing

- **Unit — super-pass math.** `S = ceil(N/dataset_size)` across
  `dataset_size > N`, `dataset_size == N`, `dataset_size < N` (the
  `point_28080`-style case). Boundary bucketing of requests into passes and
  super-passes.
- **Unit — windowing core.** Synthetic issue/complete timestamps: verify
  throughput denominator excludes drain, latency set includes drain completions,
  membership identical between the two metric families.
- **Unit — stopping rules.** Rule A returns the right region for a given `K`.
  Rule B: a synthetic series that stabilises at a known super-pass converges
  there; a non-converging series runs to the end (and is reported as such — no
  silent truncation).
- **Unit — CoV convergence** on hand-built percentile series (stable →
  converges, drifting → does not).
- **Integration — replay.** A recorded `events.jsonl` fixture through the full
  pipeline produces a `steady_state` block; assert it excludes the warmup
  super-pass and that `total` vs `steady_state` divergence is reported.
- Reuse existing fixtures where possible; follow the
  `early_stopping_estimate_from_events.py` test pattern.

## Out of scope

- Poisson / fixed-QPS steady-state detection.
- Live (in-run) early stopping — Rule B halting the run as it executes. The
  series and rules are designed to allow it later, but v1 is offline only.
- Changing hot-path metrics or the live snapshot cadence.

## Open questions

- Default value of `K` (Rule A) and the CoV bound + trailing-window length
  (Rule B) — to be tuned empirically via the A/B harness on real runs.
- Exact MLPerf policy proposal for the issue-time window definition — drafted
  separately once the A/B results quantify the correction.
