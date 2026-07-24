# Steady-State Windowing — Findings (Working Draft)

**Status:** Living document — updated as analysis proceeds. Not yet team-final.
**Date started:** 2026-07-24
**Scope:** Concurrency-mode LLM inference benchmarks. Offline analysis from `events.jsonl`.
**Related:** design `docs/design/2026-07-21-steady-state-windowing-design.md`, plan
`docs/design/2026-07-21-steady-state-windowing-plan.md`.

## 1. Problem

Concurrency-mode runs report artificially deflated throughput and inflated tail
latency. Two non-steady regions contaminate the measured window:

- **Ramp-up:** the client must issue enough samples to fill the target in-flight
  concurrency `N`. Until the pipe is full, a fill-burst of `N` requests queues
  against a finite-rate server and their TTFT inflates dramatically; the effect
  grows with `N`.
- **Ramp-down (drain):** after issuance stops, in-flight decays below `N` and
  throughput deflates.

## 2. Approach (implemented)

- **Issue-time window, unified sample set.** Measured set = requests _issued_
  during the steady region. Throughput denominator = issue-time span
  (`first_issue → last_issue`); latency = full lifetime of that same set. Drain
  is excluded from the denominator for free (it lives after `last_issue`), yet
  drain completions still count for latency. No end-crop needed; only the
  ramp-up start is cropped.
- **Super-pass unit:** `S = ceil(N / dataset_size)` passes = enough issuance to
  hold `N` distinct samples in flight. `super_pass_samples = dataset_size * S`.
  Everything (warmup crop, convergence windows) measures in super-passes.
- **Warmup crop:** drop the first super-pass (the fill region).
- **Stopping rules (pure functions over the per-super-pass series):**
  - A — fixed budget (`k` super-passes).
  - B — adaptive CoV: stop when the coefficient of variation (`stdev/mean`) of
    per-super-pass scalar percentiles (p50/p99 of TTFT + latency) across a
    trailing window falls below a bound. CoV, not KL divergence, because the HDR
    histogram bucket edges shift per snapshot.
- **Best-effort branch (`status`):** `windowable` / `insufficient_passes` /
  `partial_dataset` — never hard-fails; short runs get a flagged best-effort
  window so a batch sweep stays complete.

Tooling (all cold-path, mirror `scripts/early_stopping_estimate_from_events.py`):

- `src/inference_endpoint/metrics/steady_state/` — `series`, `window`, `stopping`, `harness`.
- `scripts/steady_state_from_events.py` — total vs steady + A/B table + JSON.
- `scripts/steady_state_cov_sweep.py` — `cov_window × cov_bound` ablation.

## 3. Corpus

GB300-NVL72 gpt-oss-120b, TRT-LLM disagg, dataset `perf_eval_ref.parquet`
(**dataset_size = 6396**), concurrency mode. Runs analyzed:

| run    | concurrency | issued | full super-passes | status                                   |
| ------ | ----------- | ------ | ----------------- | ---------------------------------------- |
| C8     | 8           | 2226   | 0                 | `partial_dataset` (time-capped < 1 pass) |
| C140   | 140         | 29915  | 4                 | windowable                               |
| C1024  | 1024        | 141169 | 22                | windowable                               |
| C2048  | 2048        | 216535 | 33                | windowable                               |
| C7168  | 7168        | 445756 | 34                | windowable (but drifting — see §6)       |
| C22528 | 22528       | 647616 | 25                | windowable                               |

DSR1 GB300 (`dataset_size 4388`) available but not yet analyzed here. A tentative
DSR1 c28080 run (`_numa`, sha `e072ac2`) is being pulled.

## 4. Finding — ramp deflation is real and scales with concurrency

p99 TTFT, total (warmup-included) vs steady (asymptote):

| C     | total p99 TTFT | steady p99 TTFT | recovery | QPS Δ |
| ----- | -------------- | --------------- | -------- | ----- |
| 140   | 0.294 s        | 0.247 s         | −16%     | ~0    |
| 1024  | 1.29 s         | 1.11 s          | −14%     | ~0    |
| 2048  | 1.18 s         | 0.96 s          | −19%     | ~0    |
| 22528 | 15.72 s        | 2.47 s          | **−84%** | −1%   |

**QPS (offered-rate throughput) is barely affected by the ramp (~0–1%).** The
ramp deflates **tail latency**, not throughput — a request issued during a fill
burst of `N` waits behind the burst for its first token, but the completion rate
is unchanged. At the extreme (C=22528, a 22528-deep fill), cropping one warmup
super-pass recovers **6.4×** on p99 TTFT.

Note: absolute latencies are **not** comparable across concurrency points here —
each point uses a different disagg deployment (ng/nc/dep counts differ), so this
is not a controlled sweep. The goal is per-run steady-state correctness, not
cross-run comparison.

## 5. Finding — no universal `(cov_window, cov_bound)`

Best config per run (scored vs the full-series asymptote; target p99-TTFT err ≤ 5%):

| C     | best (window, bound) | super-passes | p99-TTFT err                |
| ----- | -------------------- | ------------ | --------------------------- |
| 140   | (3, 0.05)            | 3            | 0.0%                        |
| 1024  | (3, 0.03)            | 3            | 1.5%                        |
| 2048  | (3, 0.08)            | 3            | 0.3%                        |
| 22528 | (6, 0.15)            | 8            | 1.6%                        |
| 7168  | (5, 0.05)            | 11           | **35.7% — none met target** |

- Low/mid concurrency settles fast; `cov_window=3` with almost any bound is fine.
- High concurrency (C=22528) needs a **longer window + looser bound** (`w=6, b=0.15`):
  each super-pass's p99 is estimated from ~256 tail samples, so it carries a
  few-percent sampling jitter — a CoV bound below that floor (≤0.02) never
  converges, and a short window triggers on spurious early dips.
- **A fixed config cannot serve all concurrencies.** This motivates an adaptive
  approach (§7).

## 6. Finding (key) — p99 TTFT has no steady state; it drifts

Per-super-pass p99 TTFT trajectory (after the dropped fill super-pass):

- **C=7168:** sp1 2.9 s → sp7 5.7 → sp15 7.3 → sp21 8.3 → sp33 **10.9 s** — a
  **3.8× monotonic climb** over ~40 min. Meanwhile **p50 TTFT falls** 1.73 → 0.80 s.
- **C=22528:** sp3 0.65 s (settles fast) → sp10 1.22 → sp17 1.55 → sp25 **3.21 s**
  — ~5× climb off a lower floor. p50 flat ~0.37 s.

**p99 TTFT drifts monotonically upward through the entire run — there is no
plateau.** The median settles (even improves) while the tail worsens. So
steady-state is **metric-dependent**:

- **Has a steady state:** QPS, p50 TTFT, e2e latency (p50 latency flat).
- **Does NOT:** p99 TTFT — still climbing at end-of-run.

This is why no CoV config works for C=7168: there is no flat region to detect, so
the "asymptote" is just a point on a rising ramp, not a steady value.

**Not throttling.** The decode-server logs show zero preempt / evict / OOM, and a
global slowdown would raise p50 too. Median-down / tail-up over a sustained run
points to **progressive tail degradation** — a growing scheduling imbalance or KV
fragmentation starving the worst-case requests while the bulk speeds up. Root
cause (prefill-queue-depth trend) is not yet pinned; the _measurement_ conclusion
holds regardless.

## 7. Implication — detect steady-vs-drift, don't assume steady

The real question is not "what is the steady value" but **"does this metric even
have a steady state, or a persistent trend?"** Reporting a single "steady" p99
TTFT for a drifting run is a false number.

Proposed direction (prototype in progress): an **ensemble of CoV detectors** at
varied `(window, bound)` plus a **plateau-vs-trend comparator**:

- Fit a slope of the per-super-pass metric across the post-warmup series; test its
  significance relative to the per-super-pass noise floor.
- Metric classification: **steady** (detectors concur on a region AND slope
  insignificant) vs **drifting** (detectors scatter OR slope significant).
- Extend the `status` field with `drifting`; for a drifting metric, surface the
  trend (and a caveated last-window value) instead of a bogus steady point.

## 8. Data-validity note — client concurrency cap

The single-process endpoints-client hits the Linux ephemeral-port limit, so
**effective concurrency caps at ~28k** (verified via `ss`). Configured c32768 /
c65536 runs could not reach their targets — their behavior clustered just above
c16k. Those logs are debugging artifacts and **must not be used**; the max
concurrency config was capped to **28080** (the achieved ceiling). For the c32768
events, super-pass math must use the achieved concurrency (28080), not 32768.

## 9. Open questions

- Does the p99 TTFT drift reproduce on DSR1 (different workload) and on a clean
  capped-28080 re-run? (C=7168 excluded from the prototype for now.)
- Root cause of progressive tail degradation (prefill queue vs KV fragmentation).
- Ensemble parameters: detector grid, slope-significance threshold, how to report
  a drifting metric in the run report.
- MLPerf policy: how to define/attest a steady window when the tail drifts.

## Appendix — reproduction

Tests + tooling run in the Linux dev container (`inference-endpoint-dev`,
`--shm-size=8G`); pre-commit on host. Example:

```
docker run --rm --shm-size=8G -v "$PWD":/mnt/inference-endpoint -w /mnt/inference-endpoint \
  -v <artifacts>:/data inference-endpoint-dev bash -lc \
  "uv run python scripts/steady_state_cov_sweep.py /data/C22528/events.jsonl \
     --dataset-size 6396 --concurrency 22528 --windows 3,4,5,6"
```
