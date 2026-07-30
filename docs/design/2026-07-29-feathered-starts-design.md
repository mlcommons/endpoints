# Feathered / Staggered Starts — Staged Ramp for the Load Generator

**Status:** Design + offline simulation (Study 2 of the steady-state investigation)
**Date:** 2026-07-29
**Branch:** `design/steady-state-windowing`
**Scope:** Design + prototype + sim only. **No live runs, no merged strategy change.**
**Related:**
`docs/design/2026-07-21-steady-state-windowing-design.md` (ramp-up/drain problem),
`docs/design/2026-07-29-steady-state-studies-coordination.md` (Study 2 spec),
`scripts/staggered_ramp_sim.py` (the simulation backing every number here).

## Problem

At `t=0` the load generator fills its whole target in-flight population in a
single burst. Against a finite-rate server this shocks the first-token
("prefill") pipeline: all requests land in the admission queue at once, and the
k-th request admitted waits ~`k / prefill_rate` for its first token. This is the
ramp-up TTFT inflation the steady-state windowing design targets — p99 TTFT
82.7 s against a 0.36 s median (~230×) on `point_28080`
(`docs/design/2026-07-21-steady-state-windowing-design.md`, "Observed
magnitude").

This document studies whether a **staged ("feathered") ramp** — issuing the
target in `S` increments instead of all at once — smooths the offered-load curve
and shrinks the non-steady ramp region, in both `concurrency` and
`max_throughput` modes.

## Current behavior (grounded in `strategy.py`)

The two burst strategies live in
`src/inference_endpoint/load_generator/strategy.py`.

### `ConcurrencyStrategy` — the t=0 fill-burst of N

- The gating semaphore is constructed **already full** with `N` permits:
  `self._sem = asyncio.Semaphore(target_concurrency)`
  (`strategy.py:276`).
- `execute()` loops the sample order, `await self._sem.acquire()` then
  `phase_issuer.issue(idx)` (`strategy.py:279-285`). Because the semaphore starts
  with `N` permits, the **first `N` acquires succeed immediately with no
  completion in between** — so `N` samples are issued back-to-back at `t≈0`
  before `on_query_complete()` releases the first permit
  (`strategy.py:287-288`). That is the fill-burst.
- After the fill, issuance is completion-driven: each `on_query_complete()`
  releases one permit, admitting exactly one replacement, holding in-flight at
  `N`. In `session.py`, `PhaseIssuer.issue()` increments `inflight`
  (`session.py:292`) and `mark_inflight_complete()` decrements it
  (`session.py:213-216`).

### `BurstStrategy` (max_throughput) — issue-as-fast-as-possible

- `issue_next()` pulls the next index, issues it, and re-schedules itself via
  `self._loop.call_soon(issue_next)` with **no delay**
  (`strategy.py:231-238`); the initial kick is a single `call_soon`
  (`strategy.py:243`). The whole sample order is therefore issued as fast as the
  event loop drains callbacks — effectively the entire query pool enters the
  server's queue near `t=0`, bounded only by the receiver coroutine yielding.

### Factory

`create_load_strategy()` (`strategy.py:296-345`) maps
`LoadPatternType.MAX_THROUGHPUT → BurstStrategy` (`strategy.py:321-322`) and
`LoadPatternType.CONCURRENCY → ConcurrencyStrategy` (`strategy.py:330-335`).
`TimedIssueStrategy` (Poisson) is **out of scope**: its arrivals are already
paced by inter-arrival delays, so there is no fill-burst.

## Staged-ramp policy

Generalize "fill `N`" (or "issue the whole pool") into "fill in `S` equal
increments, `step = ceil(target / S)` per increment, spaced `step_interval`
apart, then behave exactly as today."

### Step size

`step = ceil(target / S)`, where `target = N` (concurrency) or the query-pool
size (max_throughput). `S = 1` reproduces today's burst exactly, so the change
is strictly opt-in and backward-compatible.

### Step interval — the critically-damped choice

The server admits first-tokens at a finite `prefill_rate`. A step of `step`
requests takes `step / prefill_rate` to fully drain out of the prefill queue.
Choosing

```
step_interval ≈ step / prefill_rate = target / (S · prefill_rate)
```

lets each increment's prefill queue drain **before** the next increment arrives,
so the queue never grows past one step's worth. This is the key property: the
peak first-token wait drops from `target / prefill_rate` (burst) to
`step / prefill_rate = (target/S) / prefill_rate` — an `S×` reduction — while the
**total fill span stays `target / prefill_rate`** (identical to how long the
burst takes to drain). Staged ramp is therefore a near-Pareto improvement: same
fill duration, `S×` smaller spike.

`prefill_rate` is not known a priori. Practical options, in preference order:

1. **Auto-adaptive (recommended):** advance to the next step when the current
   step's in-flight requests have all reached first token (drain-driven, no
   constant needed). Requires a first-token signal to the strategy — see Open
   questions.
2. **Fixed interval from a probe:** measure `prefill_rate` in the existing
   `probe` warmup and set `step_interval` from it.
3. **Fixed wall-clock interval** (config): simplest, least adaptive; a
   too-short interval degrades toward burst, a too-long one lengthens the ramp.

### How "fill N" generalizes

Staged fill only changes issue **timing**, never issue **order** — samples are
still drawn from `sample_order` in the same sequence. Concretely:

- **Concurrency:** start the semaphore with `step` permits instead of `N`, and
  release `step` more every `step_interval` until `N` permits have been granted;
  completion-driven release is unchanged. In-flight climbs `step → 2·step → … →
N` in a staircase, then holds at `N`.
- **Max_throughput:** issue `step` requests, wait `step_interval`, repeat, until
  the pool is exhausted. No completion-driven replacement (unchanged).

## Interaction with the super-pass warmup crop

The windowing design
(`docs/design/2026-07-21-steady-state-windowing-design.md`, "The super-pass
unit") drops the first `S_pass = ceil(N / dataset_size)` passes as warmup, keyed
on **issue order**. Staged ramp changes issue timing, not issue order, so the two
are **orthogonal and compose cleanly**:

- The super-pass bucketer (issue-order based) sees the identical sample sequence;
  the warmup crop drops the same issue-order-leading set either way.
- Staged ramp cannot _replace_ the crop: it does not fill the pipe faster (the
  server rate is the bound), so the ramp's **duration** — hence the number of
  passes the crop must drop — is essentially unchanged (see Simulation).
- What staged ramp _does_ is shrink the **severity** of TTFT inflation inside
  those cropped passes by `~S×`. That de-risks an under-sized crop: if the crop
  is one super-pass short, the residual contamination bleeding into the measured
  window is a small sawtooth (`≤ step/prefill_rate`) rather than one
  catastrophic `N/prefill_rate` spike. It also shrinks the `total` vs
  `steady_state` divergence signal.
- The issue-time throughput denominator (`first_issue → last_issue`) gains up to
  `(S-1)·step_interval` of front-loaded low-load time, but that time lives in the
  warmup super-pass the crop already removes, so the steady-region denominator is
  unaffected as long as the crop covers the ramp.

**Net:** staged ramp is complementary to — not a substitute for — the warmup
crop. Keep the crop; use staging to make the crop robust and the pre-crop tail
benign.

## Prototype (UNMERGED — illustrative, do not wire in)

These sketches are intentionally _not_ applied to the production classes. They
show the shape of the change for a future live-validation PR.

### Staged `ConcurrencyStrategy`

```python
class StagedConcurrencyStrategy:
    """Concurrency strategy that fills N in `steps` increments (UNMERGED sketch)."""

    def __init__(self, target_concurrency, sample_order, loop, *, steps, step_interval):
        self._target = target_concurrency
        self._per_step = math.ceil(target_concurrency / steps)
        self._steps = steps
        self._step_interval = step_interval
        # Start EMPTY, not full: the ramp task grants permits incrementally.
        self._sem = asyncio.Semaphore(0)
        self._sample_order = sample_order
        self._loop = loop

    async def _ramp(self):
        granted = 0
        for _ in range(self._steps):
            n = min(self._per_step, self._target - granted)
            for _ in range(n):
                self._sem.release()
            granted += n
            if granted >= self._target:
                break
            await asyncio.sleep(self._step_interval)

    async def execute(self, phase_issuer):
        ramp = asyncio.create_task(self._ramp())
        for idx in self._sample_order:
            await self._sem.acquire()          # unchanged acquire loop
            if phase_issuer.issue(idx) is None:
                self._sem.release()
                break
        await ramp
        return phase_issuer.issued_count

    def on_query_complete(self, query_id):     # unchanged
        self._sem.release()
```

In-flight is capped by the running total of ramp-granted permits (climbs `0 →
N`), then held at `N` by completion-driven release — exactly staged fill with the
completion path untouched.

### Staged `BurstStrategy`

```python
class StagedBurstStrategy:
    """Max-throughput in `steps` batches instead of one call_soon chain (UNMERGED)."""

    def __init__(self, sample_order, loop, *, batch_size, step_interval):
        self._sample_order = sample_order
        self._loop = loop
        self._batch_size = batch_size
        self._step_interval = step_interval

    async def execute(self, phase_issuer):
        done = asyncio.Event()

        def issue_batch():
            for _ in range(self._batch_size):
                idx = next(self._sample_order, None)
                if idx is None or phase_issuer.issue(idx) is None:
                    done.set()
                    return
            self._loop.call_later(self._step_interval, issue_batch)  # was call_soon

        self._loop.call_soon(issue_batch)
        await done.wait()
        return phase_issuer.issued_count
```

The only change from `BurstStrategy` is `call_later(step_interval, …)` in place
of the immediate `call_soon(issue_next)` (`strategy.py:238`), issuing in batches.

The factory (`strategy.py:296-345`) would select the staged variants when a
(new, default-off) `ramp.steps > 1` setting is present, falling back to today's
`BurstStrategy` / `ConcurrencyStrategy` when `steps == 1`.

## Simulation

`scripts/staggered_ramp_sim.py` models a finite-rate server (FIFO first-token
admission queue at `prefill_rate`, lognormal decode times for OSL heterogeneity,
concurrency replacement after fill). It compares `steps ∈ {1,2,4,8,16}`. It is a
**model, not a measurement** — see Assumptions & limits.

**Calibration.** No Study-1 measured burst-tail magnitude was available at
authoring time (checked the `steady-state-studies` hivemind topic). The default
`prefill_rate = 276 req/s` is back-calibrated to the design doc's `point_28080`
observation (max TTFT ~101.8 s at N=28080 ⇒ 28080/101.8 ≈ 276), with floor TTFT
0.36 s (the observed median). `--prefill-rate` recalibrates to any measured tail.

### Results (N=2048 concurrency; decode_mean 12 s, cv 0.8)

| steps | step_interval | peak TTFT | vs burst | p99 TTFT | ramp_end | non-steady count | vs burst |
| ----- | ------------- | --------- | -------- | -------- | -------- | ---------------- | -------- |
| 1     | 7.42 s        | 7.78 s    | 1.00×    | 7.24 s   | 8.62 s   | 2439             | 1.00×    |
| 2     | 3.71 s        | 4.06 s    | 0.52×    | 3.82 s   | 8.48 s   | 2397             | 0.98×    |
| 4     | 1.86 s        | 2.20 s    | 0.28×    | 2.08 s   | 7.98 s   | 2262             | 0.93×    |
| 8     | 0.93 s        | 1.28 s    | 0.16×    | 1.22 s   | 7.62 s   | 2160             | 0.89×    |
| 16    | 0.46 s        | 0.82 s    | 0.11×    | 0.78 s   | 7.40 s   | 2101             | 0.86×    |

Max_throughput (pool=8192) shows the same `~S×` peak-TTFT collapse (30.0 s →
2.2 s at 16 steps) but **100% of samples stay non-steady** at every step count —
see verdict.

### Reading the results (honest interpretation)

Two distinct quantities move very differently:

1. **Ramp severity (peak / p99 TTFT): reduced ~`S×`.** This is the headline win
   and it is large and robust in the model: `1/S` scaling, exactly the
   critically-damped prediction. The plot (`ramp_sim_concurrency.png`) shows the
   single tall burst spike replaced by `S` small sawtooth teeth, each `~peak/S`.

2. **Ramp duration / non-steady sample count: barely reduced (0.86× at 16
   steps).** Because the critically-damped interval keeps the total fill span at
   `target / prefill_rate`, staging spreads issuance over the _same_ wall-clock
   window into a still-warming pipe. It flattens the spike; it does **not** fill
   the pipe faster (the server rate is the hard bound). So the number of passes
   the warmup crop must drop is essentially unchanged.

The non-steady **count** exceeds `N` (2439 > 2048) because the decode CoV lets
some requests finish before the fill completes, and their replacements are issued
back into the still-draining queue.

## Verdict

- **Concurrency mode: adopt staged ramp; recommended schedule `S = 4–8` at the
  critically-damped interval** (`step = ceil(N/S)`, `step_interval ≈
step/prefill_rate`, ideally drain-driven per Open questions). Expected effect:
  peak and p99 ramp-up TTFT reduced `~4–8×` (model: 0.28× at S=4, 0.16× at S=8)
  at effectively unchanged ramp duration. `S=4` captures most of the benefit
  (0.28×) with the fewest steps; beyond `S≈8` returns diminish.
- **Expected reduction in the non-steady ramp region:** **severity** shrinks
  `~S×`; **extent** (duration / sample count / passes cropped) shrinks only
  modestly (`~0.9×`). Staged ramp is a _spike-flattener_, not a _ramp-shortener_.
  Its value for steady-state windowing is de-risking the warmup crop and
  shrinking the `total` vs `steady_state` divergence — **complementary to the
  crop, not a replacement**.
- **Max_throughput mode: staging is largely moot.** Max-throughput is a
  deliberately saturating burst with no steady in-flight target; every request
  queues behind the batch and there is no steady baseline. Staging redistributes
  _which_ requests wait (flattening peak TTFT `~S×`) but not _whether_ they wait,
  and it _lengthens_ the issue span (ramp_end grows with `S`). Only pursue it if
  a cold-start first-token artifact specifically needs smoothing; otherwise leave
  max_throughput as a pure burst.

## Assumptions & limits (this is a simulation)

- Single-bottleneck server: one FIFO first-token queue at a fixed `prefill_rate`.
  Real servers batch prefill/decode dynamically (continuous batching, chunked
  prefill), share compute between prefill and decode, and have queue-admission
  policies — `prefill_rate` is an effective aggregate, not a physical constant.
- TTFT modeled as `queue_wait + fixed floor`; no prefill-length dependence, no
  interference from concurrent decode, no KV-cache pressure or preemption.
- Decode time is i.i.d. lognormal, independent of arrival time and of load —
  real decode slows under contention, which would _amplify_ the burst penalty and
  thus _understate_ staging's benefit here.
- The replacement model refills to `N` only after the fill phase completes, so
  many-step ramps settle slightly below `N` in the plot (a model artifact of not
  refilling mid-ramp completions; it does not affect the peak-TTFT conclusion).
- Numbers are single-seed point estimates under one parameter set. Treat the
  `~S×` peak reduction as a shape result, not a calibrated forecast.

## Open questions for live validation

1. **Does the `S×` peak-TTFT reduction survive on a real server?** Continuous
   batching may already absorb part of the burst; measure burst vs `S=4` TTFT on
   one high-concurrency run (e.g. re-run a `C7168`/`C22528`-class point).
2. **Drain-driven vs fixed interval.** The critically-damped interval needs
   `prefill_rate`. Is a first-token-driven auto-advance (option 1) worth the
   plumbing of a first-token signal into the strategy, or is a probe-measured
   fixed interval sufficient?
3. **Does staging measurably shrink the passes the warmup crop must drop?** The
   model says barely — confirm on real `events.jsonl` that staged and burst runs
   need the same super-pass crop.
4. **MLPerf reproducibility.** A staged ramp adds `(S-1)·step_interval` of
   deterministic front-loading; confirm this is compatible with the run-length
   determinism the windowing design wants, and whether the ramp schedule must be
   recorded in the report for reproducibility.
5. **Throughput neutrality.** Confirm staged ramp does not depress measured
   steady-state throughput (it should not — steady region is post-crop — but this
   needs an A/B on real data).
