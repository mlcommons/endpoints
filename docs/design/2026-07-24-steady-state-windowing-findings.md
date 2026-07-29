# Steady-State Windowing — Strategy, Findings & Production Options

**Status:** Living document — team review draft.
**Date started:** 2026-07-24
**Scope:** Concurrency, offline (max-throughput), and poisson LLM inference benchmarks. Post-hoc analysis from `events.jsonl`.
**Related:** design `docs/design/2026-07-21-steady-state-windowing-design.md`, plan
`docs/design/2026-07-21-steady-state-windowing-plan.md`.
**Branch:** `design/steady-state-windowing`.

## 1. Problem

Concurrency-mode runs report artificially deflated throughput and inflated tail
latency. Two non-steady regions contaminate the measured window:

- **Ramp-up:** the client must issue enough samples to fill the target in-flight
  concurrency `N`. Until the pipe is full, a fill-burst of `N` requests queues
  against a finite-rate server; those requests' TTFT inflates dramatically, and
  the effect grows with `N`.
- **Ramp-down (drain):** after issuance stops, in-flight decays below `N` and
  throughput deflates.

The goal is **per-run steady-state correctness** — ensure a single run's report
reflects sustained steady behavior — not cross-run comparison (absolute latencies
are not comparable across points that use different deployments).

## 2. The strategy, end to end

The pipeline is a sequence of pure functions over a run's event log. Each stage
is independently testable; nothing touches the hot path.

### 2.1 Ingest → per-super-pass series

Re-read `events.jsonl` (the durable log; mirrors
`scripts/early_stopping_estimate_from_events.py`). For each performance-tracked
sample, record issue / first-token / complete timestamps. Bucket samples by
**issue order** into **super-passes**.

- **Super-pass:** `S = ceil(N / dataset_size)` dataset passes = the minimum
  issuance to hold `N` distinct samples in flight; `super_pass_samples =
dataset_size * S`. The k-th first-time-issued sample belongs to
  `super_pass = k // super_pass_samples`. (Duplicate ISSUED = retry; does not
  advance the counter.) The super-pass is the atomic unit for every downstream
  decision, so windows always contain a representative full-dataset workload mix
  even when `dataset_size < N`.

Each super-pass rollup carries raw TTFT / latency / TPOT values, output-token
count, and first/last issue timestamps.

### 2.2 Issue-time window (the core correction)

The measured set for any window `[sp_start, sp_end)` is every request **issued**
in that span. Metrics are computed as:

- **Throughput denominator = issue-time span** `last_issue − first_issue`. In
  concurrency mode in-flight is held at `N` by construction until issuance stops,
  so issue-rate = completion-rate across the span, and the **drain lives after
  `last_issue` — it never enters the denominator**. No end-crop is needed.
- **Latency / tokens = full lifetime of that same set**, including the high-OSL
  requests that complete during drain (those are wanted for the tail).

One membership set feeds both metric families, so throughput and latency can
never disagree on which samples they measured. This is the key insight: define
the window on **issue time, not completion time**.

### 2.3 Coverage status (never hard-fail)

`n_full_super_passes = n_issued // super_pass_samples`; classify the run:

- `partial_dataset` — issued < 1 full dataset pass (a low-C run time-capped
  mid-pass): biased ISL/OSL subset, low confidence.
- `insufficient_passes` — ≥1 pass but `< warmup+1` super-passes.
- `windowable` — enough for a normal warmup-cropped window.

Non-windowable runs get a flagged best-effort window instead of an error, so a
batch sweep over runs of any length stays complete.

### 2.4 Adaptive warmup (drop the settling transient)

The fill burst drains over **many** super-passes at high concurrency — a fixed
one-super-pass crop is far too small (c28080 keeps settling for ~7 super-passes /
20 min). Warmup is chosen by a **band rule**:

1. Estimate the steady level = median of the driver metric (p99 TTFT — the
   slowest to settle) over the series' back half.
2. Drop leading super-passes whose driver value exceeds `steady * (1 + band)`.

Band-based (not slope-based) so a single settling outlier can't collapse the
crop, and only HIGH leading values are dropped — a brief settle followed by an
upward drift still leaves the drift measurable.

### 2.5 Per-metric drift detection (does a steady state even exist?)

For each metric (QPS, p50/p99 TTFT, p50/p99 latency), fit an OLS trend to the
post-warmup per-super-pass trajectory and classify:

- **`steady`** — the run-length change is small relative to both the typical
  value AND the super-pass-to-super-pass residual scatter.
- **`drifting_up`** — sustained upward trend: **pathological, no steady state**
  (e.g. p99 TTFT worsening across the whole run). Do NOT report a single steady
  value; surface the trend.
- **`drifting_down`** — sustained downward trend after warmup: **residual
  settling** (the driver-based warmup under-cropped this slower-settling metric),
  not pathological.

This is the crucial addition: **steady state is metric-dependent.** A run can be
steady in QPS and median but have no steady p99 TTFT.

### 2.6 CoV-detector ensemble (corroboration)

Run the adaptive-CoV stopping rule at several `(window, bound)` settings and
measure **concordance** (do most detectors converge near the same super-pass?).
A single fixed `(window, bound)` cannot serve all concurrencies, but the ensemble
is robust: tight concordance corroborates a steady state; **0/N converged
corroborates "no steady state."** CoV, not KL divergence — the HDR histogram
bucket edges shift per snapshot, so KL compares mismatched bins.

### 2.7 Output

Per run: `status`, adaptive warmup (+ settle passes / samples / wall-clock time),
ensemble concordance, and a per-metric verdict table. A `steady` metric reports
its windowed value; a `drifting_up` metric is flagged instead of reporting a
false steady number.

## 3. Production implementation

### 3.1 Today: pure post-processing (offline)

Everything above is **cold-path**: it re-ingests the durable `events.jsonl` after
the run, exactly like `early_stopping_estimate_from_events.py`. No hot-path
change, no dependency on live snapshots (which can lag under load). The run
report grows a `steady_state` block alongside `total`; steady-state is the
official number and the `total`-vs-steady divergence is itself a signal. This is
the most accurate mode — it has the whole series, so the band warmup and the
drift trend both see the full run.

### 3.2 Can it work in flight? Partially — with caveats

The per-super-pass series can be built **incrementally** as events stream (the
metrics aggregator already sees them live; super-pass boundaries are known from
the issue count). Which stages can run live:

- **Forward CoV ensemble → yes.** `rule_cov_converged` is a forward/streaming
  detector; it can fire "steady reached" during the run (early-stopping style),
  enabling an **adaptive run duration** — stop once steady is detected plus a few
  measurement super-passes have accumulated.
- **Band adaptive warmup → no (retrospective).** The band needs the steady level
  = median of the back half, which only exists once the run is mostly done. A
  live surrogate is **forward settling detection** (declare settling complete
  when the driver stops decreasing beyond the noise floor), but it is less robust
  than the retrospective band.
- **Drift (`drifting_up`) detection → inherently needs duration.** You cannot
  know a metric will keep drifting up until you have watched it for a while. Live,
  the best you can do is _not_ declare steady while a trend is still present, and
  warn "no steady state reached yet."

### 3.3 Recommended: hybrid

- **Live (in aggregator):** forward CoV ensemble + forward settling detector to
  drive **adaptive stopping** — end the run when steady is detected and enough
  measurement super-passes exist, or warn/extend when it will not settle. Saves
  wall-clock on runs that stabilize fast; flags runs that never do.
- **Offline (in report):** the full pipeline (band warmup + per-metric drift +
  asymptote) as the **official, audited number**. If a metric is `drifting_up`,
  the report says so instead of publishing a false steady value.

Integration point: the aggregator already produces per-sample series; add a
super-pass-boundary snapshot cadence for the live path, and run the pure
functions in `Report.from_snapshot` for the offline path (design-doc Milestone 2).

## 4. Corpus

GB300-NVL72, TRT-LLM disagg, concurrency mode.

| workload     | dataset_size | runs analyzed                                              |
| ------------ | ------------ | ---------------------------------------------------------- |
| gpt-oss-120b | 6396         | c8 (partial), c140, c1024, c2048, c7168, c22528            |
| DeepSeek-R1  | 4388         | c28080 (`_numa`, sha `e072ac2`, tentative — trust pending) |

Absolute latencies are **not** comparable across concurrency points — each uses a
different disagg deployment (ng/nc/dep differ). Per-run correctness only.

## 5. Findings

### 5.1 Ramp deflates tail latency, not throughput

p99 TTFT, total (warmup-included) vs steady:

| C     | total p99 TTFT | steady p99 TTFT | recovery        | QPS Δ |
| ----- | -------------- | --------------- | --------------- | ----- |
| 140   | 0.294 s        | 0.247 s         | −16%            | ~0    |
| 1024  | 1.29 s         | 1.11 s          | −14%            | ~0    |
| 2048  | 1.18 s         | 0.96 s          | −19%            | ~0    |
| 22528 | 15.72 s        | 2.47 s          | **−84% (6.4×)** | −1%   |

QPS (offered-rate throughput) is barely touched by the ramp; the artifact is
almost entirely in the **tail latency**.

### 5.2 No universal `(cov_window, cov_bound)`

Low/mid concurrency settles in ~3 super-passes (`cov_window=3`, any bound, ~0–1.5%
error). c22528 needs `cov_window=6, cov_bound=0.15`. c7168 has no config under
35% error. Each super-pass's p99 is ~256 tail samples, so a CoV bound below that
sampling-noise floor (≤0.02) never converges. A fixed config cannot serve all
concurrencies — hence the ensemble.

### 5.3 p99 TTFT often has no steady state; it drifts up

Per-super-pass p99 TTFT climbs monotonically through the whole run while the
median settles:

- c7168: post-fill 2.9 s → 10.9 s (3.8×) over ~40 min.
- c22528: 0.65 s → 3.2 s (~5×). The drift detector (after 2-super-pass band
  warmup) flags this `drifting_up` (rel_drift +1.64, snr 7.8).
- c28080 (DSR1): p99 TTFT starts at 80.7 s (the 28080-deep fill), settles to a
  ~7 s plateau by super-pass 7, then holds — `steady` after warmup. (This
  reproduces the original ~82 s c28k+16k observation.)

Median-down / tail-up over a sustained run is **not throttling** (decode logs
show zero preempt/evict/OOM; a global slowdown would raise the median too). It is
progressive tail degradation — a growing scheduling imbalance or KV fragmentation
starving the worst-case requests. Root cause not yet pinned; the measurement
conclusion holds regardless.

### 5.4 Warmup scales with concurrency

The settling region is not one super-pass. c28080 needs **~7 super-passes = 49
dataset passes = ~215k samples = ~19 min** of warmup before the p99 TTFT plateau.
The adaptive band warmup computes this automatically and reports the settle cost,
answering "how long / how many samples to reach steady" directly. The c28080 run
(22 super-passes, 658k samples, 67 min) had ~2× margin; `n_samples_to_issue`
could drop to ~400k and still capture a solid steady window.

## 6. Alternatives considered

| approach                                                                    | verdict                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fixed-budget stop (rule A)**                                              | Too short at high C — 46% p99 error at c22528 (stops before steady). Kept only as a baseline.                                                                                                                                                                                                                 |
| **Single-config adaptive CoV (rule B)**                                     | Right idea, but no universal `(window, bound)` (§5.2). Superseded by the ensemble.                                                                                                                                                                                                                            |
| **KL divergence for convergence**                                           | Rejected: HDR bucket edges shift per snapshot, so KL compares mismatched bins. CoV of scalar percentiles is bucket-independent.                                                                                                                                                                               |
| **Feathered / staggered start** (issuance ramp-in so the fill never bursts) | Orthogonal — attacks the ramp at _issuance_ time, not measurement. Non-reproducible for MLPerf and impossible to time perfectly with high-variance OSL (can't predict when a request ends). Complementary to, not a replacement for, post-hoc windowing; a gentle ramp-in could shrink the region we discard. |
| **Measure [fill-complete → first-drain]** (occupancy framing)               | Equivalent to our issue-time window in concurrency mode: in-flight never drops below `N` until issuance stops, so "first drain" = `last_issue`. Convergent design.                                                                                                                                            |
| **Ensemble of CoV checkers + significance** (adopted)                       | Evolved into per-metric plateau-vs-trend detection + ensemble concordance — the current strategy.                                                                                                                                                                                                             |
| **OLS trend + band adaptive warmup + directional verdict** (current)        | Catches upward drift, distinguishes it from settling, computes settle cost. Open gap: single-driver warmup under-crops slower-settling metrics (labeled `drifting_down`, not wrong). Per-metric warmup is the follow-up.                                                                                      |

## 7. Data-validity note — client concurrency cap

The single-process endpoints-client hits the Linux ephemeral-port limit, so
**effective concurrency caps at ~28k** (verified via `ss`). Configured c32768 /
c65536 runs could not reach their targets — their behavior clustered just above
c16k. Those logs are debugging artifacts and **must not be used**; the max
concurrency config was capped to **28080** (the achieved ceiling). For c32768
events, super-pass math must use the achieved concurrency (28080), not 32768.

## 8. Cross-mode detection — algorithms, criteria, ablations, failure modes

The load pattern determines both _what deflates_ and _which detector applies_. Three
modes, three worked examples from real DSR1/gpt-oss GB300 runs.

### 8.1 Mode taxonomy + worked examples

| mode                    | issuance           | deflation source                    | metric that matters     | detector                                                          | worked example                                                                                                |
| ----------------------- | ------------------ | ----------------------------------- | ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **concurrency**         | hold `N` in flight | fill ramp + **p99 TTFT drifts up**  | tail latency            | issue-time super-pass window + adaptive warmup + per-metric drift | c22528: total p99 TTFT 15.7 s → steady 2.47 s (−84%); p99 TTFT `drifting_up` (rel_drift +1.64)                |
| **offline** (max-tput)  | burst at `t=0`     | **throughput drain tail** (backlog) | sustained throughput    | completion-rate plateau-edge                                      | 293038: plateau min 11–95, 279 req/s vs 191 full-run (**31% undercount**); 272936: plateau min 7–72 (**12%**) |
| **poisson** (under-sat) | fixed rate `qps`   | short queue-fill ramp only          | latency (rate is fixed) | _pass-unit = the concurrency tool, `--concurrency 1`_             | 232386 (37 qps): adaptive warmup 2 passes, **all metrics steady**, qps=37, ttft_p99 flat ~330 ms              |

Key cross-mode facts learned from the data:

- **Concurrency is the only mode where a metric has no steady state** (p99 TTFT drifts up).
- **Offline's tail is a backlog artifact**, not present in the workload's steady behavior;
  its metric is a _rate over the plateau_, which sidesteps the dataset-boundary/OSL-bias
  concern entirely (§5, §6).
- **Under-saturated poisson is genuinely steady** — completion≈offered rate, ~3-min drain;
  it needs no new machinery. A _near-saturation_ poisson would backlog like offline (untested).
- Offline is typically **non-streaming** → no `recv_first`/TTFT; throughput-only.

### 8.2 Concurrency algorithm (exact criteria)

1. **Super-pass unit.** `S = ceil(N / dataset_size)`; `super_pass_samples = dataset_size * S`.
   The k-th first-time-issued sample (retries excluded) → `super_pass = k // super_pass_samples`.
2. **Coverage status.** `partial_dataset` if `n_issued < dataset_size`; else
   `insufficient_passes` if `n_full = n_issued // super_pass_samples < warmup + 1`; else
   `windowable`. Non-windowable → flagged best-effort, never a hard error.
3. **Issue-time window** `[sp_start, sp_end)`: throughput denominator =
   `last_issue − first_issue`; latency/token percentiles over the full lifetime of the
   same issued set (drain completions kept). `percentile_lower` (method="lower",
   index `int(p·(n−1))`).
4. **Adaptive warmup (band).** `steady = median(driver p99 TTFT over the back half)`;
   drop leading super-passes while `value > steady · (1 + band)`, `band = 0.5`, `min_warmup = 1`,
   cap `0.5·len`. Band-based (not slope) so a single settling outlier can't collapse the crop,
   and only HIGH leading values are dropped (a brief settle then upward drift stays measurable).
5. **Per-metric trend.** OLS fit over the post-warmup per-super-pass series;
   `total_change = slope·(n−1)`; `rel_drift = total_change / median`; `resid_std = pstdev(residuals)`;
   `snr = |total_change| / resid_std`. **Drift iff `|rel_drift| ≥ 0.15` AND `snr ≥ 2.0`.**
   Verdict `drifting_up` (rel_drift > 0, pathological) / `drifting_down` (< 0, residual settling) / `steady`.
   `n < 4` → `insufficient`.
6. **CoV ensemble (corroboration).** Run `rule_cov_converged` at 6 configs
   `(window,bound) ∈ {(3,.03),(3,.05),(4,.05),(5,.08),(6,.10),(6,.15)}`;
   `concordance = max(0, 1 − (max sp_end − min sp_end)/(len − warmup))`.
   **`0/6` converged ⇒ no steady state.**

**The CoV convergence test in depth — math → `events.jsonl` → ensemble.**

_The statistic._ The coefficient of variation is the population standard deviation
over the mean:

```
CoV = σ / μ = √( (1/N) Σ (xᵢ − μ)² ) / μ
```

It is **dimensionless** (σ and μ share units and cancel), so `CoV = 0.05` means
"the spread is ≤ 5 % of the typical value" for _any_ metric. The code:

```python
from statistics import pstdev
def cov(values):
    if len(values) < 2: return 0.0     # no spread definable for <2 points
    m = sum(values) / len(values)
    if m == 0: return 0.0              # guard: CoV undefined at mean 0
    return pstdev(values) / abs(m)     # POPULATION stdev / |mean|
```

Population stdev (÷ N, not ÷ N−1): we describe the window we hold, not infer a
larger population. Unitless is the point — one `bound` works across TTFT (~2 s),
e2e latency (~180 s), and QPS alike, where a raw-stdev threshold would have to be
retuned per metric and per concurrency. CoV of scalar percentiles, **not KL
divergence over the histograms**, because HDR bucket edges are re-derived per
snapshot so KL would compare mismatched bins.

_What `x` is — application to `events.jsonl`._

1. Ingest → per-super-pass series (§8.2 step 1): each super-pass holds that
   super-pass's raw TTFT/latency arrays.
2. Reduce each super-pass to **scalar percentiles** via `percentile_lower` over
   its own values: p50 and p99 of TTFT and of latency → one number per
   (metric, super-pass). These per-super-pass scalars are the `xᵢ`.
3. `rule_cov_converged(window=W, bound=b, warmup)` scans `sp_end` from
   `warmup + W` upward; over the **trailing window** `series[sp_end − W : sp_end]`
   it computes `cov(...)` of each of the four metric series (p50/p99 × TTFT/latency).
4. **Converged at `sp_end` iff _every_ metric's CoV `< b`.** The first such
   `sp_end` yields the steady region `(warmup, sp_end)`; `None` if it never flattens.

So CoV is measured _across super-passes within a sliding window_, per metric —
"has this percentile stopped moving, relative to its own level, over the last `W`
super-passes?"

_The ensemble (exact configs)._ No single `(window, bound)` serves all
concurrencies (§8.5), so a **fixed 6-detector ensemble** runs in parallel:

```
(window, bound) ∈ { (3, 0.03), (3, 0.05), (4, 0.05), (5, 0.08), (6, 0.10), (6, 0.15) }
```

— from short-window/tight-bound (fast, converges early at low C) to
long-window/loose-bound (averages down the jitter, needed at high C). Each detector
returns its own `sp_end` (or nothing). **Concordance** measures agreement:

```
concordance = max(0, 1 − (max sp_end − min sp_end) / (len − warmup))
```

≈ 1 when detectors agree on where steady begins (corroborated steady state), lower
when they scatter. **`0 / 6` converged ⇒ no steady state** — e.g. c7168's p99 TTFT
drifts the whole run, so no window ever flattens under any bound.

_The noise floor._ Each super-pass p99 is estimated from ≈256 tail samples, so it
jitters a few percent super-pass to super-pass even in true steady state; CoV cannot
fall below that sampling floor. Bounds `≤ 0.02` therefore **never** converge at high
C — structural, not a tuning miss, and precisely why the ensemble mixes bounds
rather than pinning one tight value.

### 8.3 Offline algorithm (exact criteria)

1. **Completion-rate series.** Bin `sample.complete` events by completion time into 1-min
   bins; `rate[b] = completions in bin b`. (Issue-time is degenerate — all at `t=0`.)
2. **Plateau-edge (band + longest run).** `steady = median(rate)`;
   `lo = steady·(1−band)`, `hi = steady·(1+band)`, `band = 0.10`. The **plateau is the longest
   contiguous run of bins with `lo ≤ rate ≤ hi`.** Its start = end of the leading ramp/settle;
   its end = **drain-onset**.
3. **Report.** Sustained throughput = `median(rate[plateau])` (× tokens/req for tok/s). No
   dataset-boundary snap — a rate over the plateau reflects the steady ISL/OSL mix, so there is
   no sample-set to bias.

Why "longest in-band run" and not "drop until in band then until out of band": the leading
region can be _below_ the band (partial first bin), _above_ it (ramp overshoot / fast-request
settle), or both — a single directional scan mis-handles it. The longest-run picks the flat
plateau regardless of leading shape (verified: 293038 settles _down_ into the band, 272936 ramps
_up_ into it; both resolve correctly).

### 8.4 Poisson (reuse, unchanged)

Poisson issues over time in dataset passes, so it is the concurrency algorithm with `N`
undefined ⇒ `S = 1` (one pass per super-pass, run with `--concurrency 1`). The adaptive warmup
crops the queue-fill ramp; the per-metric drift test runs on per-pass TTFT/latency. Verified on
232386 with no code changes. Throughput = offered `qps` (not a measured quantity); latency is
the target.

### 8.5 Hyperparameters & ablations

**Concurrency `(cov_window, cov_bound)` — best per run** (scored vs full-series asymptote, target p99-TTFT err ≤ 5%):

| C     | best `(w, b)` | super-passes | p99-TTFT err                           |
| ----- | ------------- | ------------ | -------------------------------------- |
| 140   | (3, 0.05)     | 3            | 0.0%                                   |
| 1024  | (3, 0.03)     | 3            | 1.5%                                   |
| 2048  | (3, 0.08)     | 3            | 0.3%                                   |
| 22528 | (6, 0.15)     | 8            | 1.6%                                   |
| 7168  | (5, 0.05)     | 11           | 35.7% (none met target — drifting run) |

- **Noise floor:** each super-pass's p99 is estimated from ≈256 tail samples, so it carries
  a few-percent sampling jitter; `cov_bound ≤ 0.02` is below that floor and **never converges**
  at high C. This is structural, not a tuning miss — it motivates the _ensemble_ (mix of bounds)
  over one fixed pair.
- **Universal-ish default:** `(6, 0.15)` is the only single config under 5% at c22528 while
  still fine at c1024 (0.9%). Longer window + looser bound wins at high C (smooths the jitter).

**Drift thresholds `(rel_drift ≥ 0.15, snr ≥ 2.0)`:** chosen so a _noisy-but-flat_ series
(large residual scatter, small net change) is not mistaken for a trend. Lowering `snr` toward 1
starts flagging noise; raising `rel_drift` past ~0.25 misses the slow c1024-scale drifts.
`band = 0.5` (warmup): large enough to keep the plateau, small enough to drop the fill spike
(c28k fill p99 100 s ≫ 7 s plateau → dropped; c22528 sp1 5.3 s vs ~1.6 s steady → dropped).

**Offline `band = 0.10`:** resolves both example runs (plateaus are flat to ±few %). Tighter
(0.05) risks fragmenting a slightly-sloped plateau into multiple short runs; looser (0.20) risks
swallowing the ramp shoulder or early drain. Robust because `median` lands in the plateau
whenever plateau bins outnumber tail bins (true for both examples: 85/157 and 66/88).

### 8.6 Failure modes & how they're handled

**Concurrency:**

- _A metric has no steady state (p99 TTFT drifts up)._ Detected: `drifting_up` verdict + the
  ensemble fails to converge tightly. Handled by **flagging** rather than reporting a false steady
  value. c7168 is the extreme — _no_ `(w,b)` lands within 5% of the asymptote because the
  asymptote itself sits on a rising ramp; the detector reports drift instead of a number.
- _Metrics settle at different rates._ A single `ttft_p99`-driven warmup under-crops a
  slower-settling metric (c28k `lat_p99`), which then reads `drifting_down`. Handled by the
  **directional label** (down = residual settling, transparent — not a false alarm). Follow-up:
  per-metric warmup.
- _Settling outlier (c22528 sp1)._ A single high post-fill bin would collapse a slope-based
  warmup. The **band + back-half-median** warmup ignores it (drops only values above the band).
- _Too-short run._ `partial_dataset` / `insufficient_passes` → best-effort window over all
  available super-passes (`warmup = 0`) with the status flag; trend returns `insufficient` for
  `n < 4`.

**Offline:**

- _Variable ramp shape_ (settle-down vs ramp-up): the **longest-in-band run** is shape-agnostic
  (§8.3). Verified on both.
- _Second-issuance blip._ Both example runs re-issue exactly one dataset pass late in the run
  (293038 @ min 150, 272936 @ min 81), producing a small completion bump inside the drain tail.
  It falls **after** drain-onset (outside the longest in-band run) and does not perturb the
  plateau. Handled implicitly.
- _Short or very tail-heavy plateau._ If tail bins outnumber plateau bins, `median` can slip
  below the plateau and mis-center the band. Mitigation for such runs: estimate `steady` from an
  upper quantile (e.g. p75) or the histogram mode instead of the median. Not needed on the
  examples (plateau dominates); flagged as a known limitation.
- _Few bins_ (very short run): with < ~10 bins the plateau/band is unreliable — fall back to
  reporting the full-run rate with a low-confidence flag.

**Poisson:**

- _Near saturation._ At an offered rate above capacity, poisson backlogs and grows an
  offline-style drain tail; the pass-unit latency view alone would miss it. Handling: detect the
  backlog (completion rate falling below offered rate) and switch to / add the offline
  completion-rate plateau-edge. **Untested — needs a near-saturation poisson run (the missing
  279320 case).**

**Cross-cutting:**

- _Non-streaming runs_ (offline): `recv_first` absent ⇒ TTFT metrics are simply N/A; the tool
  reports throughput/e2e only rather than erroring.
- _Malformed/truncated log lines_ are skipped (count logged), so a partially-written
  `events.jsonl` yields a best-effort series rather than a crash.

### 8.7 Edge cases summary

| edge case                                | detection                          | handling                                                      |
| ---------------------------------------- | ---------------------------------- | ------------------------------------------------------------- |
| < 1 full dataset pass                    | `n_issued < dataset_size`          | `partial_dataset`, best-effort, low-confidence flag           |
| ≥1 pass, `< warmup+1` super-passes       | `n_full < warmup+1`                | `insufficient_passes`, window all super-passes (`warmup=0`)   |
| metric drifts up (no steady state)       | trend `rel_drift ≥ 0.15 ∧ snr ≥ 2` | `drifting_up` flag; do not emit a steady value                |
| metric still settling after warmup       | trend, `rel_drift < 0`             | `drifting_down` flag (residual settling, not pathological)    |
| CoV bound below the sampling-noise floor | `0/6` ensemble converge            | reported as "no steady state"                                 |
| offline second-issuance blip             | occurs after drain-onset           | outside the longest in-band run; ignored                      |
| tail-heavy offline plateau               | plateau bins < tail bins           | switch `steady` estimator to upper-quantile/mode (limitation) |
| non-streaming (no TTFT)                  | `recv_first` absent                | TTFT N/A; throughput/e2e only                                 |

## 9. Open questions

- Does `drifting_up` reproduce on DSR1 (different workload) and on a clean
  capped-28080 re-run? (The tentative c28080 `_numa` is not yet trusted.)
- Root cause of the progressive tail degradation (prefill queue vs KV
  fragmentation).
- Per-metric adaptive warmup (metrics settle at different rates).
- Live/in-flight forward detectors: how aggressively to adaptively stop vs. the
  reproducibility a fixed duration gives.
- MLPerf policy: how to define/attest a steady window, and how to report a
  metric that has no steady state (drift).

## Appendix — code & reproduction

- Library: `src/inference_endpoint/metrics/steady_state/`
  (`series`, `window`, `stopping`, `harness`, `drift`).
- Scripts: `steady_state_from_events.py` (total vs steady + A/B),
  `steady_state_cov_sweep.py` (`cov_window × cov_bound` ablation),
  `steady_state_drift.py` (adaptive warmup + per-metric drift + ensemble).
- Tests + tooling run in the Linux dev container (`inference-endpoint-dev`,
  `--shm-size=8G`); pre-commit on host. Example:

```
docker run --rm --shm-size=8G -v "$PWD":/mnt/inference-endpoint -w /mnt/inference-endpoint \
  -v <artifacts>:/data inference-endpoint-dev bash -lc \
  "uv run python scripts/steady_state_drift.py /data/C22528/events.jsonl \
     --dataset-size 6396 --concurrency 22528"
```
