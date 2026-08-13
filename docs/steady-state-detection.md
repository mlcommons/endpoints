# Steady-State Metrics Reporting

# 1 Objective {#1-objective}

Add a post-processing step that runs after a benchmark completes and reports the
sustained steady-state metrics, rather than the whole-run average,
which is deflated by the ramp-up and drain transients. The step is a pure
function over the durable event log; it also ships as an ad-hoc command-line tool
that ingests an `events.jsonl` file.

Goals:

- Emit a `steady_state` block alongside the existing whole-run (`total`) metrics
  in the run report, reported as the **official result** with the whole-run
  (`total`) metrics retained as supplementary context.
- Detect _when there is no steady state_ (a progressively degrading run) and say
  so, instead of reporting an unstable/unsteady result.
- Provide a single implementation reachable two ways: automatically as the
  post-processing step at the end of a run, and manually as the same step invoked
  ad hoc over any recorded `events.jsonl`.
- Add no cost to the measured run: all work is off the hot path.

Non-goals:

- Changing the whole-run (`total`) numbers, the live metrics snapshot cadence, or
  the wire schema of the live aggregator.
- Changing how load is issued during a run. A staggered-issuance option is
  discussed as complementary (§5.7) but is out of scope for this step.
- Accuracy scoring, submission checking, or any change to the audit path.

# 2 Background {#2-background}

A run's reported metrics today are aggregated over the entire measurement window.
Two regions of that window are not steady state:

- **Ramp-up.** At the start of a run the client raises offered load to its target.
  Under a concurrency load pattern the target in-flight population is filled as a
  burst; against a finite-rate server the leading requests queue and
  time-to-first-token (TTFT) inflates, and the inflation grows with concurrency.
- **Ramp-down (drain).** After issuance stops, in-flight decays below the target
  while the last requests finish. Throughput deflates because the wall-clock
  denominator keeps advancing while offered load is below steady state.

Averaging over these transients understates throughput and overstates tail
latency. The magnitude is workload-dependent and can be large for the tail: in
experiments over recorded runs (single-turn concurrency, offline/max-throughput,
Poisson, and multi-turn agentic), the reported p99 TTFT
was dominated by the ramp spike and fell substantially once the ramp was
excluded, while per-token latency (TPOT) was essentially unchanged.

There is already a precedent in the codebase for exactly this shape of feature:
`src/inference_endpoint/metrics/early_stopping.py` computes MLPerf early-stopping
percentile estimates as a cold-path calculation, and
`scripts/early_stopping_estimate_from_events.py` re-runs the same math ad hoc
from a recorded `events.jsonl` (see `docs/early_stopping.md`). This design follows
the same two-entry-point pattern. There is also a precedent for a post-run
orchestration step in `src/inference_endpoint/commands/audit.py`, which
`src/inference_endpoint/commands/benchmark/cli.py` dispatches after the main
benchmark completes.

# 3 Assumptions and Risks {#3-assumptions-and-risks}

- **The durable event log is complete and authoritative.** The step reads the
  per-sample event stream, not the live snapshot (which can lag under load). Risk:
  a run killed by `SIGKILL`/OOM before the log is flushed yields a truncated log;
  the step must degrade to a best-effort result with a status flag rather than
  fail (§5.6).
- **Per-token latency is approximately sample-invariant in a healthy run.** The
  guarded tail-cut (§5.4) rests on this. It held across the recorded corpora, but
  it is a property of a healthy run, not a guarantee, so the cut is _guarded_: it
  is applied only when the condition is measured to hold, and otherwise the tail
  is retained.
- **Bucketing is by issue order.** The unit of analysis is a fixed-size group of
  issued requests (§5.1). Risk: for load modes with no repeated dataset pass, the
  group size is a free parameter; experiments show the qualitative verdict is
  robust to that choice, but it is called out as tunable (§6).
- **Token counts are available or derivable.** TPOT needs an output-token count
  per request. When the run already records one it is used directly; otherwise the
  step tokenizes on the cold path (§5.6). Risk: cost on very large logs, bounded
  by sampling.

# 4 Alternatives considered {#4-alternatives-considered}

- **Fix it live, in the aggregator.** Detect and crop the ramp inside the hot-path
  aggregator so `total` is already steady. Rejected: it adds latency-critical work
  to the hot path, the live snapshot can lag under load, and it couples a
  still-evolving heuristic to the measured numbers. A cold-path step keeps the hot
  path untouched and the heuristic revisable.
- **Report a fixed time/percentage crop (e.g. drop the first N seconds).** Simple
  but wrong across modes and concurrencies: the ramp length depends on load, and a
  fixed crop under- or over-cuts. The adaptive, data-driven window (§5.2, §5.5)
  self-sizes.
- **Only exclude the ramp; keep the whole tail.** Leaves the drain in the
  throughput denominator, re-deflating it. Defining the window on _issue time_
  (§5.1) excludes the drain from throughput for free, without an end-crop.
- **Trust a single convergence detector.** A lone coefficient-of-variation (CoV)
  stopping rule converges even on a slowly drifting series, picking a window far
  from the asymptote. Rejected in favor of an ensemble plus a mandatory trend gate
  (§5.5).

# 5 Design {#5-design}

## 5.1 Overview and definitions

The step is a pure function from a recorded event stream to a `steady_state`
result. Definitions used throughout:

- **Healthy server.** A server that can support the maximum load issued by the
  client. The window measures the sustained behavior of a healthy server; a
  genuinely unhealthy server produces bad-but-real numbers, which the drift
  detector (§5.5) distinguishes from transient pollution.
- **Super-pass.** The atomic unit of the analysis: a contiguous block of requests
  in _issue order_, sized so each block is a representative full-dataset workload
  mix. It is named distinctly from a _dataset pass_ on purpose — it is **not**
  always one pass. A low-concurrency run may not issue even a single full pass, and
  the block size is a tunable hyperparameter (for example two dataset passes per
  super-pass, to reduce per-super-pass variance). In the common case it is exactly
  one dataset pass (`dataset_size` requests), so a run issues about
  `ceil(N / dataset_size)` super-passes, where `N` is the total number of samples
  issued.
- **Long-running sample.** A sample whose output length (OSL) or sample latency is
  much larger than the dataset average.
- **Hairball.** A build-up of long-running samples that grows as more dataset
  passes are issued and completed: because datasets are issued without
  replacement, fresh copies of a long-running sample are issued before earlier
  copies finish, so long-running samples remain in flight long after the rest have
  completed.
- **Hairball weight.** At issuance-stop under max-concurrency `C`, the percentage
  of in-flight samples that are _not_ among the last `C` issued (the lingering
  hairball). Ideally 0 — at stop the `C` in-flight samples would be exactly the
  last `C` issued.
- **Relative active concurrency.** In-flight count as a percentage of the
  max-concurrency budget (100% at saturation).
- **Issue-time window.** A contiguous range of super-passes `[start, end)`. Its
  measured set is every request _issued_ in that range. Throughput uses the
  issue-time span `last_issue - first_issue`; latency and per-token metrics use
  the full lifetime of that same set. One membership set feeds both families, so
  they can never disagree on which requests they measured. A sample enters the
  window only if it has at least one logged event (for example a first token)
  inside it — only metrics _logged within_ the window are counted, so a sample that
  was issued but received no response contributes no logged metric and is naturally
  excluded. The drain lives after the last issue, so it never enters the throughput
  denominator — no end-crop is needed.
- **Steadiness metrics.** Metrics whose variation reflects _system state_ rather
  than workload composition: TTFT (admission / prefill queue) and TPOT (per-token
  decode). Both are output-length-independent. End-to-end latency is deliberately
  excluded — it is TTFT plus decode time, and decode scales with output length, so
  its variation tracks the output-length mix, not steadiness.

## 5.2 Where the step runs

The step is invoked automatically after a run completes, and is also runnable
standalone. Both paths call the same pure-function core.

```text
  +--------------------------------------+
  |  benchmark run (load gen + workers)  |
  +--------------------------------------+
                     |
                     |  emits events -> durable event log
                     v
  +--------------------------------------+
  |  live metrics aggregator (hot path)  |
  |  writes final_snapshot.json [total]  |
  +--------------------------------------+
                     |
                     |  run ends (COMPLETE); cold path begins
                     v
  +--------------------------------------+
  |  steady-state post-process           |
  |  reads the event log, off hot path   |
  +--------------------------------------+
                     |
                     |  steady_state block
                     v
  +--------------------------------------+
  |  Report { total, steady_state }      |
  +--------------------------------------+
```

- **Automatic path.** The run-completion path in
  `src/inference_endpoint/commands/benchmark/execute.py` (finalize) — or the
  dispatch in `src/inference_endpoint/commands/benchmark/cli.py`, mirroring how it
  already dispatches `src/inference_endpoint/commands/audit.py` — invokes the
  steady-state builder over the durable event log after the live aggregator has
  written `final_snapshot.json`. The builder returns a `steady_state` result that
  `src/inference_endpoint/metrics/report.py` attaches next to `total`. This is
  gated by a new settings field in `src/inference_endpoint/config/schema.py`,
  following the existing `early_stopping.enabled` flag.
- **Ad-hoc path.** A new script re-runs the identical core over any recorded
  `events.jsonl`, mirroring `scripts/early_stopping_estimate_from_events.py`. This
  is the tool used to analyze historical runs and to iterate on parameters
  without re-running a benchmark.

The builder never reads the live snapshot; the durable event log is the source of
truth, consistent with the existing early-stopping recomputation path.

## 5.3 The analysis pipeline

The core is a sequence of pure stages over the per-super-pass series.

```text
  event log
     |
     v
  [ ingest -> per-super-pass series ]     issue-order bucketing
     |
     v
  [ adaptive warmup crop ]                remove the ramp
     |
     v
  [ guarded drain-tail cut ]              only if per-token-invariant
     |
     v
  [ convergence: CoV ensemble + trend ]   window edge + steady/drift verdict
     |
     v
  { steady_state metrics + status }
```

### The hairball (drain tail)

The dataset is issued in full, without replacement, repeating across passes, so a
tail of long-output requests accumulates toward the end of every run; only the
magnitude differs by mode. Under concurrency the tail is exactly the in-flight
population at issuance-stop — a representative issue-time snapshot, small for
uniform-output models. Under offline/max-throughput **every** sample is issued in
one huge burst at `t=0` and it is left to the server to work through the flood, so
from the client's perspective there is no issue-phase/drain boundary at all — under
the normal (issue-time) definition the entire run, or very nearly all of it, is
drain, which loses meaning. The drain here is instead _observed_ from the TPS trend
over time (throughput falls once the server can no longer keep the system
saturated). A more robust definition — the drain begins once the server no longer
has enough in-flight samples to saturate the pipeline — depends on server-side
occupancy that the client cannot see, so it is handwaved for now. Under Poisson,
arrival pacing throttles the pile-up, so the tail is smallest. The windowing
response is therefore mode-specific: concurrency crops the ramp and applies the
guarded tail-cut; offline finds its steady region from the TPS/completion-rate
trend rather than an issue-time window (a client-side tail-cut is not meaningful);
Poisson reuses the concurrency tooling with a single-pass super-pass.

## 5.4 Guarded drain-tail cut

Excluding the tail is safe for per-token latency, latency tails, and throughput —
_conditional on the tail sharing the steady per-token distribution_. This is the
condition that makes it safe to ignore dataset-pass boundaries and drop
high-output samples: the reported per-token latency is unbiased by which samples
are included **iff** inter-token latency is approximately invariant across
samples. With `ITL(S_i) = (t_last(S_i) - t_first(S_i)) / (OSL(S_i) - 1)` the mean
inter-token latency of sample `i` (`OSL - 1` because `OSL` output tokens have
`OSL - 1` inter-token gaps), and population mean `mu`, median `m`, and standard
deviation `sigma` over the samples:

```text
  cut the tail   <=>   sigma / mu <= epsilon   AND   |mu - m| / mu <= delta
```

for small tolerances `epsilon` and `delta`. The first term (low coefficient of variation) is the
necessary-and-sufficient core; the second (low skew) is a robustness guard against
a heavy-tailed distribution and is redundant under low CoV. When the condition
fails — for example a mid-run server anomaly that slows the tail — the tail is
retained and the affected metric is flagged rather than cut. For long-reasoning
workloads the tail is a strong long-output selection, so output-length coverage is
reported alongside the steady metrics so a reader can see the steady set
under-samples the output-length tail.

## 5.5 Convergence: CoV ensemble, trend gate, drift up/down

The window edge and the steady/drift verdict come from two signals combined.

**Metric set.** The convergence and trend tests run on TTFT and TPOT at the 50th,
95th, 99th, and 99.9th percentiles (the 99.9th only where a super-pass holds
enough samples to estimate it). End-to-end latency is reported as context only,
never as a convergence signal (§5.1). A window is steady only when _every_ metric
and percentile plateaus and is within its CoV threshold.

_note: There is some discussion to relax core reported metric percentiles to p95 from p99.
In this case, then the p99 and p99.9 percentiles not converging will be a warning, not a
hard-fail signal._

**CoV stopping rule.** The coefficient of variation `CoV = sigma / mu` of a
metric's per-super-pass percentile, over a trailing window of super-passes, is a
scale-free measure of how much the metric is still moving relative to its own
level. A region is a candidate steady state when `CoV < bound` for every tracked
metric and percentile. The bound loosens toward the tail (a p99 is estimated from
fewer samples per super-pass, so its sampling-noise floor is higher than a p50's).

**Trend gate (mandatory), drift up vs down.** A low CoV over a _trailing_ window
certifies local flatness, not global convergence: a slowly drifting series can
sit locally flat while climbing overall. An ordinary-least-squares trend test over
the whole post-warmup series is therefore applied on top. A metric is _drifting_
when the run-length change is both a large fraction of its typical value and large
relative to the super-pass-to-super-pass residual scatter (so a noisy-but-flat
series is not mistaken for a trend). The slope sign classifies the metric into one
of three states — **Drifting Down**, **Plateau**, **Drifting Up**:

- **Drifting Up** — the metric worsens across the run (for example a p99 TTFT tail
  that never plateaus). This is pathological: there is _no_ steady state to report
  for that metric, and the step says so rather than emitting a number.
- **Plateau** — near-zero slope: the metric is genuinely steady. Only a Plateau
  metric is eligible to contribute a steady value.
- **Drifting Down** — the metric is still settling downward, i.e. the warmup crop
  was slightly short. This is transparent (not a false alarm); the follow-up is a
  larger crop for that metric.

**Ensemble and window selection.** CoV alone is insufficient — a window is only
meaningful if the metric is in Plateau. In practice, over a large set of runs, no
single `(window, bound)` fits all metrics and workloads, so the rule is run as an
**ensemble** of preset `(window, bound)` settings. Among the settings that report
a steady window, the **largest** steady window is selected (most statistically
significant). Detector concordance (agreement on where the run settles) is a
corroborating guardrail; the trend gate remains mandatory and primary, the CoV
ensemble secondary.

## 5.6 Edge cases and error handling

The step never hard-fails a run; every input yields a result plus a `status`.

- **Coverage status.** Before windowing, classify the run by how much steady
  signal it supports and window accordingly:

```text
  status                condition                             behavior
  --------------------  ------------------------------------  -----------------------------
  windowable            >= warmup + 1 full super-passes       normal warmup-cropped window
  insufficient_passes   >= 1 pass, < warmup + 1 super-passes  best-effort, low confidence
  partial_dataset       < 1 full dataset pass                 best-effort, flagged unreliable
```

A batch sweep over many runs then never aborts on a short run: each run yields a
row carrying its status.

- **No steady state.** If a tracked metric is Drifting Up, report drift for that
  metric instead of a point estimate (§5.5). This is a first-class outcome (see the
  open question on invalid runs, §6).
- **Truncated / interrupted log.** If the run did not complete cleanly, the step
  computes a best-effort result over whatever was logged and marks it as partial,
  mirroring how the report already distinguishes interrupted runs.
- **Missing token counts.** If the log carries no per-request output-token count,
  TPOT is derived by tokenizing outputs on the cold path. Not every output is
  tokenized: for large logs a sample sufficient to estimate the per-super-pass
  percentiles is enough, and full re-tokenization of every output is avoided. If
  neither a count nor a tokenizer is available, the step falls back to TTFT-only
  and records that TPOT was not assessed (a metric with no data is skipped, not
  treated as zero, so an absent metric can never fake convergence).
- **TPOT from timestamps.** TPOT derived as `(complete − first_token) / (OSL − 1)`
  assumes the request stayed resident in the decode phase for its whole lifetime.
  If the server evicts or preempts a request mid-decode (paging it out and back),
  the wall-clock span includes queue time that is not per-token decode, inflating
  TPOT. The derivation is trustworthy only when the server does not evict in-flight
  requests during decode; where it might, a server-reported per-token count is
  preferred over the timestamp span.
- **Degenerate modes.** Offline/max-throughput has a degenerate issue time
  (everything issued at `t=0`), so there is no client-side issue-time window and no
  client-side drain boundary. The step detects the mode and finds the steady region
  from the **TPS trend** over time — the plateau before throughput falls off —
  rather than an issue-time window; the exact drain-onset (server occupancy dropping
  below saturation) is server-side and is handwaved for now (§5.3). Near-saturation
  Poisson backlogs like offline and is detected by the completion rate falling below
  the offered rate.

## 5.7 Complementary: staggered ("feathered") issuance

The ramp exists because the target concurrency `C` is filled as a burst. A
staggered fill flattens the ramp-up spike: issue in steps of `ceil(C/k)`, where `k`
is the number of fill steps used to reach the target concurrency (larger `k` = more,
smaller steps), and, between steps, wait until a sample from the previous step has
completed before issuing the next; begin measurements only once `C` is reached. This helps the p99 TTFT
explosion and the initial server hammering. It _reduces the severity_ of the ramp
(a smaller crop is then enough) but does not shorten it, because the server
admission rate, not the client schedule, bounds how fast the pipe fills. It is
therefore
complementary to the warmup crop (timing vs. issue-order), not a replacement, and
under offline/max-throughput it is largely moot (a saturating burst has no steady
baseline). This is a load-generator change, out of scope for the reporting step,
and would require live validation before adoption; it is recorded here so the two
efforts stay aligned.

## 5.8 Output and consumption

The steady-state metrics are reported as the **official result**; the whole-run
(`total`) metrics remain as **supplementary** context (in `report.txt` and the
machine-readable summary produced from the report). Each tracked metric carries
its steady value plus its state (`Plateau` / `Drifting Up` / `Drifting Down`), and
the run carries its coverage `status`. Reported quantities:

- **TPS** (output tokens/s).
- **TTFT and TPOT** percentiles and histograms.
- **QPS is dropped** for text/token-based LLM workloads — a request is not a unit
  of work when output length varies widely, so QPS is a legacy metric of little
  meaning; it is retained only for token-free, uniform-work loads.
- **ISL / OSL** are reported for analytics, not as validation numbers: once the
  window is allowed to drop samples (not enforcing full dataset-pass boundaries),
  the input/output-length distribution is skewed relative to the constructed
  dataset and is no longer a meaningful validation quantity, though it remains
  useful for analysis.

The `total`-vs-`steady_state` divergence is itself surfaced: a large gap indicates
excessive ramp relative to run length, or a run too short to window. Existing
plotting (`src/inference_endpoint/metrics/results_plots.py`,
`scripts/plot_results.py`) can be extended to overlay the window on the
per-super-pass series.

# 6 Open questions {#6-open-questions}

- **Default parameters.** The warmup band, trailing-window length, per-percentile
  CoV bounds, and the guard tolerances (`epsilon`, `delta`) are set from sweeps
  over recorded runs; the defaults should be reviewed on a wider set before they
  are locked as the official reporting parameters.
- **Super-pass sizing for modes without dataset repetition.** For multi-turn
  agentic and other single-pass workloads there is no natural dataset-pass unit.
  Experiments show the steady/drift verdict is robust to the group-size choice, but
  a principled default (for example keyed to concurrency) is still to be picked.
- **Guard distance measure.** The exact two-sample statistic and acceptance bound
  for the per-token-invariance guard (§5.4) need to be fixed.
- **No-steady-state runs.** When no steady state is found (a tracked metric is
  Drifting Up), should the run be reported _invalid_ — analogous to legacy
  LoadGen's statistical-significance gate — or reported with the offending metrics
  flagged as unstable while the rest are reported steady? The current proposal is
  the latter (show which metrics were stable vs unstable).
- **Offline drain-onset.** In offline/max-throughput the drain has no client-side
  boundary; the steady region is read from the TPS trend, and the robust definition
  (server occupancy dropping below saturation) is server-side and currently
  handwaved (§5.3, §5.6). Whether a server-side occupancy signal can be plumbed
  through, and whether the TPS-trend plateau detector lives in the same core or a
  sibling, is open.
- **ISL / OSL reporting basis (task force).** ISL/OSL are reported over the
  _included_ (windowed) samples, which skews them relative to the constructed
  dataset (§5.8). Whether the official artifact should instead report these over
  the full issued set, or carry both, is a policy call deferred to the benchmark
  task force.
- **First pass as warmup (task force).** Whether to treat the first full dataset
  pass (or first super-pass) as warmup by construction — rather than inferring the
  warmup band from the data — and the exact band/window/bound defaults that would
  accompany such a rule are deferred to the benchmark task force.
- **Per-benchmark gates and invalidation (task force).** Whether the steady-state
  gates (CoV bounds, trend thresholds) are tuned per benchmark, and whether a run
  that fails them is declared _invalid_ versus reported-with-flags (see
  "No-steady-state runs" above), is a benchmark-task-force decision, not fixed by
  this proposal.
