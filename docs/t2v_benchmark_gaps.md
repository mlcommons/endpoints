# Text-to-video benchmark: closing the client gaps

Status: proposal · Baseline: `4235a9c` · Scope: this repository only.

> **Rules basis.** Requirements below are cited against `endpoints_policies` branch
> `v1.0_rules_dev` @ `6b0b1ef` (2026-09-22). That branch is **unmerged and still moving**; every
> rule citation here is as-of that commit and should be re-checked before being relied on.

Running a text-to-video workload as an Endpoints submission needs a **pareto curve**: several
measurement points at different concurrencies, each sustaining a minimum steady-state window, plus
an Offline point and accuracy runs. Two things stand in the way. The client measures **one point
per invocation**, and a video response is a **single artifact rather than a token stream**, which
the metric and steady-state machinery assumes throughout.

This records the distance between what the client does today and what a text-to-video submission
needs, and itemises the work in §5. The adapter itself (`videogen/`) and the example workload
(`examples/09_Wan22_VideoGen_Example/`) already exist; what is missing is everything around them.

Three of the eight gaps are specific to a tokenless workload, two of them detailed in §3.1 and
§3.2. The other five block a multi-point curve for **any** model and are marked `any model` in
the gap table, so fixing them unblocks video and every other benchmark at once. Existing
text-model curves are produced by running each concurrency separately and stitching the results
downstream, which is the workaround for exactly those five.

This covers the client slice only. §6 lists what it deliberately excludes.

## 0. What is missing

| Category | Missing |
| --- | --- |
| [Naming](#a-naming) | A ruleset model and dataset entry, so the tooling recognises the benchmark at all |
| [Missing metrics](#b-missing-metrics) | The per-user rate as a named field |
| [Duration support](#c-duration-support) | A floor, so a point can meet a minimum steady-state window |
| [Multi-point support](#d-multi-point-support) | A sweep driver, and a publish layout for more than one point |
| [Validation support](#e-validation-support) | Checking a curve rather than a single single-stream point |
| [Tokenless support](#f-tokenless-support) | Steady-state gating without TPOT, and artifact-safe responses |
| [Example configs](#g-example-configs) | Concurrency-region configs sized to whole dataset passes |

Nothing is missing on the client side of the request path: the adapter and the example workload
already run. The gaps
are in measuring a curve, proving it valid, and describing the benchmark to the tooling. §3 and §5
follow the same order as this table.

## 1. Target shape

Per v1.0 §5.3 and §5.7, a non-agentic submission is four mandatory concurrency points (Ultra Low
1-32, Low, Medium, High), three submitter's-choice points, one Offline point, and five accuracy
runs.

Two rules govern how a single point must run:

- **§6.4**: total samples issued at a point MUST be a positive integer multiple of the dataset
  size. A point ends on a whole pass, not on a clock.
- **§6.2**: minimum steady-state duration is 600 s in the Ultra Low Concurrency region and
  **1200 s** in the Low, Medium, and High regions.

Neither is expressible today for a concurrency-scheduled run: see gaps 3 and 4.

### 1.1 Metrics for video generation

For video generation the per-point metrics cannot be token-derived, because a response is one
artifact rather than a token stream:

| Token metric | Substitute |
| --- | --- |
| `system_tps` | `completed_videos / elapsed_seconds` |
| `tps_per_user` (v1.0: `1000 / tpot_p90_ms`) | `1000 / latency_p90_ms`, the same latency-reciprocal shape |
| `ttft_*` | end-to-end issue to complete-response percentile |

The per-user metric mirrors v1.0's definition rather than dividing throughput by concurrency,
because v1.0 derives per-user rate from the tail latency a user actually experiences.

**Percentile: P90.** This *matches* v1.0 rather than diverging from it, since §4.1 moved TTFT from
P95 to P90 with an explicit versioning note. The independent reason also holds: a stable P95 needs
roughly twice the completed queries of a P90, and one video takes tens of seconds to minutes, so a
P95 at concurrency 1 would need on the order of 100+ videos per point. `DEFAULT_PERCENTILES`
already carries both (`async_utils/services/metrics_aggregator/registry.py:411-423`).

**Offline point.** `max_throughput` is the Offline load pattern, so the existing
`examples/09_Wan22_VideoGen_Example/offline_wan22*.yaml` configs are already the right shape. Per
§5.7 the reported concurrency is the dataset cardinality, throughput is the only metric of
interest, and latency metrics explicitly do not apply.

## 2. Already works, no change needed

- **System throughput.** `Report.qps` = `n_completed / duration_s`, duration from the
  `tracked_duration_ns` counter (`metrics/report.py:360-361`, `:390-396`). For a video workload
  that *is* videos/second. The legacy LoadGen window is poisson-only
  (`commands/benchmark/pipeline.py:222-225`), so concurrency uses the native window.
- **Per-request latency percentile.** `sample_latency_ns` maps to `latency`
  (`metrics/report.py:50`, `:234`), with P90 and P95 both in the default grid.
- **`ConcurrencyScheduler`.** Semaphore released on every terminal result, errors included
  (`load_generator/strategy.py`).
- **Phase isolation.** `max_issue_duration_ms` bounds only the performance phase
  (`commands/benchmark/watchdog.py`).
- **Sample order is infinite**, never raising `StopIteration` (`load_generator/sample_order.py`).
- **Point identity.** `target_concurrency` lands in `result_summary.json` via `run_config`.
- **Audit.** TEST04 already accepts `concurrency`
  (`compliance/audit_test/output_caching_test.py`).
- **Missing tokenizer is handled gracefully**, not fatally (`commands/benchmark/execute.py`).

## 3. Gaps

Ordered by the categories in §0. `Scope` distinguishes gaps that block any model from those
specific to a tokenless workload.

| # | Gap | Evidence | Scope | Kind |
| --- | --- | --- | --- | --- |
| 1 | No ruleset entry exists for the benchmark, so `_resolve_model` raises `KeyError` before any other check runs. Without one there is also no golden accuracy and no validity thresholds. | `compliance/checker.py:156-164` | tokenless | code |
| 2 | The per-user rate is not emitted as a named field; `Report` carries `qps` and `tps` only. Minor, since it is one reciprocal of the already-emitted `latency` P90. | `metrics/report.py:250` | any model | code |
| 3 | Stop predicate is a pure OR of stop-requested / count-reached / `max_duration_ns` exceeded. No branch holds off the count stop until a wall-clock floor passes, so a phase is count-driven or capped, never floored. | `load_generator/session.py:866-880` | any model | code |
| 4 | `min_issue_duration_ms` is a poisson count-sizer (`target_qps × duration`), not a floor. Rejected for non-poisson patterns, and relaxing that validator alone is insufficient because `total_samples_to_issue()` then raises, since concurrency has no `target_qps`. | `config/schema.py:1021-1027`, `config/runtime_settings.py:249` | any model | code |
| 5 | No sweep driver. One invocation is one point, with no per-point report-dir convention and no cross-point aggregation. `publish_submission.py` publishes a single run into a single scenario directory. | n/a | any model | tooling |
| 6 | Config lock requires `target_concurrency == 1`, so it fails at every point of a curve above 1, and expects a `temperature` field artifact adapters do not carry. | `compliance/checker.py:149`, `:115-119` | any model | code |
| 7 | Steady state cannot certify a tokenless window. See §3.1. | `metrics/steady_state_diagnostics.py` | tokenless | code |
| 8 | `VideoGenAdapter` mirrors the video *path string* into `response_output`, which the OSL trigger tokenizes. Harmless only because the model name resolves to no tokenizer; supplying one yields meaningless OSL and TPS. See §3.2. | `videogen/adapter.py` | tokenless | latent |

Gaps 3 and 4 together mean the steady-state window a pareto point needs **cannot currently be
expressed** for a concurrency run, for any model. Gap 6 means that even a correctly measured curve
cannot be validated by this repository's checker.

### 3.1 Why steady state cannot gate a tokenless run

This is the most consequential of the tokenless gaps. Under v1.0 §4.4 the detected steady-state
window is not a diagnostic. It is **the official reporting basis**: a point's metrics are computed
over that window, with whole-run values kept only as supplementary. §4.4 names the gating metric
as TPOT at P50 and P90. A workload with no TPOT therefore has no gating metric, and so no official
result.

See [`steady_state_diagnostics.md`](steady_state_diagnostics.md) for the algorithm; this is only
why it does not apply.

- **The exclusion is explicit, not accidental.** `steady_state_profile`
  (`commands/benchmark/pipeline.py:96-118`) returns `None`, meaning no collection, unless the run
  has a resolved tokenizer *and* streaming, on top of `settings.steady_state.enabled` being set at
  all (it is off by default). A tokenless run fails both conditions by construction.
- The reasoning is sound for its purpose: without streaming there is no `TpotTrigger`, so every
  plateau gate would see an empty TPOT series and every verdict would be `found: false`.
- Outcome is a `None` verdict, not an error: `Report.steady_state` is `SteadyState | None`
  (`metrics/report.py:289`, `:456`).
- There is no way for a workload to declare a different gating metric. Eligibility comes from
  `Profile.supported` (`metrics/steady_state_diagnostics.py:311`), and `SteadyStateConfig`
  (`config/schema.py:947`, `:994`) carries no model allowlist (`:958`).

The tension is between a defensible client design and the v1.0 rule. The client correctly declines
to emit a verdict it cannot compute, while §4.4 makes that verdict the basis of the official
result. Closing it needs a gating metric that exists for tokenless workloads, not a change on
either side alone.

### 3.2 Tokenizer footgun for artifact responses

`VideoGenAdapter` puts the response's file path into `response_output`, which is the field the
output-sequence-length trigger tokenizes. Nothing breaks today only because the model name does
not resolve to a tokenizer, so token metrics stay disabled. Supplying `--tokenizer` would silently
produce OSL and TPS figures computed from a filesystem path.

## 4. Workaround available today

Since a point must end on a whole dataset pass (§6.4) *and* meet a duration floor (§6.2), and no
floor setting exists (gaps 3 and 4), size the count to the duration instead of capping the clock:

1. Calibrate. Run the point briefly to estimate sustained throughput at that concurrency.
2. Choose the smallest integer `N` where `N × dataset_size / throughput` comfortably exceeds the
   region's minimum duration.
3. Issue exactly that many samples, with no clock cap:

```yaml
settings:
  runtime:
    n_samples_to_issue: 1240        # N x dataset_size (e.g. 5 x 248), an exact multiple
  load_pattern:
    type: "concurrency"
    target_concurrency: 1
```

The phase then ends on the count, at a pass boundary, having run at least the required duration,
satisfying both rules without a client patch. `n_samples_to_issue` is returned verbatim and the
sample order is infinite, so nothing clamps it.

- **The duration is achieved, not enforced.** If throughput is lower than calibrated the point
  simply runs longer, which is safe. If higher, it may undershoot the floor, so size `N` with
  margin and check the achieved duration afterwards.
- Every point above concurrency 1 still fails the compliance checker (gap 6).
- Do **not** use `min_issue_duration_ms` for this (gap 4).
- Capping with `max_issue_duration_ms` instead would end the run mid-pass and violate §6.4. It
  remains useful only as a runaway guard set well above the expected finish.

## 5. Pending work

Grouped by the categories in §0. Every code task carries tests; `AGENTS.md` sets >90% coverage and
requires an explicit marker.

### A. Naming

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| A1 | Register the benchmark as a ruleset model and dataset (`config/rulesets/mlcommons/`). Not needed to *run*, but without it there is no golden accuracy, no validity thresholds, and `_resolve_model` raises. | 1 | n/a | S |

### B. Missing metrics

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| B1 | Emit the per-user rate as a named field rather than leaving every consumer to derive it. | 2 | n/a | XS |
| B2 | Make the reported latency percentile explicit (P90, per §4.1). | n/a | B1 | XS |

### C. Duration support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| C1 | Duration floor: gate the count branch on minimum elapsed time (AND semantics), and accept the setting for `concurrency`. Must land as one change, because relaxing the validator alone makes `total_samples_to_issue()` raise. Prefer a new setting name over widening `min_issue_duration_ms`. | 3, 4 | n/a | S |

### D. Multi-point support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| D1 | Decide whether a multi-point sweep driver belongs here or in submitter tooling, then build or document accordingly. | 5 | n/a | M |
| D2 | Multi-point publish layout for `publish_submission.py`. | 5 | D1 | M |

### E. Validation support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| E1 | Validate multi-point curves. Redesigns what `check_submission` covers, since the config lock is built around a single single-stream point. | 6 | C1 | L |

### F. Tokenless support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| F1 | Modality-aware steady-state gating: gate on an alternative metric where the gated one has no samples (end-to-end latency exists for every workload), or report un-gated with a reason instead of returning `None`. Requires a way for a workload to declare its gating metric. | 7 | n/a | M |
| F2 | Remove the tokenizer footgun: stop routing a path through `response_output`, or mark the field non-tokenizable for artifact-output adapters. | 8 | n/a | S |

### G. Example configs

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| G1 | Add concurrency-region example configs sized per §6.4. The shipped video configs cover only `max_throughput` (the Offline point) and `concurrency: 1`. | 5 | C1 | XS |

### H. Documentation

| # | Task | Depends on |
| --- | --- | --- |
| H1 | Update this document as items land; update `AGENTS.md` if any module moves or is added. | any |
| H2 | Note the modality constraint in `steady_state_diagnostics.md` once F1 settles. | F1 |

**Critical path:** C1 → G1 → (E1, D1). C1 unblocks a legitimate multi-point run for every model.
E1 is the only item that is unavoidable rather than convenient.

## 6. Out of scope for this repository

- The accuracy quality target. The scorer exists; the threshold is a benchmark-definition
  decision, not a client one.
- Submitter-side result layout, aggregation, and visualisation of a finished curve.
- The model server and serving stack under test.
- Executing sweeps and producing results.
