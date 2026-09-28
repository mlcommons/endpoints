# Video-generation pareto support — gaps and plan

Status: proposal · Baseline: `4235a9c` · Scope: this repository only.

> **Rules basis.** Requirements below are cited against `endpoints_policies` branch
> `v1.0_rules_dev` @ `6b0b1ef` (2026-09-22). That branch is **unmerged and still moving**; every
> rule citation here is as-of that commit and should be re-checked before being relied on.

The client already drives video-generation endpoints (`videogen/`,
`examples/09_Wan22_VideoGen_Example/`), but every shipped video config is a **single measurement
point**. This records what is missing to run one as a **multi-point pareto sweep**, and itemises
the pending work in §5.

This covers the client slice only. §6 lists what it deliberately excludes — admissibility under
the Endpoints rules, the accuracy target, submitter-side tooling, and the serving stack are all
out of scope here.

## 1. Target shape

Points at fixed concurrency under `ConcurrencyScheduler`, each sustaining a minimum steady-state
window, **plus one Offline point** (§5.7 requires it for every non-agentic benchmark), plus
accuracy runs — 5 for a non-agentic benchmark, one per mandatory point (§5.3).

Two rules shape a video run in particular:

- **§6.4** — total samples issued at a point MUST be a positive integer multiple of the dataset
  size. Runs end on a whole pass, not on a clock.
- **§6.2** — minimum steady-state duration is 600 s in the Ultra Low Concurrency region (1–32)
  and **1200 s** in the Low, Medium, and High Concurrency regions.

Per-point metrics cannot be token-derived — a response is one artifact, not a token stream:

| Token metric | Video-generation substitute |
| --- | --- |
| `system_tps` | `completed_videos / elapsed_seconds` |
| `tps_per_user` (v1.0: `1000 / tpot_p90_ms`) | `1000 / latency_p90_ms` — the same latency-reciprocal shape |
| `ttft_*` | end-to-end issue → complete-response percentile |

The per-user metric mirrors v1.0's definition rather than dividing throughput by concurrency: v1.0
defines per-user rate from the tail latency a user actually experiences, and the video analog
should keep that shape.

**Percentile: P90.** This now *matches* v1.0 rather than diverging from it — §4.1 moved TTFT from
P95 (v0.7) to P90, with an explicit versioning note. The independent reason still holds: a stable
P95 needs roughly twice the completed queries of a P90, and since one video takes tens of seconds
to minutes, a P95 at concurrency 1 would need on the order of 100+ videos per point. Already
available — `DEFAULT_PERCENTILES` carries both
(`async_utils/services/metrics_aggregator/registry.py:411-423`).

**Offline point.** `max_throughput` is the Offline load pattern, so the existing
`examples/09_Wan22_VideoGen_Example/offline_wan22*.yaml` configs are already the right shape for
it. Per §5.7 the reported concurrency is the dataset cardinality, throughput is the only metric of
interest, and latency metrics explicitly do not apply.

## 2. Already works — no change needed

- **System throughput.** `Report.qps` = `n_completed / duration_s`, duration from the
  `tracked_duration_ns` counter (`metrics/report.py:360-361`, `:390-396`). For video that *is*
  videos/second. The legacy LoadGen window is poisson-only
  (`commands/benchmark/pipeline.py:222-225`), so concurrency uses the native window.
- **Per-request latency percentile.** `sample_latency_ns` → `latency`
  (`metrics/report.py:50`, `:234`), P90 and P95 both in the default grid.
- **Per-user rate is already derivable.** `1000 / latency_p90_ms` needs no new measurement — the
  P90 is in the emitted grid. `target_concurrency` also lands in `result_summary.json` via
  `run_config`, so points are self-identifying.
- **`ConcurrencyScheduler`.** Semaphore released on every terminal result, errors included
  (`load_generator/strategy.py`).
- **Phase isolation.** `max_issue_duration_ms` bounds only the performance phase
  (`commands/benchmark/watchdog.py`).
- **Sample order is infinite** — never raises `StopIteration` (`load_generator/sample_order.py`).
- **Accuracy.** VBench path runs; the accuracy phase mirrors the performance load pattern.
- **Audit.** TEST04 already accepts `concurrency`
  (`compliance/audit_test/output_caching_test.py`).
- **Missing tokenizer is handled gracefully**, not fatally (`commands/benchmark/execute.py`).

## 3. Gaps

| # | Gap | Evidence | Kind |
| --- | --- | --- | --- |
| 1 | Stop predicate is a pure OR of stop-requested / count-reached / `max_duration_ns` exceeded. No branch holds off the count stop until a wall-clock floor passes, so a phase is count-driven or capped — never floored. | `load_generator/session.py:866-880` | code |
| 2 | `min_issue_duration_ms` is a poisson count-sizer (`target_qps × duration`), not a floor. Rejected for non-poisson patterns; relaxing that validator alone is insufficient — `total_samples_to_issue()` then raises, since concurrency has no `target_qps`. | `config/schema.py:1021-1027`, `config/runtime_settings.py:249` | code |
| 3 | Steady state cannot certify a tokenless window — see §3.1. | `metrics/steady_state_diagnostics.py` | code |
| 4 | The per-user rate is not emitted as a named field; `Report` carries `qps` and `tps` only. Minor — it is one reciprocal of the already-emitted `latency` P90. | `metrics/report.py:250` | code |
| 5 | Config lock requires `target_concurrency == 1`, so it fails at every point of a curve above 1. `_resolve_model` also raises `KeyError` for a model absent from a ruleset, and the lock expects a `temperature` field the video request type lacks. | `compliance/checker.py:149`, `:115-119` | code |
| 6 | `VideoGenAdapter` mirrors the video *path string* into `response_output`, which the OSL trigger tokenizes. Harmless only because the model name resolves to no tokenizer; supplying one yields meaningless OSL/TPS. | `videogen/adapter.py` | latent |
| 7 | No sweep driver — one invocation is one point, no per-point report-dir convention, no cross-point aggregation. No shipped video config uses concurrency > 1. `publish_submission.py` publishes a single run. | — | tooling |

Gaps 1 and 2 together mean the steady-state window a pareto point needs **cannot currently be
expressed** for a concurrency run.

### 3.1 Why steady state cannot gate a video run

This is the most consequential gap. Under v1.0 §4.4 the detected steady-state window is not a
diagnostic — it is **the official reporting basis**: a point's metrics are computed over that
window, with whole-run values kept only as supplementary. §4.4 names the gating metric as TPOT at
P50 and P90. A workload with no TPOT therefore has no gating metric, and so no official result —
not merely a missing diagnostic section.

See [`steady_state_diagnostics.md`](steady_state_diagnostics.md) for the algorithm; this is only
why it does not apply.

- **The exclusion is explicit, not accidental.** `steady_state_profile`
  (`commands/benchmark/pipeline.py:96-118`) returns `None` — no collection — unless the run has a
  resolved tokenizer *and* streaming, on top of `settings.steady_state.enabled` being set at all
  (it is off by default). A video run fails both conditions by construction.
- The reasoning is sound for its purpose: without streaming there is no `TpotTrigger`, so every
  plateau gate would see an empty TPOT series and every verdict would be `found: false`.
- `tpot_ns` derives from tokens after the first chunk; TPOT is the gated metric, TTFT is
  diagnostic only. `MIN_DUR_FLOOR_S` is already 600 s but never reached.
- Outcome is a `None` verdict, not an error: `Report.steady_state` is `SteadyState | None`
  (`metrics/report.py:289`, `:456`).
- There is no way for a workload to declare a different gating metric — eligibility comes from
  `Profile.supported` (`metrics/steady_state_diagnostics.py:311`) and `SteadyStateConfig`
  (`config/schema.py:947`, `:994`) carries no model allowlist (`:958`).

The tension is between a defensible client design and the v1.0 rule: the client correctly declines
to emit a verdict it cannot compute, while §4.4 makes that verdict the basis of the official
result. Closing it needs a gating metric that exists for tokenless workloads, not a change on
either side alone.

## 4. Workaround available today

Since a run must end on a whole dataset pass (§6.4) *and* meet a duration floor (§6.2), and no
floor setting exists (gaps 1–2), size the count to the duration instead of capping the clock:

1. Calibrate — run the point briefly to estimate sustained throughput at that concurrency.
2. Choose the smallest integer `N` where `N × dataset_size / throughput` comfortably exceeds the
   region's minimum duration.
3. Issue exactly that many samples, with no clock cap:

```yaml
settings:
  runtime:
    n_samples_to_issue: 1240        # N x dataset_size (e.g. 5 x 248) — an exact multiple
  load_pattern:
    type: "concurrency"
    target_concurrency: 1
```

The phase then ends on the count, at a pass boundary, having run at least the required duration —
satisfying both rules without a client patch. `n_samples_to_issue` is returned verbatim and the
sample order is infinite, so no clamping interferes.

- **The duration is achieved, not enforced.** If throughput is lower than calibrated the point
  simply runs longer, which is safe; if higher, it may undershoot the floor, so size `N` with
  margin and check the achieved duration afterwards.
- Every point above concurrency 1 still fails the compliance checker (gap 5).
- Do **not** use `min_issue_duration_ms` for this (gap 2).
- Capping with `max_issue_duration_ms` instead would end the run mid-pass and violate §6.4; it
  remains useful only as a runaway guard set well above the expected finish.

## 5. Pending work

Every code task carries tests — `AGENTS.md` sets >90% coverage and requires an explicit marker.
`videogen/` already has unit and integration suites (`tests/unit/videogen/`,
`tests/integration/videogen/`) to extend rather than start from.

### A. Core runtime

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| A1 | Duration floor: gate the count branch on minimum elapsed time (AND semantics), and accept the setting for `concurrency`. Must land as one change — relaxing the validator alone makes `total_samples_to_issue()` raise. Prefer a new setting name over widening `min_issue_duration_ms`. | 1, 2 | — | S |
| A2 | Emit the per-user rate (`1000 / latency_p90_ms`) as a named field rather than leaving every consumer to derive it. | 4 | — | XS |
| A3 | Modality-aware steady-state gating: gate on an alternative metric where the gated one has no samples, or report un-gated with a reason instead of returning `None`. Requires a way for a workload to declare its gating metric. | 3 | — | M |
| A4 | Make the reported latency percentile explicit (P90, per §4.1) rather than leaving consumers to pick from the grid. | — | A2 | XS |
| A5 | Remove the tokenizer footgun: stop routing a path through `response_output`, or mark the field non-tokenizable for artifact-output adapters. | 6 | — | S |

### B. Compliance and validation

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| B1 | Validate multi-point curves: config lock currently requires `target_concurrency == 1`, and expects a `temperature` field artifact adapters lack. Redesigns what `check_submission` covers. | 5 | A1 | L |
| B2 | Register a video benchmark as a ruleset model and dataset (`config/rulesets/mlcommons/`). Not needed to *run*, but without it there is no golden accuracy, no validity thresholds, and `_resolve_model` raises. | 5 | — | S |

### C. Configuration and tooling

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| C1 | Add concurrency-region example configs sized per §6.4. The shipped configs cover only `max_throughput` (the Offline point) and `concurrency: 1`. | 7 | A1 | XS |
| C2 | Decide whether a multi-point sweep driver belongs here or in submitter tooling, then build or document accordingly. | 7 | — | M |
| C3 | Multi-point publish layout — `publish_submission.py` publishes one run into one scenario directory. | 7 | C2 | M |

### D. Documentation

| # | Task | Depends on |
| --- | --- | --- |
| D1 | Update this document as items land; update `AGENTS.md` if any module moves or is added. | any |
| D2 | Note the modality constraint in `steady_state_diagnostics.md` once A3 settles. | A3 |

**Critical path:** A1 → C1 → (B1, C2). A1 unblocks a legitimate multi-point run; B1 is the only
item that is unavoidable rather than convenient.

## 6. Out of scope for this repository

Listed so the boundary is explicit. None are tracked here.

- Whether a video-generation benchmark is admissible under the Endpoints rules, and any rule text
  it would need — a working-group question.
- The accuracy quality target itself. The scorer exists; the threshold is a benchmark-definition
  decision, not a client one.
- Submitter-side result layout, aggregation, and visualisation of a finished curve.
- The model server and serving stack under test.
- Executing sweeps and producing results.

## 7. Open questions

- Should a sweep driver live here or in submitter tooling? Nothing here orchestrates multiple
  points today.
- Sizing `N` per §6.4 needs a throughput estimate before each point. Should calibration be a
  documented manual step, or something the client can do itself?
- §4.5 introduces provisioned-power normalisation of throughput, marked tentative. Nothing here
  computes it, and the units are not yet settled — worth tracking, not building.
