# Text-to-video benchmark: closing the client gaps

Status: proposal · Baseline: `4235a9c` · Scope: this repository only.

> **Rules basis.** Requirements below are cited against `endpoints_policies` branch
> `v1.0_rules_dev` @ `cdb203c` (2026-09-30). That branch is **unmerged and still moving**; every
> rule citation here is as-of that commit and should be re-checked before being relied on.

An Endpoints submission needs a **pareto curve**: several runs at different concurrency levels,
plus 1 Offline run and a set of accuracy runs. Each run has to hold steady for a minimum time.

**Blockers:**

- The client measures **1 point per run**. There is no way to sweep several.
- A video reply is **1 file, not a stream of tokens**. The metric and steady-state code assume
  tokens everywhere.

**What already exists:**

- The adapter, `videogen/`.
- The example workload, `examples/09_Wan22_VideoGen_Example/`.

Everything around them is what is missing, and §5 lists that work.

**The gaps:**

- 4 gaps (§3), plus example configs.
- 1 is a hard blocker: steady state cannot certify a window without tokens (§3.1).
- 1 is a **rules** conflict, not a client change: §6.5 mandates `stream_all_chunks = true`, which
  a non-streaming workload cannot set. It needs an amendment.
- The client also has limits that affect every model, not just video. They do **not** block a
  T2V benchmark, because the same workaround text curves already use works here too. See §3.3.

**Scope:** the client only. §6 lists what is left out.

## 0. What is missing

| Category | Missing |
| --- | --- |
| [Naming](#a-naming) | A ruleset model and dataset entry, so the tooling recognises the benchmark at all |
| [Tokenless support](#b-tokenless-support) | Steady-state gating without TPOT, and artifact-safe responses |
| [Example configs](#c-example-configs) | Concurrency-region configs sized to whole dataset passes |
| Rules amendment | An exemption from §6.5's `stream_all_chunks = true`. Not a client task; recorded so the gap list is complete |

The adapter and the example workload already run. What is missing is the description of the
benchmark to the tooling, and the handling of a reply that carries no tokens.

## 1. Target shape

Per v1.0 §5.3 and §5.7, a non-agentic submission is 4 mandatory concurrency points (Ultra Low
1-32, Low, Medium, High), 3 submitter's-choice points, 1 Offline point, and 5 accuracy
runs.

2 rules govern how a single point must run:

- **§6.4**: total samples issued at a point MUST be a positive integer multiple of the dataset
  size. A point ends on a whole pass, not on a clock. §6.4 separately requires a minimum number
  of *completed* queries, one pass over the dataset.
- **§6.2**: minimum steady-state duration is 600 s in the Ultra Low Concurrency region and
  **1200 s** in the Low, Medium, and High regions.
- **§4.4**: the steady window must span **more than 4 super-passes** (`MIN_TREND_N = 4`), and
  §6.2's minimum is measured over that window's **issue-time span**, not wall-clock. The effective
  floor is `max(4 super-passes, §6.2 minimum)`.

§6.4's issued-count rule **is** expressible today, and §4 shows how. What cannot be expressed is a
duration floor (§3.3), and what cannot be *certified* is §6.2 or §4.4, because neither is
computable without a steady window (§3.1).

### 1.1 Metrics for video generation

For video generation the per-point metrics cannot be token-derived, because a response is 1
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
roughly twice the completed queries of a P90, and 1 video takes tens of seconds to minutes, so a
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
- **Sample order is infinite**, never raising `StopIteration` (`load_generator/sample_order.py`).
- **Audit.** TEST04 already accepts `concurrency`
  (`compliance/audit_test/output_caching_test.py`).
- **Missing tokenizer is handled gracefully**, not fatally (`commands/benchmark/execute.py`).

## 3. Gaps

Ordered by the categories in §0.

| # | Gap | Evidence | Kind |
| --- | --- | --- | --- |
| 1 | No ruleset entry exists for the benchmark, so `_resolve_model` raises `KeyError` before any other check runs. Without one there is also no golden accuracy and no validity thresholds. | `compliance/checker.py:156-164` | code |
| 2 | Steady state cannot certify a window with no tokens. See §3.1. **This is the only hard blocker.** | `metrics/steady_state_diagnostics.py` | code |
| 3 | `VideoGenAdapter` mirrors the video *path string* into `response_output`, which the OSL trigger tokenizes. Harmless only because the model name resolves to no tokenizer; supplying one yields meaningless OSL and TPS. See §3.2. | `videogen/adapter.py` | latent |
| 4 | §6.5 requires `stream_all_chunks = true` for **all** performance runs. Every video config sets `streaming: "off"`, and `VideoGenAdapter.decode_sse_message` raises by design. The rule text forbids the only configuration the workload can run, so this needs a rules amendment, not a client change. | `videogen/adapter.py`, §6.5 | rules |

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
  plateau gate would see an empty TPOT series and nothing would ever validate.
- Outcome is a `None` result, not an error: `Report.steady_state` is `SteadyState | None`
  (`metrics/report.py:289`, `:456`).
- There is no way for a workload to declare a different gating metric. Eligibility comes from
  `Profile.supported` (`metrics/steady_state_diagnostics.py:311`), and `SteadyStateConfig`
  (`config/schema.py:947`, `:994`) carries no model allowlist (`:958`).

The tension is between a defensible client design and the v1.0 rule. The client correctly declines
to emit a result it cannot compute, while §4.4 makes that result the basis of the official
result. Closing it needs a gating metric that exists for tokenless workloads, not a change on
either side alone.

### 3.2 Tokenizer footgun for artifact responses

`VideoGenAdapter` puts the response's file path into `response_output`, which is the field the
output-sequence-length trigger tokenizes. Nothing breaks today only because the model name does
not resolve to a tokenizer, so token metrics stay disabled. Supplying `--tokenizer` would silently
produce OSL and TPS figures computed from a filesystem path.

### 3.3 Client limits that are not T2V problems

Working out how to run a sweep surfaced several limits in the client. None of them block a T2V
benchmark, and they are recorded here only so the §4 recipe makes sense.

- There is no minimum-duration floor in the run loop, and `min_issue_duration_ms` is a poisson
  sample-count sizer rather than a floor, rejected for `concurrency`
  (`load_generator/session.py:866-880`, `config/schema.py:1021-1027`,
  `config/runtime_settings.py:249`). The §4 recipe meets the floor without either.
- One invocation measures one point. There is no sweep driver and no multi-point publish layout.
  Existing text curves are built by running each concurrency on its own and joining the results
  afterwards; the same applies here.
- The per-user rate is not emitted as a named field, but it is the reciprocal of the latency P90
  already reported inside the `latency` rollup (`metrics/report.py:234`), so it is one division in
  post-processing. It is *not* derived from `qps` (`:250`); §1.1 defines it from tail latency.
- `compliance/checker.py` requires `target_concurrency == 1`, but it is scoped to Edge-Agentic
  (BFCL v4) submissions and is reachable only through `scripts/check_compliance.py`. It is not the
  validator for an Endpoints pareto submission, so it is not on this path.

Fixing any of these would help every benchmark, not just video. That is a separate argument and
not a prerequisite here.

## 4. Workaround available today

A point must end on a whole dataset pass (§6.4), span more than 4 super-passes (§4.4), and meet a
duration floor (§6.2). No floor setting exists (§3.3), so size the count instead of capping the
clock:

1. Calibrate. Run the point briefly to estimate sustained throughput at that concurrency.
2. Choose `N = max(5, ceil(min_duration x throughput / dataset_size))`. The **5** is §4.4's
   trend-test floor: a window needs more than 4 super-passes, and one super-pass is one dataset
   pass by default. Sizing on duration alone gives `N = 1` at low concurrency, which §4.4 classes
   as `insufficient_passes` and does **not** count as an official result.
3. Issue exactly that many samples, with no clock cap:

```yaml
settings:
  runtime:
    n_samples_to_issue: 1240        # N x dataset_size (5 x 248), an exact multiple
    sample_order: with_replacement  # sec 6.5 requires WithReplacementSampleOrder for perf runs
  load_pattern:
    type: "concurrency"
    target_concurrency: 1
```

The phase ends on the count, at a pass boundary. `n_samples_to_issue` is returned verbatim and the
sample order is infinite, so nothing clamps it.

**What this does and does not satisfy.** It satisfies §6.4's issued-count rule outright. It makes
the run long enough in *wall-clock* to clear §6.2, which is necessary but **not sufficient**: §6.2
is measured over the steady window's issue-time span, after warmup and ramp-crop and before the
drain, and §4.4 gates on that window spanning more than 4 super-passes. Neither can be certified
for this workload until gap 2 is closed, because there is no window to measure (§3.1). Size for
them anyway, so the data qualifies once it can be checked.

- **The duration is achieved, not enforced.** If throughput is lower than calibrated the point
  runs longer, which is safe. If higher, it may undershoot, so size `N` with margin and check the
  achieved duration afterwards.
- **Termination counts issued, not completed** (`load_generator/session.py:868-873`), while §6.4's
  minimum-query rule counts *completed*. A point with request failures issues exactly
  `N x dataset_size` yet can complete fewer, and finishes early because failures return fast.
  Check `n_samples_completed`, not just elapsed time.
- Do **not** use `min_issue_duration_ms` for this (§3.3).
- Capping with `max_issue_duration_ms` would end the run mid-pass and violate §6.4. It is useful
  only as a runaway guard set well above the expected finish.

## 5. Pending work

Grouped by the categories in §0. Every code task carries tests; `AGENTS.md` sets >90% coverage and
requires an explicit marker.

### A. Naming

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| A1 | Register the benchmark as a ruleset model and dataset (`config/rulesets/mlcommons/`). Not needed to *run*, but without it there is no golden accuracy, no validity thresholds, and `_resolve_model` raises. | 1 | n/a | S |

### B. Tokenless support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| B1 | Modality-aware steady-state gating: gate on an alternative metric where the gated one has no samples (end-to-end latency exists for every workload), or report un-gated with a reason instead of returning `None`. Requires a way for a workload to declare its gating metric. | 2 | n/a | M |
| B2 | Remove the tokenizer footgun: stop routing a path through `response_output`, or mark the field non-tokenizable for artifact-output adapters. | 3 | n/a | S |

### C. Example configs

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| C1 | Add concurrency-region example configs sized per §6.4, using the §4 recipe. The shipped video configs cover only `max_throughput` (the Offline point) and `concurrency: 1`. | n/a | n/a | XS |

### D. Documentation

| # | Task | Depends on |
| --- | --- | --- |
| D1 | Update this document as items land; update `AGENTS.md` if any module moves or is added. | any |
| D2 | Note the modality constraint in `steady_state_diagnostics.md` once B1 settles. | B1 |

**B1 is the critical item.** Under v1.0 §4.4 the steady-state result is the official reporting
basis, so without it a T2V run has no official result. A1 and C1 are small and independent.

## 6. Out of scope for this repository

- The accuracy quality target. The scorer exists; the threshold is a benchmark-definition
  decision, not a client one.
- Submitter-side result layout, aggregation, and visualisation of a finished curve.
- The model server and serving stack under test.
- Executing sweeps and producing results.
