# Text-to-video benchmark: closing the client gaps

Status: proposal · Baseline: `4235a9c` · Scope: this repository only.

> **Rules basis.** Requirements below cite the MLPerf Endpoints rules on the `endpoints_policies`
> branch `v1.0_rules_dev` @ `cdb203c` (2026-09-30):
> [`endpoints_rules.md`](https://github.com/mlcommons/endpoints_policies/blob/cdb203c/endpoints_rules.md)
> and
> [`endpoints_submission_rules.md`](https://github.com/mlcommons/endpoints_policies/blob/cdb203c/endpoints_submission_rules.md).
> That branch is unmerged; re-check every rule citation before relying on it.

An Endpoints submission is a **pareto curve**: runs at several concurrency levels, an Offline
result, and accuracy runs. This document covers what the client must change so a text-to-video
(T2V) workload, Wan2.2 with 248 prompts and 1 video file per response, can produce valid points.

**What already exists:**

- The adapter, `videogen/`, and the example workload, `examples/09_Wan22_VideoGen_Example/`.
- System throughput and a P90 latency percentile, both emitted natively (§2).

**What blocks a valid point:**

- Perf runs cannot select with-replacement sample order, which §6.5 requires (gap 4). Small.
- The shipped example configs are not submission-shaped (gap 5). Small.

**What does not block a valid point under the rules as written:**

- Steady state cannot gate a tokenless run (gap 2). The point falls back to the whole-run `total`
  metrics, which §4.4 names "the official fallback where no steady state holds". This becomes a
  blocker only if the working group (WG) rules a `not found` run invalid, which §4.4 lists as
  pending ratification.
- `stream_all_chunks = true` (§6.5) is a plain client flag that a video config can set (§3.4).
  A pending rules change (`arekay/streaming_chunks_client_false`, `d9e1ea6`) would instead require
  `model_params.streaming` to resolve to `on` for every fixed-concurrency perf run. The video
  adapter cannot stream, so that would block every T2V point until the rule exempts
  non-streaming classes.

**Not client work:** the §9.1 metric consistency check requires a TPOT distribution and rejects a
submission at Week 0. That needs WG rule text, tracked in the NVIDIA readiness plan.

## 0. What is missing

| Category | Missing |
| --- | --- |
| [Naming](#a-naming) | A dataset entry and 1 model slug shared with the results repository |
| [Tokenless support](#b-tokenless-support) | Artifact-safe tokenizer handling; steady-state gating without TPOT, if the WG requires it |
| [Run settings](#c-run-settings) | A config setting for with-replacement sample order on perf runs |
| [Example configs](#d-example-configs) | Point configs sized to whole passes, with seeds bound and no binding time cap |
| Rules amendments | Video forms of §9.1 metric consistency, §4.1 metrics, and §4.4 gating; a non-streaming exemption if the pending streaming change merges. Not client tasks |

## 1. Target shape

Per v1.0 §5.3 and §5.7.2, a non-agentic submission with an **elected** Offline result is **7
runs**: 1 Ultra Low point, 3 mandatory points (Low, Medium, High), and 3 submitter's-choice points,
with the `C_max` point also reported as the Offline result. Accuracy is required at the 4 mandatory
points plus Offline; the elected `C_max` point fills both roles, so the plan uses **4 accuracy
runs**. A pending Rules TF patch (`nvashutoshd_v1.0_rules_patch`, `b36c220`) makes `N` = 5 an
exact count checked automatically, without saying how an elected point counts; if it needs 2
results, the plan needs 5 runs. Choice points must not carry accuracy results under that patch.

The planned point set, from the readiness plan: `C_min = 1`, `C_max = 72`, 1 x 72-GPU layout,
points at concurrency 1, 2, 4, 8, 12, 36, 72; accuracy at 1, 4, 12, 72.

3 rules govern how a single point runs:

- **§6.4**: total samples issued MUST be a positive integer multiple of the dataset size, and at
  least 1 full pass must *complete*.
- **§6.2**: minimum duration is 600 s in Ultra Low and 1200 s in Low, Medium, and High, measured
  on the steady window's issue-time span when a window exists.
- **§4.4**: a steady-state result needs a window of **≥ 4 super-passes**, so a run of **more than
  4**, and a gating metric that is a Plateau. Otherwise the point reports `total`.

### 1.1 Metrics for video generation

Proposed, pending rule text; names match the results-repository plan:

| Token metric | Video metric | From the client report |
| --- | --- | --- |
| `system_tps` | `system_qps` | `qps` |
| `tps_per_user` (v1.0: `1000 / tpot_p90_ms`) | `qps_per_user` = `1000 / latency_p90_ms` | derived |
| `ttft_p90_ms` | `latency_p90_ms`, issue to complete response | `latency.percentiles["90.0"] / 1e6` (reported in ns) |

The per-user metric mirrors v1.0's latency-reciprocal definition rather than dividing throughput by
concurrency. **P90** aligns with v1.0 §4.1, which moved TTFT from P95 to P90. `DEFAULT_PERCENTILES`
carries both (`async_utils/services/metrics_aggregator/registry.py:411-423`).

**Offline.** The elected `C_max` point keeps its latency metrics (§5.7.2). No `max_throughput` run
is needed.

## 2. Already works, no change needed

- **System throughput.** `Report.qps` = `n_completed / duration_s` (`metrics/report.py:360-361`,
  `:390-396`). For a video workload that is videos/second on the `total` basis. A steady-window
  videos/s would need CL-B1.
- **Per-request latency percentile.** `sample_latency_ns` maps to `latency` (`metrics/report.py:50`,
  `:234`), with P90 and P95 in the default grid.
- **Sample order is infinite**, never raising `StopIteration` (`load_generator/sample_order.py`).
- **Audit.** TEST04 accepts `concurrency` (`compliance/audit_test/output_caching_test.py`).
- **Missing tokenizer is handled.** `_check_tokenizer_exists` returns `False` with a warning for a
  name that is not a tokenizer (`commands/benchmark/execute.py:215-259`), so token metrics stay off.
- **`stream_all_chunks`** is settable for any workload (§3.4).

## 3. Gaps

Ordered by the categories in §0.

| # | Gap | Evidence | Kind |
| --- | --- | --- | --- |
| 1 | No dataset entry or agreed slug. The Endpoints round ruleset deliberately has no per-model rulesets (`benchmark_rulesets={}`), so a model entry has no consumer; the canonical dataset and the slug do. | `config/rulesets/mlcommons/rules.py:364-367`, `datasets.py` | naming |
| 2 | Steady state cannot gate a tokenless run, so the point reports `total`. Blocks a valid point only if the WG rules `not found` invalid. See §3.1. | `commands/benchmark/pipeline.py:96-138`, `metrics/steady_state_diagnostics.py` | code, conditional |
| 3 | `VideoGenAdapter` mirrors the video path into `response_output`, which the OSL trigger tokenizes if a tokenizer is supplied. See §3.2. | `videogen/adapter.py:123-127` | latent |
| 4 | §6.5 requires `WithReplacementSampleOrder` for perf runs, but no config field selects it. See §3.3. | `config/runtime_settings.py:174`, `config/schema.py:593-600` | code |
| 5 | The example configs are not submission-shaped: `offline_wan22_submission.yaml` issues a 144-prompt subset; `offline_wan22.yaml` and `offline_wan22_accuracy.yaml` cap at 10 min where 1 pass takes about 24 min; `single_stream_wan22_submission.yaml` issues 20; none sets `submission_ref`, so seeds stay at the defaults. | `examples/09_Wan22_VideoGen_Example/` | config |

### 3.1 Steady state for a tokenless run

- **The exclusion is explicit.** `steady_state_profile` (`commands/benchmark/pipeline.py:96-138`)
  returns `None` unless the run has a resolved tokenizer *and* streaming, on top of
  `settings.steady_state.enabled` (off by default).
- **The rules fall back, they do not void.** §4.4 gives `total` as the official result for
  `insufficient_passes` and `not found`. Pending ratification: whether `not found` is *invalid* or
  *reported-with-flags*.
- **What CL-B1 would take**, if the WG requires a window:
  1. A latency-gated profile in `PROFILES` (`steady_state_diagnostics.py:303-315`) and a way for
     the workload to select it; the profile crosses to the aggregator subprocess as
     `--steady-state-profile`.
  2. Relax the tokenizer and streaming refusals in `pipeline.steady_state_profile` for that profile.
     The collector already records latency without a first chunk.
  3. Generalise TPOT-only logic: anomaly detection, the minimum-duration precision term, drift
     watch, and the token-based `TpsBlock` (`:163`). Add a window videos/s.
  4. The standalone post-processing CLI that §4.4 names requires a tokenizer (`:2172`).
  5. It depends on WG text naming the tokenless gating metric. The rules disagree with themselves
     today: the definitions table says TPOT P50/P90, the operative paragraph says TTFT and TPOT.

### 3.2 Tokenizer footgun for artifact responses

The path in `response_output` is what VBench scores: the adapter mirrors it "so the event log
carries it to the accuracy scorer" (`videogen/adapter.py:123-124`). Removing it breaks accuracy.
The safe fix is to never tokenize an artifact-output adapter: force `tokenizer_name=None` for it in
`setup_benchmark` (`commands/benchmark/execute.py:516-525`), or reject `--tokenizer`.

### 3.3 Sample order cannot be set for perf runs

§6.5 (`endpoints_rules.md:1228-1229`) requires `WithReplacementSampleOrder` for performance runs
and `WithoutReplacementSampleOrder` for accuracy runs. The client implements both
(`load_generator/sample_order.py`), but the config cannot choose:

- `RuntimeConfig` (`config/schema.py:593`) has no `sample_order` field and sets `extra="forbid"`
  (`:600`), so a `sample_order:` key in YAML is a validation error.
- `RuntimeSettings.from_config` sets `"sample_order": SampleOrderSpec()`
  (`config/runtime_settings.py:174`), which is `WITHOUT_REPLACEMENT`.
- Only the TEST04 audit overrides it.

Every perf run today samples without replacement. This affects every benchmark, but a T2V perf
point cannot be valid until it is fixed. With replacement, a super-pass is 248 random draws rather
than every prompt once; for fixed-shape video output that changes little.

### 3.4 Client limits that are not T2V problems

None of these block a T2V benchmark; they explain the §4 recipe.

- **No duration floor.** `min_issue_duration_ms` sizes poisson runs and is rejected for
  `concurrency` (`load_generator/session.py:866-880`, `config/schema.py:1021-1027`). The recipe
  meets the floor by count.
- **1 invocation measures 1 point.** Curves are built by running each concurrency separately.
- **Failed requests count as completed.** `n_samples_completed` includes failures
  (`metrics/report.py:215`), and `qps` divides by it (`:396`), so a point with fast failures
  reports inflated `qps` and deflated latency. Check `n_samples_failed`.
- **`stream_all_chunks`** is a plain `settings.client` flag (`endpoint_client/config.py:186`), read
  only on the SSE path (`endpoint_client/worker.py:429`); nothing ties it to `streaming`. Setting
  it in a video config meets §6.5 and the §9.1 streaming check as written at `cdb203c`; see the
  pending change noted at the top of this document.
- **`compliance/checker.py`** is the Edge-Agentic (BFCL v4) checker, reachable only through
  `scripts/check_compliance.py`. It is not the Endpoints validator.

## 4. Running a point today

1. **Size `N`, the number of dataset passes.** `N = 1` meets §6.4 and, at the measured 5.7-7.6 s
   per video, §6.2 (1 pass is 1,414 s at 5.7 s; the 1200 s floor binds only below 4.84 s). The
   point reports `total`.
2. **If a steady window is required** (after CL-B1), size `N = w + max(4, ceil(T_min x X / D))`,
   where `w ≥ 1` is the ramp crop, `T_min` the §6.2 floor, `X` throughput, `D` = 248. `N = 5` has
   no margin: the adaptive crop can take more than 1 super-pass.
3. **Configure the point:**

```yaml
submission_ref:
  ruleset: "endpoints-v1.0-2026-10-C1-A"  # binds the seeds (§9.1 seed-set validity)
  model: "wan-2.2-t2v-a14b"
settings:
  runtime:
    n_samples_to_issue: 248        # N x 248, an exact multiple
  load_pattern:
    type: "concurrency"
    target_concurrency: 1
  client:
    stream_all_chunks: true        # sec 6.5; no effect on a non-SSE adapter
```

There is no setting for sample order yet, so this run samples without replacement and does not
meet §6.5 until CL-C1 lands. The `submission_ref` model name must match the slug CL-A1 registers.
Keep `model_params.name` set to the served model name: when it is unset, the client fills it from
`submission_ref.model` (`config/schema.py:1179-1180`) and would send the slug to the server.

**Checks after the run:**

- **`n_samples_failed == 0`.** Termination counts issued, not completed
  (`load_generator/session.py:868-873`), and completed includes failures (§3.4).
- **Achieved duration.** The duration is achieved by count, not enforced.
- Do **not** use `min_issue_duration_ms`. Use `max_issue_duration_ms` only as a runaway guard set
  well above the expected finish; a binding cap ends the point mid-pass and breaks §6.4.

## 5. Pending work

Grouped by the categories in §0. Every code task carries tests; `AGENTS.md` sets > 90% coverage.

### A. Naming

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| CL-A1 | Register the 248-prompt dataset in `config/rulesets/mlcommons/datasets.py` with its MLPerf Inference provenance, under the slug `wan-2.2-t2v-a14b`. No per-model ruleset: the Endpoints round has none by design. | 1 | n/a | XS |

### B. Tokenless support

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| CL-B1 | Latency-gated steady state for tokenless workloads (§3.1, items 1-4). **Only if** the WG rules `not found` invalid or a steady-state basis is wanted. | 2 | WG §4.4 text | L |
| CL-B2 | Never tokenize artifact-output adapters (§3.2). | 3 | n/a | XS |

### C. Run settings

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| CL-C1 | Add `sample_order` to `RuntimeConfig` and map it in `from_config`. Accuracy phases stay safe: they run sequentially, and the audit override still wins. The `regenerate-templates` pre-commit hook updates the YAML templates. | 4 | n/a | XS |

### D. Example configs

| # | Task | Fixes | Depends on | Size |
| --- | --- | --- | --- | --- |
| CL-D1 | Point configs for the planned set (§1): full 248-prompt dataset, `n_samples_to_issue = N x 248`, `submission_ref`, `stream_all_chunks: true`, `sample_order: with_replacement`, no binding cap. Retire or relabel the 144-prompt and 20-sample configs. | 5 | CL-A1, CL-C1 | S |

### E. Documentation

| # | Task | Depends on |
| --- | --- | --- |
| CL-E1 | Update this document as items land; update `AGENTS.md` if a module moves or is added. | any |
| CL-E2 | Note the modality constraint in `steady_state_diagnostics.md` once CL-B1 settles. | CL-B1 |

**Critical path:** CL-C1 → CL-D1, with CL-A1 alongside. CL-B1 is conditional on the WG.

## 6. Out of scope for this repository

- The accuracy quality target (the scorer exists; the threshold is a benchmark-definition item).
- Rule text: §9.1 metric consistency, §4.1 metrics, §4.4 gating.
- Result layout, aggregation, and visualisation of a finished curve.
- The model server and serving stack under test.
