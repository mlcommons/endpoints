# Steady-State Studies — Coordination Brief

**Status:** Active — multi-agent investigation.
**Date:** 2026-07-29
**Branch:** `design/steady-state-windowing`.
**Orchestrator:** main session (in-session background subagents).
**Hivemind topic:** `steady-state-studies`.
**Related:** design `docs/design/2026-07-21-steady-state-windowing-design.md`,
findings `docs/design/2026-07-24-steady-state-windowing-findings.md`.

Three investigations extending the steady-state windowing work. Each agent reads
its own section below and executes. Deliverables: findings docs under
`docs/design/` + analysis scripts under `scripts/`. All agents run on opus.

## Corpus

Ground-truth `events.jsonl` runs live at
`~/skritch/endpoints-events-jsonl-artifacts/<point>/` (NOT in the repo — too large;
never copy into the tree). Each point has a sibling `config.yaml` carrying
`load_pattern.target_concurrency` (= `N`) and the performance dataset's sample
count (= `dataset_size`).

| point                 | N (concurrency) | issued | events lines | notes                   |
| --------------------- | --------------- | ------ | ------------ | ----------------------- |
| `C8`                  | 8               | 2226   | 19867        | negligible-ramp control |
| `C140`                | 140             | —      | 102934       |                         |
| `C1024`               | 1024            | 141169 | 436696       | mid contrast            |
| `C2048`               | 2048            | —      | 662794       |                         |
| `C7168`               | 7168            | 445756 | 1350457      | big drain               |
| `C22528`              | 22528           | 647616 | 1956037      | big drain               |
| `dsr1-c28k-tentative` | ~28080          | —      | 1996544      | big drain               |
| `poisson-232386`      | n/a (poisson)   | —      | 420004       | CoV generality only     |

`dataset_size` per point = the `n_samples` of the `type: performance` dataset in
that point's `config.yaml`. Derive `super_pass_size` via
`inference_endpoint.metrics.steady_state.series.super_pass_size(dataset_size, N)`.

All Python runs via `uv` (never bare `python3`), e.g.
`uv run --with pyarrow python scripts/...`.

## Phase 0 — shared sample extractor (SPINE — blocks Studies 1 & 3)

**Owner:** extractor agent. **Deliverable:** `scripts/steady_state_extract_samples.py`.

One streaming pass per `events.jsonl` → compact per-sample Parquet. Reuse
`series.py`'s `EventRecord` msgspec decoder and the ISSUED issue-counter /
super-pass bucketing logic (same retry handling: duplicate ISSUED refreshes the
issue timestamp, does not advance the counter). Stream line-by-line — never load
the 18G file into memory.

Per-sample columns (raw timestamps — do NOT bake in a window/drain flag; each
study defines its own window):

- `uuid`
- `super_pass` (issue-order bucket index)
- `issue_ns`, `first_token_ns` (RECV_FIRST, nullable), `complete_ns` (COMPLETE, nullable)
- `ttft_ns` = `first_token_ns - issue_ns`
- `lifetime_ns` = `complete_ns - issue_ns`
- `out_tokens` — from the response payload's reported completion-token count **if
  present in the event data**; else leave null (studies tokenize on demand). Do
  NOT tokenize 2M outputs in the extractor.
- `tpot_ns` = `(complete_ns - first_token_ns) / out_tokens` when `out_tokens`
  known, else null.

Emit a sidecar run-meta JSON per point: `N`, `dataset_size`, `super_pass_size`,
`first_issue_ns`, `last_issue_ns` (global), `last_complete_ns`, `n_issued`,
`n_complete`.

Output tables → `~/skritch/endpoints-events-jsonl-artifacts/<point>/samples.parquet`

- `<point>/run_meta.json`. Run over every corpus point (parallelize across points
  where practical; the big ones take minutes). Confirm row counts against
  `n_issued`. Report completion + any anomalies to the orchestrator.

## Study 1 — Hairball / long-tail bunching (empirical)

**Depends on:** Phase 0 tables. **Shares tail-set definition with Study 3.**
**Deliverables:** `docs/design/2026-07-29-hairball-findings.md`,
`scripts/steady_state_tail_analysis.py`.

**Hypothesis:** long-lived requests bunch toward run end (a "hairball" of
still-in-flight samples after issuance stops). If those drain-tail samples share
the same **TPOT distribution** as shorter/steady samples, they can be cut or
dropped from the steady-state window without biasing per-token latency.

**Method (per big-drain point + C1024 contrast + C8 control):**

1. Define the drain-tail set = samples with `complete_ns > last_issue_ns` (the
   window's last issue). Define steady set = the complement within the windowed
   region. **Publish this exact tail-set definition to the `steady-state-studies`
   topic so Study 3 reuses it verbatim.**
2. Compare TPOT distribution tail vs steady: KS test, CoV, quantile overlay.
   Where `out_tokens` is null, tokenize only the tail set + a random steady sample
   of equal size (distribution comparison stays valid) via the repo tokenizer.
3. Test whether "long-lived" is simply "high-OSL": correlate `lifetime_ns` with
   `out_tokens`; is the tail a pure OSL-selection effect?
4. Quantify the bias of dropping the tail on each metric family: throughput,
   latency tail (p99/p99.9), TPOT.

**Verdict:** is tail-drop TPOT-safe? State which metrics it biases and by how
much. Recommend cut / keep / conditional.

## Study 3 — CoV robustness + issue-to-issue tail drop (empirical, coupled to 1)

**Depends on:** Phase 0 tables; reuses Study 1's tail-set definition (from the
topic). **Deliverables:** `docs/design/2026-07-29-cov-robustness-findings.md`,
`scripts/steady_state_cov_robustness.py`.

**Q1 — tail-drop bias (shared with Study 1).** The current windowing uses an
issue-to-issue throughput denominator (`first_issue → last_issue`), so drain-tail
completions never enter the throughput denominator but DO enter latency. Using
Study 1's tail set, quantify exactly what that asymmetry does to reported
throughput vs latency, across concurrency points.

**Q2 — CoV convergence-detector robustness.** Stress-test the CoV-based stopping
rule / drift detector (`src/inference_endpoint/metrics/steady_state/drift.py`,
`stopping.py`, `harness.py`; existing `scripts/steady_state_cov_sweep.py`,
`scripts/steady_state_drift.py`):

- Sensitivity to trailing-window length, CoV bound, and which percentiles feed
  the ensemble (p50/p99 TTFT/TPOT).
- False-convergence: a synthetic (or real) slowly-drifting series — does CoV
  declare steady when it should not?
- Noise robustness: inject jitter; does the verdict flip?
- Cross-point generality: run over all points incl. `poisson-232386`.

**Verdict:** where is CoV reliable, where does it mislead, recommended
default params (trailing-window, bound, metric set).

## Study 2 — Feathered / staggered starts (design + prototype + sim, independent)

**Depends on:** nothing (runs from t=0). Uses Study 1's measured burst-tail
magnitude to calibrate the sim if available, but does NOT block on it.
**Deliverables:** `docs/design/2026-07-29-feathered-starts-design.md`,
`scripts/staggered_ramp_sim.py`.

**Idea:** replace the t=0 fill-burst (all `N`, or all queries in max-throughput)
with a staged ramp — e.g. `N/4` at a time — in both `max_throughput` and
`concurrency` modes, so offered load climbs smoothly instead of shocking the
server.

**Scope this pass (offline only — NO live runs, NO merged strategy change):**

1. Design doc: staged-ramp policy for `ConcurrencyStrategy` and `BurstStrategy`
   (`src/inference_endpoint/load_generator/strategy.py`) — step size, step
   interval, how "fill N" generalizes to staged fill, interaction with the
   super-pass warmup crop.
2. Prototype (unmerged, in the doc or a scratch script): what the staggered
   `ConcurrencyStrategy` / `BurstStrategy` change would look like.
3. `scripts/staggered_ramp_sim.py`: offline sim of in-flight / offered-load curves
   for burst vs staged ramp, calibrated to the observed burst-tail magnitude, to
   estimate ramp-region shrinkage and TTFT-spike reduction.

**Verdict:** recommended ramp schedule + expected reduction in the non-steady
ramp region; open questions for a future live validation.

## Hivemind coordination protocol

- Topic `steady-state-studies` is the shared board. Orchestrator posts the kickoff
  - consolidates verdicts.
- Study 1 publishes its exact tail-set definition to the topic; Study 3 consumes it.
- Each agent posts a short completion statement (deliverable paths + one-line
  verdict) to the topic when done.
- Durable verdicts → hivemind memory via nominate/propose (human-gated), not
  written silently.

## Dependency graph

```
t=0 ─┬─ Phase 0 extractor ──────────┐
     └─ Study 2 (design + sim)      │
                                    ▼
              extractor done ─┬─ Study 1 (hairball) ──▶ tail-set def ──┐
                             └─ Study 3 (CoV robustness) ◀─────────────┘
```
