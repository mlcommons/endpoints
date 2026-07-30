# Study 3 — CoV robustness + issue-to-issue tail-drop bias (findings)

**Status:** Complete. **Date:** 2026-07-29. **Branch:** `design/steady-state-windowing`.
**Coordination:** `docs/design/2026-07-29-steady-state-studies-coordination.md` (Study 3).
**Script:** `scripts/steady_state_cov_robustness.py`
(`uv run --with pyarrow --with numpy python scripts/steady_state_cov_robustness.py`).
**Corpus:** Phase-0 per-sample Parquet at
`~/skritch/endpoints-events-jsonl-artifacts/<point>/samples.parquet` (+ `run_meta.json`).
The Parquet `super_pass` column is the same issue-order bucketing
`series.build_super_pass_series` produces, so the per-super-pass `SuperPassRollup`
series is rebuilt directly from Parquet — no re-parse of the multi-GB
`events.jsonl`. The trailing partial super-pass is dropped (`series[:n_full]`) to
match the shipped sweep scripts.

**Tail-set definition** (pinned in the coordination brief, shared with Study 1):
`tail = complete_ns > last_issue_ns`. As of this writing Study 1 had **not** yet
published its tail-def to the `steady-state-studies` topic; this study uses the
pinned definition. If Study 1's published definition differs, re-check §Q1.

**Provenance caveats.** `dsr1-c28k-tentative` and `poisson-232386` are
config-less; `N`, `dataset_size` (hence `super_pass_size`) are assumed from the
07-24 findings / 07-21 design. Treat every dsr1/poisson number below as
**tentative**. `out_tokens`/`tpot_ns` are NULL on every point (text-only COMPLETE
payload), so the shipped CoV rule's TPOT source is unavailable — but the shipped
`rule_cov_converged` only uses `ttft_ns` + `latency_ns`, so this study exercises
the real detector, not a degraded one. TPOT-based ensembles are out of scope here
(Study 1 owns TPOT-invariance).

---

## Q1 — issue-to-issue tail-drop bias: throughput vs latency

The shipped window (`metrics/steady_state/window.py`) computes throughput as
`n_issued / (last_issue - first_issue)` — an **issue-to-issue** denominator that
ends at the last issue — while latency percentiles are taken over **all** completed
samples, including drain-tail samples that complete after `last_issue`. That is
the asymmetry: drain-tail completions never extend the throughput denominator but
do enter latency.

Per point (`*` = tentative config):

| point              | tail % | drain % | **throughput overstate %** | lat p99 infl % | lat p99.9 infl % | ttft p99 infl % |
| ------------------ | ------ | ------- | -------------------------- | -------------- | ---------------- | --------------- |
| C8                 | 0.36   | 0.43    | **+0.44**                  | +0.00          | +0.00            | +0.00           |
| C140               | 0.47   | 0.68    | **+0.69**                  | +0.02          | +0.05            | +0.07           |
| C1024              | 0.73   | 0.83    | **+0.84**                  | −0.02          | −0.05            | −0.20           |
| C2048              | 0.95   | 1.32    | **+1.34**                  | −0.04          | +0.00            | −0.87           |
| C7168              | 1.61   | 2.11    | **+2.15**                  | +0.18          | +0.05            | +3.79           |
| C22528             | 3.48   | 3.40    | **+3.52**                  | −0.33          | −0.13            | −0.78           |
| dsr1-c28k `*`      | 4.27   | 11.60   | **+13.12**                 | +0.06          | +0.01            | −0.67           |
| poisson-232386 `*` | 1.17   | 3.66    | **+3.79**                  | +0.00          | −0.00            | −0.25           |

Definitions: **throughput overstate %** = issue-to-issue QPS relative to a
wall-clock `first_issue -> last_complete` denominator, i.e. how much larger the
reported number is than the sustained end-to-end completion rate (`n_issued ==
n_complete` on every point, so this equals `full_span/iss_span - 1`).
**infl %** = the drain-tail's inflation of the reported percentile,
`(all - drop_tail)/drop_tail`.

### Findings

1. **Throughput is overstated, and the overstatement scales with the drain
   fraction**, monotonically with concurrency: +0.4 % (C8) → +3.5 % (C22528). It
   is small (< 3.5 %) for the whole gpt-oss ladder and only material at the
   big-drain dsr1 point (**+13 %**, tentative), where the drain span is 11.6 % of
   the run. Poisson sits at +3.8 %.

2. **Latency and TTFT percentiles are essentially unaffected by the asymmetry**
   (|infl| < 0.4 % at p99/p99.9 across every point, dsr1 included). The lone
   exception is C7168 TTFT-p99 at +3.8 %, a single-point wobble from elevated
   TTFT in the final super-passes, not a systematic tail effect (its e2e-latency
   inflation is +0.18 %).

3. **Why latency is immune: the drain-tail is not the latency tail.** A sample is
   in the drain-tail iff `issue_ns + lifetime_ns > last_issue_ns` — dominated by
   _when_ it was issued, not by how slow it was. Most drain-tail samples are
   simply the last batch issued at ordinary latency. Dropping the tail removes a
   near-representative slice, not the worst-case slice, so p99/p99.9 barely move.
   (Tail-set median e2e latency ≈ steady median on the C-points; the genuinely
   long-lived "hairball" requests are a small minority of the tail. This
   corroborates rather than contradicts Study 1's TPOT-invariance hypothesis —
   the drain-tail is not a distinct latency population.)

### Q1 verdict

The issue-vs-complete asymmetry is **benign for latency** (no adjustment needed)
and **small for throughput on the gpt-oss ladder** (≤ 3.5 %), becoming material
only for the big-drain dsr1 point (+13 %, tentative). Issue-to-issue throughput is
defensible as the _steady offered/completion rate_ (in a closed loop, steady issue
rate == completion rate; the drain is a ramp-down that legitimately sits outside
the steady window). The residual internal inconsistency equals the drain fraction.
Recommendation: keep the issue-to-issue denominator, but (a) **report the drain
fraction** alongside throughput so the reader can see the inconsistency, and (b)
where a true completion-rate is wanted, additionally report
`n_complete / (last_complete - first_issue)`. **No tail adjustment to latency is
warranted.** Flag the dsr1 +13 % as tentative pending config confirmation.

---

## Q2 — CoV convergence-detector robustness

Stress-tests `stopping.rule_cov_converged`, `drift.ensemble_vote`,
`drift.analyze_trend` (shipped default: trailing `window=3`, `cov_bound=0.05`,
`warmup=1`, percentiles `(0.5, 0.99)` over `ttft_ns` + `latency_ns`).

### Q2d — cross-point generality (shipped default rule)

| point              | n_sp | default region | ensemble conv | concordance |
| ------------------ | ---- | -------------- | ------------- | ----------- |
| C8                 | 0    | None           | 0/6           | 0.00        |
| C140               | 4    | (1, 4)         | 1/6           | 0.00        |
| C1024              | 22   | (1, 4)         | 6/6           | 0.86        |
| C2048              | 33   | (1, 6)         | 5/6           | 0.97        |
| C7168              | 34   | (1, 10)        | 6/6           | 0.97        |
| C22528             | 25   | (1, 8)         | 6/6           | 0.62        |
| dsr1-c28k `*`      | 21   | None           | 0/6           | 0.00        |
| poisson-232386 `*` | 31   | (1, 12)        | 6/6           | 0.70        |

C8 is the negligible-ramp control (one super-pass — not windowable). The default
rule converges on the mid-ladder and poisson; it **fails to converge** on dsr1
(heavy p99-TTFT drift) — correctly refusing a false steady value.

### Q2a — sensitivity (does "converged" mean "accurate"?)

For each point, sweep `window ∈ {2,3,4,5,6,8}` × `bound ∈ {0.02..0.15}` ×
percentile-grid `{p50, p99, p50+p99}` × metric-source `{ttft+lat, ttft, lat}`, and
score each converged region's **p99-TTFT rel-err vs the full post-warmup
asymptote**:

| point       | conv frac (ttft+lat) | p99-TTFT err med | p99-TTFT err max |
| ----------- | -------------------- | ---------------- | ---------------- |
| C1024       | 0.98                 | 0.015            | 0.036            |
| C2048       | 0.83                 | 0.017            | 0.041            |
| poisson `*` | 0.91                 | 0.035            | 0.140            |
| C7168       | 0.81                 | **0.377**        | **0.600**        |
| C22528      | 0.74                 | **0.195**        | **0.703**        |
| dsr1 `*`    | 0.47                 | **0.389**        | **0.561**        |

**Critical result:** on the drifting points (C7168, C22528, dsr1) the CoV rule
still _converges_ for most parameter settings, but the resulting window's p99 TTFT
is **20–70 % off** the asymptote. "Converged" does **not** imply "accurate" — a
short trailing window can satisfy the CoV bound at a p99 level that is still
climbing. C1024/C2048/poisson (metrics genuinely plateau) land within 1.5–4 %.

**Metric-set ablation:** `lat`-only converges 100 % of the time but at the
_worst_ p99-TTFT error (0.29–0.50 med on the drifting points) — it is falsely
comforting because e2e latency plateaus while TTFT drifts. Including `ttft` and
keeping the **p99** percentile makes the rule stricter and is what surfaces the
drift. Dropping ttft or p99 from the ensemble hides the exact failure mode.

### Q2b — false convergence on a synthetic slowly-drifting series

Synthetic series (12 super-passes, 2 % lognormal per-super-pass noise), linear
median drift swept 0 → 60 % total:

| total drift | CoV (w3, b.05) | CoV (w6, b.03) | trend rel-drift | trend SNR | trend verdict |
| ----------- | -------------- | -------------- | --------------- | --------- | ------------- |
| 0 %         | (1, 4)         | (1, 7)         | +0.002          | 2.0       | steady        |
| 10 %        | (1, 4)         | (1, 7)         | +0.088          | 80.1      | steady        |
| 20 %        | (1, 4)         | (1, 8)         | +0.166          | 152.9     | drifting_up   |
| 30 %        | (1, 4)         | **None**       | +0.237          | 220.9     | drifting_up   |
| 45 %        | (1, 4)         | **None**       | +0.331          | 314.7     | drifting_up   |
| 60 %        | (1, 4)         | **None**       | +0.413          | 399.5     | drifting_up   |

**The shipped default rule (w3, b0.05) declares "converged" at every drift
level, including 60 % total growth — it never detects drift.** A 3-point trailing
window over a near-linear ramp has low local CoV even when cumulative drift is
large; short windows are structurally blind to slow drift. The strict rule
(w6, b0.03) refuses to converge from 30 % drift up. The **trend gate**
(`analyze_trend`) flags `drifting_up` from 20 % with enormous SNR — it is the
reliable drift detector, and it is exactly what `drift.classify_run` pairs with
CoV.

### Q2c — noise robustness (per-sample jitter, 40 seeds, default rule)

| point       | jitter | converged frac | sp_end median | sp_end range | concordance mean |
| ----------- | ------ | -------------- | ------------- | ------------ | ---------------- |
| C1024       | 5–20 % | 1.00           | 4             | [4, 4]       | 0.85–0.86        |
| C7168       | 5 %    | 1.00           | 10            | [10, 10]     | 0.97             |
| C7168       | 20 %   | 1.00           | 7             | [7, 10]      | 0.65             |
| C22528      | 5–20 % | 1.00           | 8             | [7, 8]       | 0.60–0.62        |
| poisson `*` | 5–20 % | 1.00           | 12            | [9, 12]      | 0.68–0.70        |

The verdict is **stable under moderate jitter** (≤ 10 %: converged fraction 1.00,
sp_end unchanged). At 20 % per-sample jitter the picked window and ensemble
concordance start to wobble (C7168 concordance 0.97 → 0.65). Low concordance
co-occurs with high p99 error (C22528, dsr1) — **ensemble concordance is itself a
usable drift/instability signal**.

### Q2 verdict — where CoV is reliable, where it misleads, recommended defaults

**Reliable** when the underlying metric genuinely plateaus (C1024, C2048,
poisson): converged windows land within ~1.5–4 % of the asymptote and are robust
to ≤ 10 % per-sample jitter.

**Misleading** when p99 TTFT drifts (C7168, C22528, dsr1): the rule still
"converges" but the window's p99 TTFT is 20–70 % wrong, and on a synthetic drift
the short-window default never fires. CoV measures _local flatness_, not _global
stationarity_ — it cannot see a slow monotone ramp.

**Recommended defaults:**

1. **Never trust CoV alone. Gate every CoV verdict with the trend test**
   (`analyze_trend` / `classify_run`). If any reported metric is `drifting_up`,
   the window is a false number — report _drift_, not a point estimate. Make the
   drift gate mandatory, not advisory. This is the single most important
   recommendation.
2. **Trailing window `≥ 4–5`, not 3.** `window=3` is structurally blind to slow
   drift (Q2b). Pair a longer window with `bound ≈ 0.03–0.05`. The (w6, b0.03)
   config caught synthetic drift; the cost is fewer converged short runs, which
   the drift gate mitigates by refusing them explicitly.
3. **Keep both `ttft_ns` and `latency_ns` and keep the p99 percentile in the
   ensemble.** Latency-only / p50-only converge more readily but hide p99 TTFT
   drift — the exact failure mode. p99 TTFT is the binding, most-drift-prone
   constraint.
4. **Use ensemble concordance as a guardrail:** treat `concordance < ~0.8` as
   "do not trust a single window; inspect for drift." It tracked the high-error
   points here (C22528 0.62, dsr1 0.00).

---

## Reproduce

```bash
uv run --with pyarrow --with numpy python scripts/steady_state_cov_robustness.py \
    --json /tmp/cov_robust.json
```

Runs Q1 (all points) + Q2a/b/c/d and writes the full machine-readable grid to the
`--json` path. Synthetic results (Q2b) are labelled as such and are independent of
the corpus; all other tables are real-data. dsr1/poisson rows are tentative
(config-less points).
