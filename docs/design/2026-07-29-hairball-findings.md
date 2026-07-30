# Hairball / long-tail bunching — findings (Study 1)

**Status:** Complete — empirical.
**Date:** 2026-07-29.
**Branch:** `design/steady-state-windowing`.
**Coordination:** `docs/design/2026-07-29-steady-state-studies-coordination.md` (Study 1).
**Script:** `scripts/steady_state_tail_analysis.py`.
**Corpus:** per-point `events.jsonl` + Phase-0 `samples.parquet` / `run_meta.json`
under `~/skritch/endpoints-events-jsonl-artifacts/<point>/` (not in the repo).
Per-point result JSONs land beside the events as `<point>/tail_analysis.json`.

## TL;DR verdict

**Dropping the drain-tail is TPOT-safe. Recommend CUT** (drop the tail from the
steady-state window), with a documentation caveat for reasoning models.

- The drain-tail's TPOT distribution is **marginally faster** than steady at
  every point (median 1.4–5.3% lower), because as issuance stops, concurrency
  falls from `N` toward 0 and per-token decode de-contends. Dropping the tail
  therefore removes the _fastest_-per-token samples and nudges reported TPOT p50
  **up** by 0.4–1.1% (more conservative), with p99 within ±0.2%. It never
  optimistically understates per-token latency.
- Latency tail (p99/p99.9) bias from tail-drop is **negligible (<0.32%)** at
  every point — the tail is only `N` samples against a 130k–620k steady body.
- Throughput is **invariant** to tail-drop: reported QPS is issue-based
  (`issued-count / (last_issue − first_issue)`), so tail completions never enter
  the numerator whether or not you drop them. Dropping the tail actually makes
  the latency window _consistent_ with the issue-to-issue throughput window.
- `lifetime_ns` is strongly OSL-driven (Pearson 0.67–0.98 vs real `out_tokens`),
  strongest for DeepSeek-R1. **But high-OSL ≠ high-TPOT** — per-token rate is
  contention-governed and roughly OSL-independent, so the tail's OSL selection
  does not bias TPOT.

## Tail-set definition (published to `steady-state-studies` for Study 3)

Windowed region = `super_pass >= 1` (drop super-pass 0 warmup). The window's last
issue == global `run_meta.last_issue_ns` (the last super-pass is retained).

```
drain_tail = { super_pass >= 1  AND  complete_ns >  last_issue_ns }
steady     = { super_pass >= 1  AND  complete_ns <= last_issue_ns }
```

`steady` is the complement of `drain_tail` inside `window = (super_pass >= 1)`.

### The tail is exactly `N`

At every concurrency point `|drain_tail| == N` — the drain-tail is _precisely_ the
`N` requests in flight at the instant issuance stops (concurrency mode holds `N`
concurrent, so exactly `N` completions land after `last_issue_ns`). The tail
fraction of the window grows with `N`:

| point                   | N      | super-passes | window  | tail = N | tail % window | drain after last-issue |
| ----------------------- | ------ | ------------ | ------- | -------- | ------------- | ---------------------- |
| C1024                   | 1024   | 23           | 134 773 | 1 024    | 0.76%         | 7.5 s                  |
| C7168                   | 7168   | 35           | 432 964 | 7 168    | 1.66%         | 18.4 s                 |
| C22528                  | 22528  | 26           | 622 032 | 22 528   | 3.62%         | 28.4 s                 |
| dsr1-c28k _(tentative)_ | ~28080 | 22           | 627 484 | 28 080   | 4.48%         | **525.7 s**            |

**C8 control is not windowable:** `n_issued = 2226 < dataset_size = 6396`, so only
super-pass 0 exists and `window = (super_pass >= 1)` is empty. C8 is
`partial_dataset` (see `series.coverage_status`); no tail-drop analysis is
possible there. C1024 serves as the low-concurrency contrast instead.

## 1. TPOT: tail vs steady

Real tokenization on demand — the full tail set + an equal-size random steady
sample per point — with the point's model tokenizer (gpt-oss-120b `o200k_harmony`
for the C-points; `deepseek-ai/DeepSeek-R1` for dsr1). TPOT denominator is
`token_count(text_after_first_chunk())`, matching the repo's own TPOT convention
(`series.py`); numerator is `complete_ns − first_token_ns`.

| point              | tail p50 (ms) | steady p50 (ms) | Δp50  | tail mean | steady mean | tail CoV | steady CoV | KS stat |
| ------------------ | ------------- | --------------- | ----- | --------- | ----------- | -------- | ---------- | ------- |
| C1024              | 4.90          | 4.97            | −1.4% | 4.79      | 4.97        | 0.048    | 0.009      | 0.443   |
| C7168              | 9.29          | 9.58            | −3.0% | 9.06      | 9.59        | 0.088    | 0.015      | 0.580   |
| C22528             | 21.33         | 22.53           | −5.3% | 20.65     | 22.55       | 0.100    | 0.016      | 0.627   |
| dsr1 _(tentative)_ | 45.59         | 47.36           | −3.7% | 43.91     | 46.53       | 0.103    | 0.060      | 0.498   |

**The tail is consistently the faster-per-token set.** As the run drains, offered
concurrency collapses from `N` to near-0, so decode batches shrink and per-token
latency drops; the tail spans that ramp, which also inflates its CoV (0.05–0.10
vs 0.01–0.06 for steady).

The KS test rejects distributional equality (`p ≈ 0`) at every point, but this is
an **n-driven artifact** (n = 1k–28k per side): the _effect size_ is a 1.4–5.3%
median shift, and it is in the **safe direction** — steady-only TPOT is
marginally _pessimistic_ relative to the full window. Dropping the tail cannot
make reported per-token latency look better than it is.

## 2. Long-lived == high-OSL?

`lifetime_ns` vs output length. Char length (`len(str(output))`) is available for
**all** completed samples (OSL proxy); real `out_tokens` is on the tokenized
subset only. The two agree, so the correlation is not a proxy artifact.

| point              | Pearson(lifetime, charlen) [all] | Pearson(lifetime, out_tokens) [subset] | tail out_tok median | steady out_tok median | tail/steady OSL |
| ------------------ | -------------------------------- | -------------------------------------- | ------------------- | --------------------- | --------------- |
| C1024              | 0.864                            | 0.957                                  | 1259                | 1227                  | 1.03×           |
| C7168              | 0.672                            | 0.669                                  | 1254                | 1215                  | 1.03×           |
| C22528             | 0.807                            | 0.885                                  | 1261                | 1213                  | 1.04×           |
| dsr1 _(tentative)_ | 0.956                            | 0.981                                  | 6835                | 1991                  | **3.43×**       |

- **Yes, lifetime is strongly OSL-driven** (r = 0.67–0.98). For the gpt-oss
  C-points the tail's OSL is only ~3% above steady — the tail is essentially a
  _random in-flight snapshot_ (slightly-above-median OSL), not a strong high-OSL
  selection. (Consistently: for gpt-oss the tail lifetime median ≈ steady; the
  tail is "whatever was running at the cutoff," not the longest-lived requests.)
- **DeepSeek-R1 is the exception:** its OSL is heavy-tailed (reasoning traces), so
  the longest-OSL requests dominate the in-flight set at drain — the dsr1 tail is
  3.4× the steady OSL and drains for **525 s** after last issue.
- **Crucially, high-OSL does not imply high-TPOT.** Even the 3.4×-OSL dsr1 tail
  has _lower_ TPOT than steady (§1). Per-token rate is set by batch contention,
  not sequence length, so the tail's OSL selection does not bias TPOT.

## 3. Per-metric bias of dropping the tail

| point              | latency p99 | latency p99.9 | TPOT p50 | TPOT p99 | throughput (QPS) |
| ------------------ | ----------- | ------------- | -------- | -------- | ---------------- |
| C1024              | +0.04%      | +0.05%        | +0.54%   | +0.22%   | 0% (issue-based) |
| C7168              | −0.31%      | −0.22%        | +0.92%   | +0.17%   | 0%               |
| C22528             | +0.07%      | +0.03%        | +1.12%   | +0.18%   | 0%               |
| dsr1 _(tentative)_ | +0.015%     | −0.007%       | +0.40%   | −0.04%   | 0%               |

- **Latency tail: negligible (<0.32%).** The tail is `N` of the longest-lifetime
  requests, but at only 0.76–4.48% of the window the steady body already sets
  p99/p99.9; removing the tail barely moves them (sign varies, magnitude tiny).
- **TPOT: negligible and safe.** p50 rises 0.4–1.1% (steady-only is more
  conservative), p99 within ±0.2%.
- **Throughput: exactly zero.** QPS = `issued-count / (last_issue − first_issue)`
  is defined on issuance, so drain-tail completions are _already_ outside the
  numerator. The "completion-rate over window vs steady" framing differs only by
  the tail fraction (0.76–4.48%), but that rate is not the reported metric.

### The real asymmetry (shared with Study 3 Q1)

The current window uses an **issue-to-issue throughput denominator** but includes
drain-tail completions in **latency**. So without tail-drop, latency reflects
`N` extra requests (the ones still draining) that throughput gets no credit for.
The magnitude of that asymmetry at the reported percentiles is small (<0.32% on
p99/p99.9 latency), but **dropping the tail removes it entirely** and makes the
latency and throughput windows consistent — a correctness improvement independent
of the (negligible) percentile shift.

## Recommendation: CUT (conditional caveat for reasoning models)

**Cut the drain-tail** (`complete_ns > last_issue_ns`) from the steady-state
window. It is:

1. **TPOT-safe** — safe/conservative direction, <1.1% on p50, <0.2% on p99.
2. **Latency-safe** — <0.32% on p99/p99.9.
3. **Throughput-neutral** — QPS is issue-based and unaffected.
4. **Consistency-improving** — aligns the latency window with the issue-to-issue
   throughput window (removes the drain-tail latency/throughput asymmetry).

**Caveat — reasoning models (DeepSeek-R1):** the tail is a strong high-OSL
selection (3.4× steady OSL) with a very long drain (525 s vs 18–28 s for
gpt-oss). Cutting it is still TPOT/latency/throughput-safe, but it discards the
longest-OSL requests wholesale. For pure per-token / latency / throughput
reporting, cut. If OSL-distribution coverage of the _reported_ set matters
(e.g. paired accuracy or OSL histograms), note that steady-only under-samples the
OSL tail and report OSL coverage separately.

## Caveats & honesty notes

- **dsr1-c28k is tentative.** It is a config-less point; `N ≈ 28080`, `S = 7`,
  `dataset_size = 4388` are _assumed_ (per the 2026-07-24 findings and Phase-0
  `run_meta` note). Treat its numbers as indicative, not authoritative.
- **No char-proxy TPOT.** TPOT used **real** tokenization (gpt-oss-120b and
  DeepSeek-R1 tokenizers), not the char-count fallback. The char length was used
  only as the _all-samples_ OSL proxy for the lifetime↔OSL correlation, and it is
  cross-checked against real `out_tokens` on the tokenized subset (§2) — the
  correlations agree, so the proxy is not load-bearing for any verdict.
- Tokenization counts registered harmony/special tokens as single tokens (fast
  tokenizer default), matching how the model emitted them; TPOT magnitudes
  (~5 ms gpt-oss decode → ~45 ms dsr1) are physically sensible.
- KS `p ≈ 0` everywhere reflects sample size, not practical effect; the reported
  effect sizes (median/percentile shifts) are the decision-relevant quantities.
