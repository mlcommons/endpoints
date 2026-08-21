# `steady_state_diagnostics.py`

Post-hoc **steady-state / drift diagnostics** for a benchmark run's `events.jsonl`.
Self-contained (no `inference_endpoint` import) — runs anywhere with a tokenizer via
`uv`. The full methodology lives in
[`docs/steady-state-detection.md`](../docs/steady-state-detection.md); this is the
operator's quick reference.

## What it does

1. Buckets performance-tracked samples into **super-passes** by issue order
   (`--superpass-size` samples each, default `--dataset-size`).
2. Reconstructs per-sample **TTFT** (`recv_first − issued`) and **TPOT**
   (`(complete − recv_first) / tokens(output-after-first-chunk)`); TPOT needs the
   `--tokenizer`, so it is required.
3. Finds the **first steady plateau**: grow a window from the start while it stays
   _admissible_ — every gated metric (TTFT/TPOT p50 & p95) is trend-steady
   (Mann–Kendall + Hamed–Rao) **and** within a CoV bound. A staircase jump breaks the
   window, segmenting the run into plateaus. The **first plateau is the reported steady
   state** (later plateaus are usually degradation). Selection follows MSER: pick by
   estimator precision, never by the throughput value.
4. Summarizes that window (TTFT/TPOT histograms + percentiles, **per-user & system
   TPS** with batch-means confidence intervals) and **flags a level shift** toward the
   end of the run (multi-plateau + Pettitt change-point) as an `anomaly`, rather than
   hiding it.

## Requirements

- `uv` (the script declares its deps inline via a PEP 723 header — only `transformers`).
- A tokenizer the run's model uses (HF id or local dir), e.g. `openai/gpt-oss-120b`,
  `deepseek-ai/DeepSeek-R1`.

## Run

```bash
uv run scripts/steady_state_diagnostics.py <run>/events.jsonl \
    --tokenizer <hf-id-or-dir> \
    --dataset-size <N> \
    [--superpass-size N]      # samples per super-pass (default: --dataset-size)
    [--window-sizes 4,6,8]    # diagnostic scan sizes, in super-passes (min useful: 4)
    [--warmup 1]              # leading super-passes dropped before analysis
    [--cov-bounds 0.03,0.05,0.08]
    [--trend-gate mk_hamed_rao]   # {mann_kendall,mk_hamed_rao,newey_west,theil_sen,slope_vs_scatter}
    [--alpha 0.05]
    [--json out.json]         # full machine-readable result
```

`--dataset-size` is the number of samples in one dataset pass (see the run's
`run_meta.json` / config).

## Interpreting the output

### Headline — `STEADY STATE`

```
=== STEADY STATE (headline) ===
  window: super-passes 0..3 (post-warmup), 23519 samples
  TPS per-user:    302.3 tok/s/user  CI [302.1, 302.5]
  TPS system:    40960.9 tok/s        CI [39399.3, 40606.8]
  TTFT p50 86.26ms  p90 156.10ms  p95 183.83ms  p99 248.51ms  mean 97.41ms
  TPOT p50 3.29ms   p90 3.44ms    p95 3.48ms    p99 3.56ms    mean 3.31ms
```

- **window** — the steady plateau, as **post-warmup** super-pass indices `lo..hi`, plus
  the pooled sample count it was measured over.
- **TPS per-user** = `1 / mean(TPOT)` — output tokens/s for a single stream
  (interactivity). **TPS system** = total output tokens ÷ window wall-clock (aggregate
  throughput). Each `CI` is a 95% batch-means interval (super-passes as batches), so it
  reflects per-super-pass variability, not a naïve iid interval.
- **TTFT / TPOT** — percentiles and mean over the pooled raw samples of the window.

If no window qualifies:

```
  not found: no admissible steady plateau
```

means no contiguous run of super-passes was steady enough — the run drifts or is too
short. The per-window diagnostics below show why (which metric failed CoV/trend).

### `ANOMALY` line

```
  ANOMALY: level shift at super-pass 6, TPOT +100.0% toward end of run (likely degradation)
```

A second, materially different plateau was detected after the first and confirmed by a
Pettitt change-point. The headline steady result is still the **first** plateau; this
line says the run degraded later (e.g. KV-cache eviction, a sick worker). `delta_pct`
is signed (+ = TPOT rose = worse).

### Diagnostics (below the headline)

Per `--window-size`: a **CoV steadiness** table (per gated/diagnostic metric, PASS/`fail`/`n/a`
against each CoV bound over the trailing window) and a **whole-run trend** summary per
metric across the algorithms. The full rolling drift scan (every window position) is
only in `--json`.

### `--json`

Full structured result: `steady_state` (window, `ttft`/`tpot` summaries + histograms,
`tps`, `anomaly` with every plateau), plus `trajectories`, `cov`, and `drift` (the
rolling scan) for deeper analysis.

## Caveats

- **TPOT parity.** Token counts use plain tokenization of the output; the live
  aggregator uses the chat-template path for reasoning/tool-call outputs, so absolute
  TPOT ms can differ for reasoning models. CoV and the trend tests are scale-invariant,
  so the steady/drift **verdicts** are unaffected — only the absolute TPOT magnitude.
- **Window sizes < 4** are useless for drift (the trend test needs ≥ 4 points); they
  still contribute to the CoV table.
- Window indices are **post-warmup relative** (add `--warmup` for absolute super-pass
  numbers).
