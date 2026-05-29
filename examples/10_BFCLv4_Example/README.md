# BFCL v4 Accuracy Benchmarking

Berkeley Function Calling Leaderboard (BFCL) v4 accuracy evaluation against an
OpenAI-compatible endpoint. Covers both single-turn function-calling subsets and
the agentic multi-turn subsets.

The sampling here is tuned so a full four-category run finishes on an embedded
device (e.g. NVIDIA Thor) in **under 3 hours**.

## Prerequisites

```bash
pip install -e ".[bfcl]"   # installs bfcl-eval
```

A served, OpenAI-compatible endpoint (the examples below assume
`http://localhost:8080`, model `Qwen3.6-27B-Q4_K_M`, `temperature=0`).

## Architecture: two run paths

BFCL v4 splits into two evaluation paths because they have very different shapes:

| Path | Categories | How it runs | Scorer |
| --- | --- | --- | --- |
| Single-turn | `non_live`, `live`, `hallucination` | benchmark accuracy pipeline (YAML) | `BFCLv4Scorer` (`ast_checker`) |
| Multi-turn | `multi_turn_*` | standalone agentic CLI | `BFCLv4MultiTurnScorer` (`multi_turn_checker`) |

Single-turn is a single request per sample. Multi-turn is an agentic loop
(parse tool calls → execute locally → feed results back → repeat), so it cannot
share the single-pass accuracy phase and is driven by its own CLI.

## 1. Single-turn (non_live + live + hallucination)

```bash
inference-endpoint benchmark from-config \
  --config offline_bfcl_v4_single_turn.yaml \
  --accuracy-only
```

`--accuracy-only` skips the performance phase entirely and forces a single worker
/ single connection for deterministic per-sample ordering.

Sampling (see `offline_bfcl_v4_single_turn.yaml`):

| Category | Rate | Floor (≤25 → 100%) |
| --- | --- | --- |
| non_live | 20% | — (no subset ≤25) |
| live | 10% | live_parallel (16) & live_parallel_multiple (24) → 100% |
| hallucination | 5% | — |

The floor (`subset_floor: 25`) is applied globally but, given the BFCL v4 subset
sizes, only the two tiny `live` subsets are actually promoted — so the result
matches a live-only floor.

## 2. Multi-turn

Multi-turn is not YAML-driven. Run it via the CLI with `--sample-pct` to match
the budget (3% ≈ ~24 entries across the multi-turn subsets):

```bash
python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \
  --endpoint http://localhost:8080 \
  --model Qwen3.6-27B-Q4_K_M \
  --sample-pct 3 \
  --temperature 0 \
  --report-dir results/bfcl_v4_multi_turn/
```

Omit `--sample-pct` to run all entries (the full ~200-entry `multi_turn_base`
plus the other multi-turn subsets), or pass `--subsets multi_turn_base` to
restrict to one subset.

## Thor budget (validated)

Full four-category run on Thor (`Qwen3.6-27B-Q4_K_M`, `temperature=0`):

| Category | % | Samples | Est. time |
| --- | --- | --- | --- |
| non_live | 20% | ~230 | ~33 min |
| live | 10% (tiny subsets → 100%) | ~171 | ~32 min |
| hallucination | 5% | ~56 | ~24 min |
| multi_turn | 3% | ~24 | ~80 min |
| **Total** | | **~481** | **~2h 49m** |

Multi-turn dominates runtime (~189 s/entry) despite being the smallest sample
count — it accounts for ~48% of the total. Single-turn samples are fast
(~9–25 s each).

Excluded from the reportable aggregates: `live_relevance` (not part of any
reported category) and `memory` (not implemented on this branch).

## Expected accuracy (reference)

`Qwen3.6-27B-Q4_K_M`, `temperature=0`, validated against evalscope:

- Single-turn `live` (10%): ~82%
- Multi-turn base (full 200): 140/200 = 70.00%, exact parity with evalscope.
