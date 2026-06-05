# BFCL v4 Accuracy Benchmarking

Berkeley Function Calling Leaderboard (BFCL) v4 accuracy evaluation against an
OpenAI-compatible endpoint. Covers both single-turn function-calling subsets and
the agentic multi-turn subsets.

The sampling here is tuned so a full four-category run finishes on an edge
device in **under 3 hours**.

## Reproducing from the PRs

This feature spans two pull requests that must both be present:

| PR | Branch | What it does |
| --- | --- | --- |
| [PR #1](https://github.com/Palanivelg/endpoints/pull/1) | `chore/relax-numpy-pin` | Relaxes `numpy==2.4.4` → `>=1.26.4` so that `bfcl-eval` (which hard-pins `numpy==1.26.4`) can be installed alongside the project |
| [PR #2](https://github.com/Palanivelg/endpoints/pull/2) | `feat/bfcl-v4-combined` | Full BFCL v4 ST + MT accuracy integration, adapter fixes, and seed forwarding |

**PR #1 must be merged into `main` before `[bfcl]` can be installed.** The numpy pin conflict means `pip install -e ".[bfcl]"` will fail without it.

### Installing from the branches

```bash
# Clone your fork (or the mlcommons upstream)
git clone https://github.com/Palanivelg/endpoints.git
cd endpoints

# Check out the BFCL v4 branch (contains both PR changes during review)
git checkout feat/bfcl-v4-combined

# Install with the [bfcl] extra (requires PR #1's numpy pin already merged,
# or you're on a branch that includes it)
pip install -e ".[dev,test,bfcl]"

# Verify the install resolves without conflict:
pip show bfcl-eval numpy | grep -E "^(Name|Version)"
```

### Prerequisites

- Python 3.12+
- A served, OpenAI-compatible endpoint. The examples below assume
  `http://localhost:8080`, model `Qwen3.6-27B-Q4_K_M`, `temperature=0`.
- At least ~16 GB RAM for the model; ~3 hours wall-clock time for the full run.

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

### Reproducible sampling with `--seed`

Both paths support a seed for deterministic server-side sampling. Pass
`--seed <N>` to lock the RNG so repeated runs with the same model and same seed
produce identical outputs (assuming a deterministic server):

```bash
# Single-turn: override seed via from-config flag
inference-endpoint benchmark from-config \
  --config offline_bfcl_v4_single_turn.yaml \
  --accuracy-only \
  --seed 42

# Multi-turn: pass --seed directly to the CLI
python -m inference_endpoint.evaluation.bfcl_v4_multi_turn_cli \
  --endpoint http://localhost:8080 \
  --model Qwen3.6-27B-Q4_K_M \
  --sample-pct 3 \
  --temperature 0 \
  --seed 42 \
  --report-dir results/bfcl_v4_multi_turn/
```

`seed` is forwarded in the `seed` field of the `/v1/chat/completions` request
body. If the server does not support `seed`, it is silently ignored.

## Edge device budget (validated)

Full four-category run on an edge device (`Qwen3.6-27B-Q4_K_M`, `temperature=0`):

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

`Qwen3.6-27B-Q4_K_M`, `temperature=0`, validated against evalscope on a Thor
edge device:

Numbers below are from two independent seed validation runs on Thor
(`Qwen3.6-27B-Q4_K_M`, `temperature=0`, 456 single-turn samples, full
multi-turn `multi_turn_base`):

| Category | Run 1 | Run 2 | Match? |
| --- | --- | --- | --- |
| Single-turn `non_live` (AST) | 86.98% | 86.98% | ✓ |
| Single-turn `live` | 84.12% | 84.12% | ✓ |
| Single-turn `hallucination` | 94.32% | 94.32% | ✓ |
| **Single-turn overall** (456 samples) | **87.50%** | **87.50%** | ✓ |
| Multi-turn `multi_turn_base` (200/200) | 70.00% (140/200) | — | exact parity with evalscope |

Both single-turn runs are identical across seeds — confirming deterministic
scoring. The multi-turn result is exact parity with evalscope on the same model
and data. Wall-clock: ~82 min single-turn (~10.8 s/sample), ~80 min multi-turn.

### Verifying the output

After a single-turn run, the score is printed to stdout and also written to:

```
results/bfcl_v4_single_turn_accuracy/accuracy_scores.json
```

After a multi-turn run, per-entry details and aggregate scores are written to:

```
results/bfcl_v4_multi_turn/results.json
results/bfcl_v4_multi_turn/per_entry_scores.json
```

A quick sanity check:

```bash
# Single-turn: print the live-category accuracy
python3 -c "
import json, pathlib
s = json.loads(pathlib.Path('results/bfcl_v4_single_turn_accuracy/accuracy_scores.json').read_text())
print(s)
"

# Multi-turn: print overall accuracy
python3 -c "
import json, pathlib
r = json.loads(pathlib.Path('results/bfcl_v4_multi_turn/results.json').read_text())
print('Overall MT accuracy:', r['accuracy_scores']['bfcl_v4::multi_turn']['score']['overall_accuracy'], '%')
"
```
