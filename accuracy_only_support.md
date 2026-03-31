# Accuracy-Only Benchmark Support

**Status**: Not yet supported. Tracked as a follow-up to the async loadgen PR.

## Goal

Allow `benchmark from-config` to run with only accuracy datasets (no performance dataset).

## Current Blockers

1. **Hard validation in `_load_datasets`** (`execute.py:218`):
   ```python
   if not performance_cfgs:
       raise InputValidationError("At least one performance dataset required")
   ```

2. **Unconditional perf phase in `_build_phases`** (`execute.py:340`):
   ```python
   phases.append(
       PhaseConfig("performance", ctx.rt_settings, ctx.dataloader, PhaseType.PERFORMANCE)
   )
   ```
   This uses `ctx.dataloader` which is the performance dataset loader — would be `None` in accuracy-only mode.

3. **`RuntimeSettings` depends on perf dataset** (`execute.py:305-308`):
   ```python
   rt_settings = RuntimeSettings.from_config(config, dataloader.num_samples())
   ```
   `num_samples()` is used to compute total samples to issue, duration estimates, and progress bar total.

## What Already Works

- **Report generation**: `finalize_benchmark` handles `perf_results` being empty (`perf = result.perf_results[0] if result.perf_results else None`). Report would show `duration_ns=None`, `qps=N/A`.
- **KVStore metrics**: `tracked_*` counters only increment during performance phases. With no perf phase, they stay at 0 — correct for accuracy-only runs.
- **Scoring artifacts**: `_write_scoring_artifacts` writes `sample_idx_map.json` from all phase results and copies `events.jsonl` — works without a perf phase.
- **Accuracy scoring**: Scorer reads `events.jsonl` and `sample_idx_map.json` independently of any perf phase.

## Required Changes

1. Remove the hard validation in `_load_datasets` — allow empty `performance_cfgs`.
2. Make `_build_phases` conditional — skip perf phase when no perf dataset exists.
3. Create a minimal `RuntimeSettings` for accuracy-only mode (no duration/QPS targets, progress bar total = sum of accuracy samples).
4. Handle `ctx.dataloader` being `None` throughout the flow.
5. Update `TestMode` handling — `TestMode.ACCURACY` should skip perf, `TestMode.BOTH` requires perf.
6. Update progress bar description for accuracy-only runs.

## Example Config (desired)

```yaml
name: "accuracy-only"
type: "offline"

model_params:
  name: "openai/gpt-oss-120b"
  streaming: "on"

datasets:
  - name: "aime25::gptoss"
    type: "accuracy"
    accuracy_config:
      eval_method: "pass_at_1"
      ground_truth: "answer"
      extractor: "boxed_math_extractor"
      num_repeats: 8

endpoint_config:
  endpoints:
    - "http://localhost:8000"
  api_type: "openai"
```
