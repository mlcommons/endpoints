# TEST04 — Output-Caching Audit (WAN 2.2)

## Introduction

The purpose of this audit is to verify the SUT is **not caching responses** when it
sees duplicate samples. It runs two performance phases back-to-back against the same
endpoint and compares their achieved throughput:

- **Reference phase** — issues distinct samples drawn from the dataset.
- **Audit phase** — repeats one fixed sample (`sample_index`) for every query.

If the SUT caches, the audit phase answers the repeated sample faster and its QPS rises
above the reference phase's. The audit fails when that speedup exceeds `threshold`.

This re-implements the intent of MLPerf Inference TEST04 (duplicate-query caching
detection). It is the only audit implemented today (`output_caching_test`).

## Prerequisites

- A reachable WAN 2.2 video-generation endpoint (`api_type: videogen`), e.g. a
  `trtllm-serve` exposing `POST /v1/videos/generations`. See
  [the WAN 2.2 accuracy runbook](../../../../examples/09_Wan22_VideoGen_Example/accuracy/RUNBOOK.md)
  for bringing one up.
- The prompt dataset
  [`wan22_prompts.jsonl`](../../../../examples/09_Wan22_VideoGen_Example/wan22_prompts.jsonl)
  (248 prompts).

## Configuration

The audit is enabled by an `audit:` block in a YAML config (YAML only — there is no CLI
flag). It runs **after** the main performance run (upstream MLPerf order: perf run,
then TEST04; restart or flush the SUT's response cache between stages so the reference
phase observes an unprimed SUT), performance-only. Both WAN 2.2 submission configs in
this directory include it:

- [`offline_wan22_submission.yaml`](../../../../examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml) — `max_throughput`
- [`single_stream_wan22_submission.yaml`](../../../../examples/09_Wan22_VideoGen_Example/single_stream_wan22_submission.yaml) — `concurrency: 1`

```yaml
audit:
  test: output_caching_test
  only: false # true = run only the audit, skip the main benchmark
  samples: 64 # reference-phase query count (required, >= 1)
  audit_samples: 64 # audit-phase query count (omit -> equals `samples`)
  sample_index: 3 # dataset row repeated in the audit phase (default 0)
  threshold: 0.10 # audit_qps must stay < ref_qps * (1 + threshold)
```

| Field           | Required | Default          | Meaning                                                                    |
| --------------- | -------- | ---------------- | -------------------------------------------------------------------------- |
| `test`          | yes      | —                | Audit test to run; `output_caching_test` is the only one registered today. |
| `samples`       | yes      | —                | Reference-phase query count (`>= 1`).                                      |
| `audit_samples` | no       | equals `samples` | Audit-phase query count; lower it to shorten the audit phase.              |
| `sample_index`  | no       | `0`              | Dataset row repeated in the audit phase; must be in range.                 |
| `threshold`     | no       | `0.10`           | Caching tolerance, `0 < threshold < 1`.                                    |

`sample_index` is bound-checked against the performance dataset **as loaded** — i.e. after
its own `samples:` truncation, not the underlying file's row count. A dataset with
`samples: 144` over a 248-row file only accepts `sample_index` in `[0, 144)`.

## Supported load patterns

The audit compares achieved QPS, so it only accepts load patterns where that QPS reflects
the SUT's own serving capacity, not an external arrival schedule:

| Load pattern     | MLPerf scenario                |
| ---------------- | ------------------------------ |
| `max_throughput` | Offline (`Samples per second`) |
| `concurrency`    | Single-Stream                  |

`poisson`, `agentic_inference`, `burst`, and `step` are rejected before any phase runs — a
rate-paced pattern like `poisson` can pin achieved QPS below SUT capacity regardless of
caching, which would mask the exact signal this audit exists to detect.

## Pass criteria

Both conditions must hold:

1. Each phase completed at least `requested * (1 - threshold)` of its queries.
2. `audit_qps < ref_qps * (1 + threshold)`.

A phase that does not complete cleanly (metrics drain timeout or interrupt) aborts the
audit with an error — partial data never produces a valid/invalid result.

The reference phase always runs before the fixed-sample audit phase, so any second-run
speedup unrelated to caching (warm connection pools, server JIT/allocator warmup) biases
toward a false FAIL; `threshold` absorbs this (`0.20` for single-stream vs `0.10` for
offline in the shipped examples).

## Running

```bash
inference-endpoint benchmark from-config \
    --config examples/09_Wan22_VideoGen_Example/offline_wan22_submission.yaml
```

The process exit code is `0` on PASS and `1` on FAIL.

## Output

All audit artifacts nest under `<report_dir>/audit/` so they do not intermingle with the
main run's output:

```
<report_dir>/audit/
├── reference/                    # reference-phase report dir
├── output_caching/               # audit-phase report dir
├── audit_result.json             # full result + comparison details (written first)
└── verify_OUTPUT_CACHING_TEST.txt  # "Performance check pass: True|False"
```

`audit_result.json`:

```json
{
  "test": "output_caching_test",
  "passed": true,
  "ref_qps": 1.234,
  "audit_qps": 1.201,
  "threshold": 0.1,
  "ref_completed": 64,
  "ref_requested": 64,
  "audit_completed": 64,
  "audit_requested": 64,
  "reason": "audit_qps=1.2010 < ref_qps * (1 + 10%) = 1.3574"
}
```

## Scope

- Only the `output_caching_test` audit is implemented.
- Phases are **count-driven** (`samples` / `audit_samples`); the MLCommons 10-minute
  minimum-duration floor is not enforced.

See [docs/compliance_audit_plan.md](../../../../docs/compliance_audit_plan.md) for the design.
