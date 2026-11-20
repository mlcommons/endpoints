# WAN 2.2 text-to-video (wan-2.2-t2v-a14b)

Client configs for a video-generation server such as `trtllm-serve` with
`POST /v1/videos/generations`. The plan for the MLPerf Endpoints benchmark is in
[`docs/t2v_benchmark_gaps.md`](../../docs/t2v_benchmark_gaps.md).

Every MLPerf Endpoints point, C1 to Cx (`C_max`), is a run of
`endpoints_point_wan22.yaml`: performance, then VBench.

## Endpoints points

Each point is one run of `endpoints_point_wan22.yaml`. `CONCURRENCY` sets the
point; `N_SAMPLES` is a multiple of the 248-prompt dataset size (§6.4), large
enough to meet the §6.2 minimum duration (600 s at concurrency 1-32, 1200 s
above). Prompts are sampled with replacement (§6.5), so this sets the count; it
does not issue every prompt the same number of times. `ENDPOINT` and
`SERVED_MODEL` default to `http://localhost:8000` and `wan22`.

```bash
CFG=examples/09_Wan22_VideoGen_Example/endpoints_point_wan22.yaml
for c in 1 4 8; do
  CONCURRENCY=$c N_SAMPLES=248 inference-endpoint benchmark from-config -c "$CFG"
done
CONCURRENCY=33 N_SAMPLES=496 inference-endpoint benchmark from-config -c "$CFG"
```

The default mode runs the performance phase, then VBench accuracy on the same
endpoint. Use `--mode perf` for a point without an accuracy run.

## Submission upload

`package_submission.py` copies the point runs into an upload tree for the cohort
checker (`python -m inference_endpoint.validation`). Gaps the checker still
reports for video runs are in `docs/t2v_benchmark_gaps.md` §5.
