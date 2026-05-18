# VBench Accuracy Smoke-Test Runbook

End-to-end validation for the WAN 2.2 VBench accuracy pipeline. Run this on a GPU host before treating the PR as functionally validated — unit tests mock VBench entirely, so the only way to catch VBench API drift, prompt-suite-coverage issues, or naming-convention mismatches is to execute the real thing.

## 0. Preconditions

- **GPU host** with CUDA-capable GPU (VBench's per-dim models — CLIP/DINO/RAFT/AMT — require it).
- **Network egress** to PyPI + HuggingFace Hub (VBench downloads model weights on first use; ~5 GB).
- **trtllm-serve** running and reachable, exposing `POST /v1/videos/generations` with `response_format=video_path` support. Videos must land on a shared filesystem readable by both trtllm-serve and this process.
- **`uv`** binary available on PATH. Recommended: `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- The parent endpoints env is **already synced** (`uv sync --extra dev` from the repo root). VBench lives in this subproject only — do not install it into the parent venv.

## 1. Sync the accuracy subproject

From the repo root:

```bash
cd examples/09_Wan22_VideoGen_Example/accuracy
uv sync
```

Expected: resolution succeeds (lockfile is committed), ~115 packages installed into `.venv/` under this directory. Heavy: torch + decord + opencv pulled here. First sync takes ~2-5 minutes on a fast link.

Sanity check the runner can import its deps:

```bash
uv run python -c "import torch; import vbench; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('vbench', vbench.__file__)"
```

Expected: `cuda True` and a `vbench` path inside `.venv/`.

## 2. Stage-1: runner argument plumbing

Goal: confirm `vbench_runner.py` accepts arguments and locates VBench's bundled `VBench_full_info.json` without crashing inside VBench's loader.

```bash
# From examples/09_Wan22_VideoGen_Example/accuracy
mkdir -p /tmp/vbench_smoke/videos /tmp/vbench_smoke/out
# Drop a placeholder mp4 named with a known VBench-suite prompt
# (subject_consistency is prompt-agnostic, so any prompt VBench knows works)
touch "/tmp/vbench_smoke/videos/a person swimming in ocean-0.mp4"

uv run python vbench_runner.py \
  --videos-dir /tmp/vbench_smoke/videos \
  --out-dir /tmp/vbench_smoke/out \
  --name smoke \
  --dims subject_consistency
```

Expected (any of these is a _pass_ for this stage — we only care that arg parsing and VBench init worked):

- VBench prints model-download progress, then fails reading the empty `.mp4` with a decord/cv2 error. Fine — proves the wiring works.
- VBench writes `/tmp/vbench_smoke/out/smoke_eval_results.json` with a `subject_consistency` entry.

_Fail signals:_ `TypeError: load_json(None)`, `argparse` errors, `ImportError: vbench`, `FileNotFoundError: VBench_full_info.json`. Stop and diagnose.

## 3. Stage-2: scorer with a hand-picked subset

Goal: exercise `VBenchScorer.score()` end-to-end against a small set of real videos, bypassing the load generator.

Pre-generate 3-5 videos for known prompts that exist in VBench's standard suite, then run the scorer directly:

```bash
# From the repo root — activate the *parent* venv, not the subproject's.
cd /lustre/fsw/coreai_mlperf_inference/tinyinl/endpoints
source .venv/bin/activate

python - <<'PY'
import json
from pathlib import Path
from inference_endpoint.evaluation.scoring import VBenchScorer
from inference_endpoint.dataset_manager.dataset import Dataset
import pandas as pd, msgspec
from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput

report = Path("/tmp/vbench_smoke/report")
report.mkdir(parents=True, exist_ok=True)

# Build a tiny "benchmark-like" report_dir from existing videos on Lustre.
# Replace these paths/prompts with 3-5 of your actual trtllm-serve outputs.
samples = [
    ("a person swimming in ocean", "/path/to/swim_0.mp4"),
    ("a cat eating food",         "/path/to/cat_0.mp4"),
    ("a panda drinking coffee",   "/path/to/panda_0.mp4"),
]
uuids = [f"uuid-{i}" for i in range(len(samples))]
(report / "sample_idx_map.json").write_bytes(
    msgspec.json.encode({"smoke": dict(zip(uuids, range(len(samples))))})
)
enc = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
with (report / "events.jsonl").open("wb") as f:
    for uid, (_, vp) in zip(uuids, samples):
        rec = EventRecord(
            event_type=SampleEventType.COMPLETE,
            sample_uuid=uid,
            data=TextModelOutput(output=vp),
        )
        f.write(enc.encode(rec) + b"\n")

# Minimal Dataset stand-in.
class _Ds:
    dataframe = pd.DataFrame({"prompt": [p for p, _ in samples]})
    def num_samples(self): return len(samples)

scorer = VBenchScorer(
    dataset_name="smoke",
    dataset=_Ds(),
    report_dir=report,
    ground_truth_column="prompt",
    # Default vbench_project_path points at this subproject; override only if needed.
)
mean, n_repeats = scorer.score()
print(f"mean score = {mean:.4f}, n_repeats = {n_repeats}")
print(json.dumps(json.loads((report / "vbench_results" / "vbench_smoke_eval_results.json").read_text()), indent=2)[:2000])
PY
```

Expected:

- Subproject subprocess runs (you'll see VBench's stdout — model loads + per-dim progress).
- `mean score` is between `0.0` and `1.0` (typically `0.4-0.9` for real WAN 2.2 outputs).
- `vbench_results/vbench_smoke_eval_results.json` has all 6 keys, each `[aggregate_score, [per_video_details...]]`.

_Fail signals:_

- `KeyError: 'scene'` in the dict → VBench did not produce that dim. Check that prompts cover the dim in VBench's `VBench_full_info.json`.
- `FileNotFoundError` on a `.mp4` → check `_stage_videos` symlink target; verify the path exists from this host (Lustre/NFS visibility).
- VBench logs "no videos matched for dimension X" → the WAN 2.2 prompts don't actually overlap with that dim's suite. This contradicts the design assumption; capture which dims and prompts and flag it.

## 4. Stage-3: full benchmark + accuracy via the YAML

Goal: confirm `inference-endpoint benchmark from-config` runs the full flow end-to-end with `offline_wan22_accuracy.yaml`.

```bash
cd /lustre/fsw/coreai_mlperf_inference/tinyinl/endpoints
source .venv/bin/activate

# Optionally reduce samples for a faster smoke (edit a copy of the YAML):
#   samples: 10
#   n_samples_to_issue: 10

inference-endpoint benchmark from-config \
  --config examples/09_Wan22_VideoGen_Example/offline_wan22_accuracy.yaml
```

Expected:

- Benchmark issues N samples to trtllm-serve, collects video paths.
- `finalize_benchmark` invokes `VBenchScorer.score()`, which spawns the subproject subprocess.
- `logs/wan22_video_accuracy_vbench/` (or whatever `report_dir` resolves to) contains:
  - `events.jsonl`, `sample_idx_map.json`
  - `vbench_videos/` with `{prompt}-{i}.mp4` symlinks
  - `vbench_results/vbench_wan22_vbench_eval_results.json` with 6 dim entries
  - `results.json` with `accuracy_scores.wan22_vbench.score` set to the mean
- Console log line: `Score for wan22_vbench: <float> (1 repeats)`.

## 5. Common failure modes

| Symptom                                                                    | Likely cause                                                          | Fix                                                                                                                                 |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `TypeError: load_json(None)`                                               | VBench got `None` as `full_info_dir`                                  | Confirm `vbench_runner.py` includes the `importlib.resources` fallback (check commit `957fe6e`)                                     |
| `FileExistsError` on second score()                                        | Symlink not unlinked                                                  | Confirm `_stage_videos` calls `dst.unlink(missing_ok=True)` (commit `957fe6e`)                                                      |
| `VBenchScorer: dropped N failed/empty-output sample(s)` warning            | Some trtllm-serve queries returned empty `response_output`            | Inspect `events.jsonl` for the failed UUIDs; usually a server-side OOM / timeout. Not a scorer bug — fix the upstream server config |
| `no videos found for dimension X`                                          | WAN 2.2 prompts don't overlap with VBench's X-dim prompt suite        | Either drop X from `_VBENCH_DIMENSIONS` or expand the prompt set. **Flag the design assumption.**                                   |
| `OSError: [Errno 28] No space left on device` during VBench model download | Default HF cache on `/tmp` or `~/.cache` is full                      | `export HF_HOME=/lustre/.../hf_cache` before running                                                                                |
| `ModuleNotFoundError: vbench` inside `vbench_runner.py`                    | Subproject `.venv` not synced or `uv run --project` pointed elsewhere | `cd examples/09_Wan22_VideoGen_Example/accuracy && uv sync`                                                                         |

## 6. When this passes

Update the PR description's test plan checkbox: "End-to-end VBench run on a GPU host" → done, citing the report directory.

If any of stages 1-3 reveals a real bug, capture the exact failure and open a follow-up commit on the branch before marking ready for review.
