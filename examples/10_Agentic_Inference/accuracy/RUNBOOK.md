# SWE-bench Accuracy Smoke-Test Runbook

End-to-end validation for the SWE-bench accuracy pipeline. Unit tests mock all
subprocesses, so running the real pipeline is the only way to catch Docker,
HuggingFace access, or mini-swe-agent wiring issues.

## 0. Preconditions

- Docker daemon running (swebench harness spawns one container per instance).
- Docker Hub auth or a pre-seeded image cache for uncached SWE-bench images.
- Network egress to PyPI and HuggingFace Hub.
- `uv` binary on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Parent endpoints env already synced (`uv sync --extra dev` from repo root).

## 1. Sync the accuracy subproject

From the repo root:

```bash
cd examples/10_Agentic_Inference/accuracy
uv sync
```

Sanity check:

```bash
uv run mini-extra --help
uv run python -m swebench.harness.run_evaluation --help
```

Override the default subproject path via env var if needed:

```bash
export SWE_BENCH_PROJECT_PATH="$(pwd)/examples/10_Agentic_Inference/accuracy"
```

## 2. End-to-end test (requires live endpoint)

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_Agentic_Inference/swe_bench_accuracy.yaml \
  --mode both
```

`--mode both` is required: `type: online` configs default to `TestMode.PERF`,
which skips accuracy datasets.

Scorer preflight resolves the requested SWE-bench instances and pre-pulls the
required Docker images before `mini-extra swebench` starts, using the configured
SWE-bench `workers` count and a compact full-count progress bar. Cached images
still complete immediately in that bar.

Qwen SWE-bench configs may opt into `enable_swebench_toolcall_patch: true` and
the `swebench_qwen_tools_template.yaml` template. That path builds a temporary
minisweagent package overlay with the replacement files shipped in this
subproject, prepends it to `PYTHONPATH` for the agent run, and leaves the
installed package untouched. Leave this flag unset for Kimi and other non-Qwen
runs.

## Common failure modes

| Symptom                                              | Likely cause                          | Fix                                                       |
| ---------------------------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| `FileNotFoundError: SWE-bench subproject not found`  | subproject not synced                 | Run `uv sync` in `examples/10_Agentic_Inference/accuracy` |
| Docker error during `run_evaluation`                 | Docker daemon not running             | Start Docker and retry                                    |
| `Failed to pre-pull required SWE-bench Docker image` | Docker Hub rate limit or missing auth | Run `docker login` or use a local image cache/mirror      |
