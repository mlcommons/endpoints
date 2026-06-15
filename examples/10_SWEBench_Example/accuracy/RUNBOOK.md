# SWE-bench Accuracy Smoke-Test Runbook

End-to-end validation for the SWE-bench accuracy pipeline. Unit tests mock all
subprocesses, so running the real pipeline is the only way to catch Docker,
HuggingFace access, or mini-swe-agent wiring issues.

## 0. Preconditions

- Docker daemon running (swebench harness spawns one container per instance).
- Network egress to PyPI and HuggingFace Hub.
- `uv` binary on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Parent endpoints env already synced (`uv sync --extra dev` from repo root).

## 1. Sync the accuracy subproject

From the repo root:

```bash
cd examples/10_SWEBench_Example/accuracy
uv sync
```

Sanity check:

```bash
uv run mini-extra --help
uv run python -m swebench.harness.run_evaluation --help
```

Override the default subproject path via env var if needed:

```bash
export SWE_BENCH_PROJECT_PATH=/path/to/examples/10_SWEBench_Example/accuracy
```

## 2. End-to-end test (requires live endpoint)

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_SWEBench_Example/swe_bench_accuracy.yaml
```

## Common failure modes

| Symptom                                             | Likely cause              | Fix                                                      |
| --------------------------------------------------- | ------------------------- | -------------------------------------------------------- |
| `FileNotFoundError: SWE-bench subproject not found` | subproject not synced     | Run `uv sync` in `examples/10_SWEBench_Example/accuracy` |
| Docker error during `run_evaluation`                | Docker daemon not running | Start Docker and retry                                   |
| HuggingFace rate limit                              | No auth token             | Set `HF_TOKEN` env var and retry                         |
| `uv: command not found`                             | uv not installed          | `curl -LsSf https://astral.sh/uv/install.sh \| sh`       |
