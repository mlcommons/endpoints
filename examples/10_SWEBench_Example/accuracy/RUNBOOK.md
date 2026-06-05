# SWE-bench Accuracy Smoke-Test Runbook

End-to-end validation for the SWE-bench accuracy pipeline. Unit tests mock all
subprocesses, so running the real pipeline is the only way to catch Docker,
HuggingFace access, or mini-swe-agent wiring issues.

## 0. Preconditions

- Docker daemon running (swebench harness spawns one container per instance).
- Network egress to PyPI and HuggingFace Hub.
- `uv` binary on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Parent endpoints env already synced (`uv sync --extra dev` from repo root).

## 1. Set up the mini-swe-agent venv

Create the venv and install dependencies once:

```bash
mkdir -p ~/vllm_test/swe_mini_combined
cd ~/vllm_test/swe_mini_combined
uv venv
uv pip install mini-swe-agent==2.3.0 swebench==4.1.0
```

Sanity check:

```bash
.venv/bin/mini-extra --help
.venv/bin/python -m swebench.harness.run_evaluation --help
```

Override the default path via env var if your venv lives elsewhere:

```bash
export MINI_SWE_AGENT_DIR=/path/to/your/swe_mini_combined
```

## 2. End-to-end test (requires live endpoint)

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_SWEBench_Example/swe_bench_accuracy.yaml
```

## Common failure modes

| Symptom                                  | Likely cause              | Fix                                                |
| ---------------------------------------- | ------------------------- | -------------------------------------------------- |
| `FileNotFoundError: mini-swe-agent venv` | venv not created          | Run the setup commands in §1                       |
| Docker error during `run_evaluation`     | Docker daemon not running | Start Docker and retry                             |
| HuggingFace rate limit                   | No auth token             | Set `HF_TOKEN` env var and retry                   |
| `uv: command not found`                  | uv not installed          | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
