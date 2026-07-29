# SWE-bench Accuracy Smoke-Test Runbook

End-to-end validation for the SWE-bench accuracy pipeline. Unit tests mock all
subprocesses, so running the real pipeline is the only way to catch Docker,
HuggingFace access, or mini-swe-agent wiring issues.

## 0. Preconditions

- Docker daemon running on the SWE-bench service host.
- Docker Hub auth or a pre-seeded image cache on the service host.
- Network egress to PyPI and HuggingFace Hub from the service host.
- Endpoint URL reachable from the service host.
- `uv` binary on PATH (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Parent endpoints env already synced (`uv sync --extra dev` from repo root).

## 1. Start the SWE-bench service

From the repo root:

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080 \
  --auth-token "$SWEBENCH_SERVICE_AUTH_TOKEN"
```

Sanity check:

```bash
curl http://localhost:18080/health
```

## 2. End-to-end test (requires live endpoint)

Select the config for the model under test:

```bash
CONFIG=examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml
# For Kimi, use examples/10_Agentic_Inference/kimi_agentic_benchmark.yaml.

# PERF (default): agentic performance only; skips SWE-bench.
uv run inference-endpoint benchmark from-config --config "$CONFIG"

# BOTH: agentic performance followed by SWE-bench.
uv run inference-endpoint benchmark from-config --config "$CONFIG" --mode both

# ACC: SWE-bench only.
uv run inference-endpoint benchmark from-config --config "$CONFIG" --mode acc
```

Both configs include a performance dataset and the SWE-bench accuracy dataset.
The default `PERF` mode skips external evaluation, so it does not require a
running SWE-bench service. Client preflight and submission require `--mode acc`
or `--mode both`; `ACC` also skips the performance dataset.

Scorer preflight calls the service `/health` endpoint. It does not check Docker
or pre-pull images on the benchmark client.

The service is trusted infrastructure. It receives one endpoint URL and optional
endpoint credentials, runs Docker-backed evaluations, and serves artifacts. It
requires `--auth-token TOKEN`; set
`accuracy_config.extras.swebench_service_auth_token: TOKEN`. Use
`--allow-unauthenticated` only for isolated local development.

Qwen SWE-bench configs opt in with
`accuracy_config.extras.swebench_template: qwen_tools`. The service loads its
packaged Qwen template and activates `QwenToolsModel` through mini-swe-agent's
`model_class` hook. Omit this setting for Kimi and other non-Qwen runs.

## Common failure modes

| Symptom                              | Likely cause                              | Fix                                                |
| ------------------------------------ | ----------------------------------------- | -------------------------------------------------- |
| `swebench_service_url is required`   | Client config missing service URL         | Set `accuracy_config.extras.swebench_service_url`  |
| Service health check fails           | Service not running or unreachable        | Start the service or fix client-to-service routing |
| Docker error during `run_evaluation` | Docker daemon not running on service host | Start Docker on the service host and retry         |
