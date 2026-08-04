# SWE-bench Service

Runs mini-swe-agent and the SWE-bench harness on a host with Docker or Pyxis. The
benchmark client only needs this service URL, but the service is trusted
infrastructure: it receives one endpoint URL and optional endpoint credentials, runs
container-backed evaluations, and serves run artifacts.

The isolated service subproject commits its own `uv.lock` so deployments use a
reproducible dependency set.

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080 \
  --auth-token "$SWEBENCH_SERVICE_AUTH_TOKEN"
```

The endpoint URL in the benchmark config must be reachable from the service
host. Service mode supports exactly one endpoint URL and follows the
LiveCodeBench-style external-service convention for heavyweight evaluation work.
Docker is required only on the service host when using the default runtime. To use
ARM64 task images from a registry on a retained one-node Slurm allocation, add
`--runtime pyxis --image-registry REGISTRY`, for example
`registry.example.com/group/project`. Images must use the name
`sweb.eval.arm64.<instance_id>:v4.1.0-arm64`.
Pyxis pulls and caches each image through Enroot; configure registry credentials
in `~/.config/enroot/.credentials` when the registry requires authentication.
The benchmark client submits a run to this service only in `ACC` or `BOTH`
mode; the default `PERF` mode skips external evaluation.

The service requires `--auth-token TOKEN` by default. Configure the client with:

```yaml
accuracy_config:
  extras:
    swebench_service_url: http://swebench-host:18080
    swebench_service_auth_token: TOKEN
```

For isolated local development only, pass `--allow-unauthenticated` explicitly.
`/health` is intentionally public for liveness probes; every run and artifact route
requires the bearer token.

The service selects templates from its packaged allowlist. Use
`accuracy_config.extras.swebench_template: qwen_tools` to select both the Qwen
template and packaged `QwenToolsModel`; otherwise omit the template option.
Completed run metadata and artifacts are retained up to `--max-stored-runs`
runs.
