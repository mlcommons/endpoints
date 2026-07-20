# SWE-bench Service

Runs mini-swe-agent and the SWE-bench harness on a host with Docker. The
benchmark client only needs this service URL, but the service is trusted
infrastructure: it receives one endpoint URL and optional endpoint credentials, runs
Docker-backed evaluations, and serves run artifacts.

The isolated service subproject commits its own `uv.lock` so deployments use a
reproducible dependency set.

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080
```

The endpoint URL in the benchmark config must be reachable from the service
host. Service mode supports exactly one endpoint URL and follows the
LiveCodeBench-style external-service convention for heavyweight evaluation work.
Docker is required only on the service host.
The benchmark client submits a run to this service only in `ACC` or `BOTH`
mode; the default `PERF` mode skips external evaluation.

For non-loopback deployments, bind only on a private network or set
`--auth-token TOKEN` and configure the client with:

```yaml
accuracy_config:
  extras:
    swebench_service_url: http://swebench-host:18080
    swebench_service_auth_token: TOKEN
```

The service selects templates from its packaged allowlist. Use
`accuracy_config.extras.swebench_template: qwen_tools` to select both the Qwen
template and packaged `QwenToolsModel`; otherwise omit the template option.
Completed run metadata and artifacts are retained up to `--max-stored-runs`
runs.
