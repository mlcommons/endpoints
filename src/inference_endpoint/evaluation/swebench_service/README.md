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

The endpoint URL in the benchmark config must be reachable from the service host.
Service mode supports exactly one endpoint URL and follows the LiveCodeBench-style
external-service convention for heavyweight evaluation work.

## Runtime workflow

### Common workflow

The benchmark client sends the selected SWE-bench instances, model configuration,
and endpoint URL to the service. The service first runs mini-swe-agent to generate
one patch per instance and writes the patches to `preds.json`. It then evaluates
those predictions with the SWE-bench harness and returns the aggregate result and
retained run artifacts. The selected runtime changes where and how the task
containers execute; it does not change the benchmark client configuration or the
model endpoint request path.

### Docker runtime

Docker is the default runtime and is required only on the service host. During
generation, the service runs `mini-extra swebench` unchanged. mini-swe-agent selects
the official per-instance x86_64 SWE-bench image, starts a writable Docker container
for the trajectory, and executes every model tool call in that container so its
filesystem changes persist across turns.

After generation, the service passes `preds.json` to
`swebench.harness.run_evaluation`. The standard SWE-bench evaluator starts a fresh
Docker container for each prediction, applies the generated patch, runs the task's
evaluation script, captures its output, and grades it. The service collects the
result file produced by the harness and removes containers belonging to the run.

### Pyxis runtime

Select Pyxis with `--runtime pyxis --image-registry REGISTRY`, for example
`registry.example.com/group/project`. The current path uses ARM64 images named
`sweb.eval.arm64.<instance_id>:v4.1.0-arm64`. Pyxis pulls and caches them through
Enroot; configure registry credentials in `~/.config/enroot/.credentials` when the
registry requires authentication. Launch the service on the compute node inside an
active one-node Slurm allocation. The runtime requires `SLURM_JOB_ID` and
`SLURMD_NODENAME` and assumes the node is exclusive to the user.

Each `srun` step is given an explicit allow-list of environment variables rather
than the service's whole environment, so that inherited `SLURM_JOB_ID` /
`SLURM_STEP_ID` cannot corrupt a nested `srun`. Several entries on that list are
load bearing on real clusters:

| Variable                                              | Why it must reach the step                                                                                                                                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SLURM_CONF`                                          | Without it the child `srun` falls back to `/etc/slurm/slurm.conf` and aborts on a configless or multi-cluster site.                                                                         |
| `http_proxy`, `https_proxy`, `no_proxy` (+ uppercase) | Enroot performs the registry pull inside the step and needs the caller's proxy policy.                                                                                                      |
| `ENROOT_TEMP_PATH`, `ENROOT_CONFIG_PATH`              | Enroot creates the container inside the step. Dropping these discards the operator's override, so the multi-gigabyte create-time temp lands back on the device holding the unpacked rootfs. |

Credentials such as `OPENAI_API_KEY`, `HF_TOKEN` and the service auth token are
never forwarded, and no other `SLURM_*` variable is.

During generation, the service still uses mini-swe-agent for the agent loop and
model requests, but replaces its Docker environment with `PyxisEnvironment`. Every
trajectory receives a named, writable Pyxis container. Each tool call becomes an
overlapping `srun` step in that container, preserving filesystem changes across
turns. Tool commands run in private PID namespaces so one trajectory cannot signal
processes belonging to another trajectory.

Each step reports its outcome through two channels. The primary one is in band: the
step script prints `__MLPERF_STEP_RC__ <nonce> <rc>` on `srun`'s stdout, which needs
no readable shared filesystem and is stripped from the command output before it is
returned. The fallback is the status file written into the container's `/tmp` mount.

When a step reports through neither, `StepNotLaunched` (a `RunnerError`) is raised.
Besides `srun`'s own output it carries `srun_rc`, the observed `status` bytes, and
`provable_non_execution` -- true only when the status file is still `pending` and no
sentinel arrived, meaning the step script did not run even its first line and the
command definitely did not execute. Anything else leaves open that it did. Callers
deciding whether re-running is safe must use that flag rather than the message text.

Cluster note: on a busy controller these failures cluster around slurmctld RPC rate
limiting (`Job credential expired`). Pacing step creation below the controller's
`rl_refill_rate` is a deployment concern rather than a property of this package.

After generation, the Pyxis worker evaluates each prediction in a fresh `srun`
container step because the Docker-based SWE-bench evaluator cannot run on the
compute node. It mounts the patch, SWE-bench evaluation script, and output file into
the task image. It preserves SWE-bench 4.1.0's patch-application order, test timeout,
captured output, and `get_eval_report` grading. A patch failure or test timeout is an
unresolved task; an `srun`, Enroot, or container-start failure is an infrastructure
error that fails the run. The service then aggregates the per-instance reports and
removes its named Pyxis containers.

Pyxis namespaces a named container by its allocation: `--container-name=X` inside
job `N` is the Enroot container `pyxis_N_X`. `PyxisEnvironment.cleanup()` removes
that name and logs a warning if the removal does not succeed. Each trajectory's
rootfs is on the order of gigabytes and is only reclaimed by this call, so a
removal that quietly fails fills the node's Enroot data path for the rest of the
allocation. `scancel` does not reclaim them either -- ending the job does not
remove Enroot containers -- which is why the removal has to be both correctly
named and audible.

### Agent-phase failures

The agent phase runs `--workers` trajectories concurrently. If it fails after
some workers have already written their patches, the service still evaluates the
predictions that reached `preds.json` rather than discarding them: the failure is
recorded in the `agent_phase_error.txt` artifact and logged at ERROR, and the run
continues into the eval phase. Instances with no prediction are simply absent from
the results, which the harness reports as unresolved.

A run whose agent phase produced no predictions at all still fails, with the agent
error chained as the cause. Cancellation is never tolerated this way: it
propagates immediately and no eval phase runs.

Consumers that need to know a run was degraded should fetch `agent_phase_error.txt`;
its presence means some instances may be missing from `swe_bench_results.json`.

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
