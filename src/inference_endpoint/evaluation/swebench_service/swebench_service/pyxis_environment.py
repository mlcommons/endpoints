# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from .runner import RunnerError

logger = logging.getLogger(__name__)

_SAFE_SRUN_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    # Proxy policy must reach enroot, which performs the registry pull inside
    # the step.
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    # srun locates its own configuration through SLURM_CONF.
    "SLURM_CONF",
    # Enroot reads these when Pyxis creates the container, which happens inside
    # the step. Dropping them silently discards the operator's override, so the
    # ~2.5 GB create-time temp lands back on whichever device holds the unpacked
    # rootfs -- exactly the device the override existed to protect.
    "ENROOT_TEMP_PATH",
    "ENROOT_CONFIG_PATH",
)
_STEP_STATUS = "/tmp/.mlperf_srun_status"
#: In-band marker the step script prints alongside its own return code. It is
#: the primary result channel: it travels back on srun's stdout and so needs no
#: readable shared filesystem. The status file remains the fallback.
_STEP_SENTINEL = "__MLPERF_STEP_RC__"
#: The status file contents before the step script runs its very first line.
_STEP_STATUS_PENDING = "pending"
_STEP_SCRIPT = r"""set +e
status_path=$1
timeout_s=$2
nonce=$3
shift 3
printf 'started\n' > "$status_path" 2>/dev/null
unshare --pid --fork --mount-proc timeout "$timeout_s" "$@"
returncode=$?
printf 'finished:%s\n' "$returncode" > "$status_path" 2>/dev/null
printf '\n__MLPERF_STEP_RC__ %s %s\n' "$nonce" "$returncode"
exit "$returncode"
"""


class StepNotLaunched(RunnerError):
    """An `srun` step that reported through neither result channel.

    Subclasses :class:`RunnerError` so every existing ``except RunnerError``
    keeps working, and records the facts a caller needs to reason about the
    failure rather than only read about it:

    ``srun_rc``
        `srun`'s own exit status.
    ``status``
        The bytes actually observed in the step status file.
    ``provable_non_execution``
        True only when the status file was still ``pending`` and no in-band
        sentinel arrived -- the step script did not run even its first line, so
        the command definitely did not execute. Anything else leaves open that
        it did, which is the distinction anyone deciding whether a re-run is
        safe has to make.
    """

    def __init__(
        self,
        message: str,
        *,
        provable_non_execution: bool,
        srun_rc: int | None,
        status: str,
    ) -> None:
        super().__init__(message)
        self.provable_non_execution = provable_non_execution
        self.srun_rc = srun_rc
        self.status = status


def read_step_sentinel(text: str, nonce: str) -> tuple[int | None, str]:
    """Return ``(returncode, output_without_the_sentinel)`` if the step reported.

    ``(None, text)`` when the step did not report in band. The nonce makes the
    marker unforgeable by the command's own output.
    """
    tag = f"{_STEP_SENTINEL} {nonce} "
    for line in reversed((text or "").splitlines()):
        if not line.startswith(tag):
            continue
        value = line[len(tag) :].strip()
        if value.lstrip("-").isdigit():
            return int(value), text[: text.rindex(line)].rstrip("\n")
    return None, text


def safe_srun_env() -> dict[str, str]:
    return {name: os.environ[name] for name in _SAFE_SRUN_ENV if name in os.environ}


def build_srun_command(
    *,
    argv: list[str],
    image: str | Path | None = None,
    name: str | None = None,
    mounts: list[tuple[Path, str]] | None = None,
    workdir: str | None = None,
) -> list[str]:
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        raise RunnerError("Pyxis runtime requires SLURM_JOB_ID")
    node = os.environ.get("SLURMD_NODENAME", "").strip()
    if not node:
        raise RunnerError("Pyxis runtime requires SLURMD_NODENAME")
    command = [
        "srun",
        "--overlap",
        f"--jobid={job_id}",
        "-N1",
        "-n1",
        f"--nodelist={node}",
    ]
    if image is not None:
        image_ref = str(image.resolve()) if isinstance(image, Path) else image
        command.append(f"--container-image={image_ref}")
    if name is not None:
        command.append(f"--container-name={name}")
    if image is not None or name is not None:
        command.extend(
            [
                "--container-writable",
                "--container-remap-root",
                "--no-container-mount-home",
            ]
        )
        if mounts:
            specs = []
            for source, destination in mounts:
                source_text = str(source.resolve())
                if "," in source_text or "," in destination:
                    raise RunnerError("Pyxis mount paths cannot contain commas")
                specs.append(f"{source_text}:{destination}")
            command.append("--container-mounts=" + ",".join(specs))
        if workdir is not None:
            command.append(f"--container-workdir={workdir}")
    command.extend(argv)
    return command


def _run_srun_step_once(
    *,
    argv: list[str],
    status_path: Path,
    timeout_s: int,
    failure_path: Path | None = None,
    image: str | Path | None = None,
    name: str | None = None,
    mounts: list[tuple[Path, str]] | None = None,
    workdir: str | None = None,
    stderr: int = subprocess.STDOUT,
) -> subprocess.CompletedProcess[str]:
    nonce = uuid.uuid4().hex
    status_path.write_text(f"{_STEP_STATUS_PENDING}\n")
    status_path.chmod(0o666)
    command = build_srun_command(
        image=image,
        name=name,
        mounts=mounts,
        workdir=workdir,
        argv=[
            "bash",
            "-c",
            _STEP_SCRIPT,
            "pyxis-step",
            _STEP_STATUS,
            str(timeout_s),
            nonce,
            *argv,
        ],
    )
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=stderr,
            timeout=timeout_s + 30,
            env=safe_srun_env(),
        )
    except subprocess.TimeoutExpired as exc:
        if failure_path is not None:
            failure_path.touch()
        raise RunnerError(
            f"Pyxis step exceeded its {timeout_s + 30}s deadline and was killed"
            + _srun_evidence(exc.output)
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        if failure_path is not None:
            failure_path.touch()
        raise RunnerError(
            "Pyxis infrastructure failure before the command completed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Primary channel: the step reported its own return code in band.
    reported, cleaned = read_step_sentinel(result.stdout, nonce)
    if reported is not None:
        result.stdout = cleaned
        result.returncode = reported
        return result

    # Fallback channel: the status file the step script wrote into the mount.
    try:
        status = status_path.read_text().strip()
    except OSError as exc:
        status = f"<unreadable: {exc}>"
    if status == f"finished:{result.returncode}":
        return result

    if failure_path is not None:
        failure_path.touch()
    raise StepNotLaunched(
        "Pyxis infrastructure failure before the command completed "
        f"(srun exited {result.returncode}, status={status!r})"
        + _srun_evidence(result.stdout),
        provable_non_execution=status == _STEP_STATUS_PENDING,
        srun_rc=result.returncode,
        status=status,
    )


def enroot_container_name(job_id: str, container_name: str) -> str:
    """The Enroot container name Pyxis derives from ``--container-name``.

    Pyxis namespaces every named container by the allocation it belongs to, so
    ``--container-name=X`` inside job ``N`` becomes the Enroot container
    ``pyxis_N_X``. Anything that later addresses the container by name --
    ``enroot list``, ``enroot remove`` -- has to use the same form.
    """
    return f"pyxis_{job_id}_{container_name}"


#: Opt-in JSONL sink for container-create durations. Off unless set, so this
#: adds nothing to a normal run. Creation is the step whose cost was invisible
#: -- it was only ever observable as a uniform block of SIGKILLs in `sacct`,
#: after the run was already lost -- so measuring it has to be possible without
#: re-deriving it from step accounting.
_CREATE_TIMING_ENV = "SWEBENCH_PYXIS_CREATE_TIMING_PATH"


def _record_create_timing(image: str | Path, seconds: float, *, ok: bool) -> None:
    path = os.environ.get(_CREATE_TIMING_ENV)
    if not path:
        return
    record = {
        "ts": time.time(),
        "image": str(image),
        "secs": round(seconds, 2),
        "ok": ok,
        "pid": os.getpid(),
    }
    try:
        # One short line per create, O_APPEND from many concurrent workers.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        # Observability must never be able to fail a run. A create that
        # succeeded and could not be logged is still a create that succeeded.
        logger.debug("could not record Pyxis create timing", exc_info=True)


def _srun_evidence(output: str | bytes | None, limit: int = 2000) -> str:
    """Attach srun's own words to a Pyxis failure.

    srun/pyxis/enroot report the actual cause -- image import failure, no space
    left, a step that never got resources -- on the stream this function
    captures. Dropping it turns every distinct infrastructure failure into one
    indistinguishable message, which is exactly what made a 200-instance run's
    17 lost units undiagnosable from its artifacts.
    """
    if not output:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    text = output.strip()
    if not text:
        return ""
    if len(text) > limit:
        text = "..." + text[-limit:]
    return f"\n--- srun output ---\n{text}"


#: Bounded re-attempts for a step that provably never launched. Set to 1 to
#: disable. A retry here is only ever reached when the step script did not run
#: its first line, so it cannot double-apply work -- see run_srun_step.
_STEP_RETRIES_ENV = "SWEBENCH_PYXIS_STEP_RETRIES"
_DEFAULT_STEP_RETRIES = 3
#: Optional JSONL sink recording every retry and its outcome. The schema matches
#: `swe_bench_distributed.infra_retry.RetryRecord`, which reads it back to
#: publish infra_retries_total / instances_saved_by_retry / run_quality. The two
#: sides cannot share code: this is an isolated subproject that must not import
#: the benchmark client, so they share a file format instead.
_STEP_RETRY_LOG_ENV = "SWEBENCH_PYXIS_INFRA_RETRY_LOG"
_RETRY_LOG_LOCK = threading.Lock()


def _step_retry_attempts() -> int:
    raw = os.environ.get(_STEP_RETRIES_ENV, "").strip()
    if not raw:
        return _DEFAULT_STEP_RETRIES
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("ignoring non-numeric %s=%r", _STEP_RETRIES_ENV, raw)
        return _DEFAULT_STEP_RETRIES


def _record_step_retry(
    *, target: str, attempt: int, outcome: str, detail: str | None = None
) -> None:
    path = os.environ.get(_STEP_RETRY_LOG_ENV)
    if not path:
        return
    record = {
        "target": target,
        "attempt": attempt,
        "outcome": outcome,
        "detail": detail,
        "at": time.time(),
    }
    try:
        # Accounting must never be able to take a run down.
        with _RETRY_LOG_LOCK, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        logger.debug("could not append to the infra retry log", exc_info=True)


def run_srun_step(**kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run one `srun` step, re-attempting only a *provable* non-launch.

    Retrying is a correctness decision, not a convenience: re-running a command
    that may already have run can apply an edit twice, delete twice, or double a
    test run, and none of those announce themselves. So the only failure retried
    here is :class:`StepNotLaunched` with ``provable_non_execution`` -- the status
    file still ``pending`` and no in-band sentinel, meaning the step script did
    not execute even its first line. Every other failure, including a
    ``StepNotLaunched`` that reached ``started``, is raised immediately.

    Measured signature, from an isolated probe with no model and no GPU (20
    nodes, 200 workers, 6273 ordinary shell steps): 63 steps failed and in all 63
    the status file still read ``pending``.

    Every attempt and outcome is appended to ``SWEBENCH_PYXIS_INFRA_RETRY_LOG``
    when set. A retry loop that quietly absorbs the defect it compensates for
    turns a broken cluster into an invisible one.
    """
    attempts = _step_retry_attempts()
    target = str(kwargs.get("name") or kwargs.get("image") or "pyxis-step")
    for attempt in range(1, attempts + 1):
        try:
            result = _run_srun_step_once(**kwargs)
        except StepNotLaunched as exc:
            if not exc.provable_non_execution:
                # The command may have run. Another attempt could double it.
                _record_step_retry(
                    target=target,
                    attempt=attempt,
                    outcome="not_retryable",
                    detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
                )
                raise
            outcome = "exhausted" if attempt == attempts else "retrying"
            _record_step_retry(
                target=target,
                attempt=attempt,
                outcome=outcome,
                detail=f"srun_rc={exc.srun_rc} status={exc.status!r}",
            )
            if attempt == attempts:
                raise
            logger.warning(
                "Pyxis step provably never launched (attempt %d/%d, srun rc=%s, "
                "status=%r); retrying",
                attempt,
                attempts,
                exc.srun_rc,
                exc.status,
            )
            time.sleep(min(30.0, 2.0 * attempt))
            continue
        if attempt > 1:
            _record_step_retry(target=target, attempt=attempt, outcome="recovered")
        return result
    raise AssertionError("unreachable")  # pragma: no cover


def resolve_image(image_registry: str, instance_id: str) -> str:
    if Path(instance_id).name != instance_id or instance_id in {".", ".."}:
        raise RunnerError(f"invalid SWE-bench instance ID: {instance_id}")
    image_registry = image_registry.rstrip("/")
    if "#" not in image_registry:
        host, separator, repository = image_registry.partition("/")
        if not separator:
            raise RunnerError("Pyxis image registry must include a repository")
        image_registry = f"{host}#{repository}"
    return f"{image_registry}/sweb.eval.arm64.{instance_id.lower()}:v4.1.0-arm64"


class PyxisEnvironmentConfig(BaseModel):
    image: str | Path
    run_id: str
    cwd: str = "/testbed"
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: int = Field(
        default=30,
        validation_alias=AliasChoices("timeout_s", "timeout"),
        serialization_alias="timeout",
    )
    #: Deadline for *creating* the container, which under Pyxis includes the
    #: enroot import of a multi-GB SWE-bench image from a remote registry.
    #: Deliberately separate from ``timeout_s``: that is a per-*command*
    #: budget, sized for `pytest`-scale work inside an already-running
    #: container. Charging an image import against it made every agent whose
    #: image was not already in the enroot cache fail once the registry was
    #: shared by enough concurrent workers to push a single import past ~5
    #: minutes. Defaults to, and accepts, mini-swe-agent's ``pull_timeout``.
    create_timeout_s: int = Field(
        default=3600,
        validation_alias=AliasChoices("create_timeout_s", "pull_timeout"),
        serialization_alias="pull_timeout",
    )
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-c"])
    infrastructure_failure_path: Path | None = None


class PyxisEnvironment:
    def __init__(self, **kwargs: Any):
        self.config = PyxisEnvironmentConfig(**kwargs)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "-", self.config.run_id)[:24]
        self.name = f"mswe_{safe_run_id}_{uuid.uuid4().hex[:8]}"
        self._tmp = tempfile.TemporaryDirectory(prefix=f"pyxis_{self.name}_")
        self._tmp_dir = Path(self._tmp.name)
        self._tmp_dir.chmod(0o1777)
        self._lock = threading.Lock()
        self._cleaned = False
        started = time.monotonic()
        try:
            # A no-op initializes and validates the named persistent container.
            run_srun_step(
                image=self.config.image,
                name=self.name,
                mounts=[(self._tmp_dir, "/tmp")],
                workdir=self.config.cwd,
                argv=["true"],
                status_path=self._tmp_dir / Path(_STEP_STATUS).name,
                timeout_s=self.config.create_timeout_s,
                failure_path=self.config.infrastructure_failure_path,
            )
        except RunnerError as exc:
            _record_create_timing(
                self.config.image, time.monotonic() - started, ok=False
            )
            self.cleanup()
            raise RunnerError(
                f"failed to start Pyxis container for {self.config.image}: {exc}"
            ) from exc
        _record_create_timing(self.config.image, time.monotonic() - started, ok=True)

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        logger.debug("Executing Pyxis command: %s", command)
        argv = ["env"]
        argv.extend(f"{key}={value}" for key, value in self.config.env.items())
        argv.extend([*self.config.interpreter, command])
        result = run_srun_step(
            argv=argv,
            status_path=self._tmp_dir / Path(_STEP_STATUS).name,
            timeout_s=timeout or self.config.timeout_s,
            failure_path=self.config.infrastructure_failure_path,
            name=self.name,
            mounts=[(self._tmp_dir, "/tmp")],
            workdir=cwd or self.config.cwd,
        )
        output: dict[str, Any]
        if result.returncode == 124:
            output = {
                "output": result.stdout,
                "returncode": -1,
                "exception_info": "The command timed out",
                "extra": {
                    "exception_type": "TimeoutExpired",
                    "exception": (
                        f"command timed out after {timeout or self.config.timeout_s}s"
                    ),
                },
            }
        else:
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
            # mini-swe-agent is installed only in the SWE-bench service subproject.
            from minisweagent.exceptions import Submitted

            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **self.config.model_dump(by_alias=True),
            **platform.uname()._asdict(),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json", by_alias=True),
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                }
            }
        }

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
        try:
            job_id = os.environ.get("SLURM_JOB_ID", "").strip()
            if job_id:
                container = enroot_container_name(job_id, self.name)
                try:
                    completed = subprocess.run(
                        build_srun_command(argv=["enroot", "remove", "-f", container]),
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=safe_srun_env(),
                    )
                except (OSError, RunnerError, subprocess.SubprocessError):
                    logger.warning(
                        "Could not remove Pyxis container %s",
                        container,
                        exc_info=True,
                    )
                else:
                    if completed.returncode != 0:
                        # Never silent: an unreclaimed rootfs is ~2.5 GB and
                        # they accumulate for the whole allocation.
                        logger.warning(
                            "enroot remove %s exited %s: %s",
                            container,
                            completed.returncode,
                            (completed.stderr or completed.stdout or "").strip()[-500:],
                        )
        finally:
            self._tmp.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            logger.warning(
                "Could not clean up Pyxis environment",
                exc_info=True,
            )
