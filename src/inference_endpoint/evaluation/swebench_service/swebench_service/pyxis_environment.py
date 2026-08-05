# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .runner import RunnerError

_SAFE_SRUN_ENV = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
)


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
    command = ["srun", "--overlap", f"--jobid={job_id}", "-N1", "-n1"]
    if node := os.environ.get("SLURMD_NODENAME", "").strip():
        command.append(f"--nodelist={node}")
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


def resolve_image(image_registry: str, instance_id: str) -> str:
    if Path(instance_id).name != instance_id or instance_id in {".", ".."}:
        raise RunnerError(f"invalid SWE-bench instance ID: {instance_id}")
    image_registry = image_registry.rstrip("/")
    if "#" not in image_registry:
        host, separator, repository = image_registry.partition("/")
        if not separator:
            raise RunnerError("Pyxis image registry must include a repository")
        image_registry = f"{host}#{repository}"
    return f"{image_registry}/sweb.eval.arm64.{instance_id}:v4.1.0-arm64"


class PyxisEnvironmentConfig(BaseModel):
    image: str | Path
    run_id: str
    cwd: str = "/testbed"
    env: dict[str, str] = Field(default_factory=dict)
    timeout: int = 30
    interpreter: list[str] = Field(default_factory=lambda: ["bash", "-c"])


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
        command = build_srun_command(
            image=self.config.image,
            name=self.name,
            mounts=[(self._tmp_dir, "/tmp")],
            workdir=self.config.cwd,
            argv=["true"],
        )
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=safe_srun_env(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.cleanup()
            raise RunnerError(
                f"failed to start Pyxis container for {self.config.image}"
            ) from exc

    def execute(
        self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None
    ) -> dict[str, Any]:
        command = action.get("command", "")
        argv = ["env"]
        argv.extend(f"{key}={value}" for key, value in self.config.env.items())
        argv.extend([*self.config.interpreter, command])
        srun_command = build_srun_command(
            image=None,
            name=self.name,
            mounts=[(self._tmp_dir, "/tmp")],
            workdir=cwd or self.config.cwd,
            argv=argv,
        )
        output: dict[str, Any]
        try:
            result = subprocess.run(
                srun_command,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=safe_srun_env(),
            )
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as exc:
            raw_output = getattr(exc, "output", None)
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output or "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                },
            }
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if (
            lines
            and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
            and output["returncode"] == 0
        ):
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
            **self.config.model_dump(),
            **platform.uname()._asdict(),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
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
            if os.environ.get("SLURM_JOB_ID", "").strip():
                try:
                    subprocess.run(
                        build_srun_command(
                            argv=["enroot", "remove", "-f", f"pyxis_{self.name}"]
                        ),
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=safe_srun_env(),
                    )
                except (OSError, RunnerError, subprocess.SubprocessError):
                    logging.getLogger(__name__).warning(
                        "Could not remove Pyxis container %s",
                        self.name,
                        exc_info=True,
                    )
        finally:
            self._tmp.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
