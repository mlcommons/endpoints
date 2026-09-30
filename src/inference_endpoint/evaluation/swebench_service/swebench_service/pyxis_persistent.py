# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command at a time in a long-lived, allocation-local Pyxis worker."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .pyxis_slurm import (
    SRUN_MAX_ATTEMPTS,
    is_retryable_prelaunch_failure,
    wait_for_prelaunch_retry,
)
from .runner import RunnerError

logger = logging.getLogger(__name__)
_POLL_S = 0.05


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    output: str
    timed_out: bool


class PersistentExecChannel:
    """Serialize commands; never replay them after an uncertain failure."""

    def __init__(
        self,
        protocol_dir: Path,
        command: list[str],
        env: dict[str, str],
        failure_path: Path | None = None,
        launch_timeout_s: float = 30,
        driver_grace_s: float = 30,
        shutdown_grace_s: float = 3,
    ) -> None:
        self.protocol_dir = protocol_dir
        self._command = command
        self._env = env
        self._failure_path = failure_path
        self._launch_timeout_s = launch_timeout_s
        self._driver_grace_s = driver_grace_s
        self._shutdown_grace_s = shutdown_grace_s
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._closed = False
        # A new directory prevents stale requests or replies from being reused.
        protocol_dir.mkdir(mode=0o700)

    def _read_log(self) -> str:
        try:
            with (self.protocol_dir / "server.log").open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 8000))
                return log.read().decode("utf-8", errors="replace")
        except OSError:
            # Startup can fail before the log is created.
            return ""

    def _fail(self, detail: str) -> RunnerError:
        detail += "\n" + self._read_log()
        try:
            if self._failure_path is not None:
                self._failure_path.touch()
        except OSError:
            logger.warning("Could not mark Pyxis infrastructure failure", exc_info=True)
        try:
            self._stop()
        except (OSError, subprocess.SubprocessError):
            logger.warning("Could not stop failed Pyxis worker", exc_info=True)
        return RunnerError(f"persistent Pyxis infrastructure failure: {detail}")

    def _wait_for(self, path: Path, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not path.exists():
            if self._process is None or self._process.poll() is not None:
                raise RunnerError("worker is not running; execution is uncertain")
            if time.monotonic() >= deadline:
                raise RunnerError(
                    "worker exceeded its driver deadline; execution is uncertain"
                )
            time.sleep(_POLL_S)

    def start(self) -> None:
        with self._lock:
            if self._closed or self._process is not None:
                raise RunnerError("persistent Pyxis channel already started or closed")
            log_path = self.protocol_dir / "server.log"
            ready = self.protocol_dir / "ready"
            for attempt in range(1, SRUN_MAX_ATTEMPTS + 1):
                try:
                    with log_path.open("wb") as log:
                        self._process = subprocess.Popen(
                            self._command,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=self._env,
                        )
                    self._wait_for(ready, self._launch_timeout_s)
                    return
                except (OSError, RunnerError) as exc:
                    # No requests can be published while start holds the lock.
                    # Never relaunch a ready worker or a possibly running step.
                    if (
                        attempt < SRUN_MAX_ATTEMPTS
                        and self._process is not None
                        and self._process.poll() not in (None, 0)
                        and not ready.exists()
                        and is_retryable_prelaunch_failure("pending", self._read_log())
                    ):
                        wait_for_prelaunch_retry(attempt)
                        continue
                    raise self._fail(f"worker did not become ready: {exc}") from exc

    def _read_completion(self, request: Path) -> CommandResult:
        try:
            returncode, timed_out, size = map(
                int, (request / "complete").read_text().split()
            )
        except ValueError as exc:
            raise RunnerError("invalid Pyxis completion marker") from exc
        if not 0 <= returncode <= 255 or timed_out not in (0, 1) or size < 0:
            raise RunnerError("invalid Pyxis completion marker")
        output = (request / "output").read_bytes()
        if len(output) != size:
            raise RunnerError("incomplete Pyxis command output")
        return CommandResult(
            returncode, output.decode("utf-8", errors="replace"), bool(timed_out)
        )

    def execute(self, *, command: str, cwd: str, timeout_s: int) -> CommandResult:
        if timeout_s <= 0:
            raise ValueError("Pyxis command timeout must be positive")
        with self._lock:
            if self._closed:
                raise RunnerError("persistent Pyxis channel is closed")
            try:
                if self._process is None or self._process.poll() is not None:
                    raise RunnerError("worker is not running")
                staging = self.protocol_dir / ".request"
                staging.mkdir()
                for name, value in (
                    ("command", command),
                    ("cwd", cwd),
                    ("timeout", str(timeout_s)),
                ):
                    (staging / name).write_text(value)
                os.replace(staging, self.protocol_dir / "request")
                running = self.protocol_dir / "running"
                self._wait_for(running / "complete", timeout_s + self._driver_grace_s)
                result = self._read_completion(running)
                shutil.rmtree(running)
                return result
            except (OSError, RunnerError) as exc:
                raise self._fail(str(exc)) from exc

    def _stop(self) -> None:
        self._closed = True
        try:
            (self.protocol_dir / "stop").touch()
        finally:
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=self._shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=self._shutdown_grace_s)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=self._shutdown_grace_s)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stop()
