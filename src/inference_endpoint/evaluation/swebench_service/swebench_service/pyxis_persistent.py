# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local command transport for a long-lived Pyxis step."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runner import RunnerError

logger = logging.getLogger(__name__)
_PERSISTENT_ROOT = "/tmp/.mlperf_persistent_exec"
_PERSISTENT_POLL_S = 0.05


def _read_step_status(status_path: Path) -> str:
    try:
        return status_path.read_text().strip()
    except OSError as exc:
        return f"<unreadable: {exc}>"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _read_sized_file(
    path: Path,
    size: int,
    deadline: float,
    *,
    read_bytes: Callable[[Path], bytes] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Read an atomically published response, retrying stale partial views."""
    reader = read_bytes or (lambda target: target.read_bytes())
    while True:
        try:
            data = reader(path)
        except OSError:
            if monotonic() >= deadline:
                raise RunnerError(
                    f"persistent Pyxis response {path} is missing"
                ) from None
            sleep(_PERSISTENT_POLL_S)
            continue
        if len(data) == size:
            return data
        if len(data) > size:
            raise RunnerError(
                f"persistent Pyxis response {path} grew past manifest size "
                f"({len(data)} > {size})"
            )
        if monotonic() >= deadline:
            raise RunnerError(
                f"persistent Pyxis response {path} remained partial "
                f"({len(data)} of {size} bytes)"
            )
        sleep(_PERSISTENT_POLL_S)


def _persistent_response_digest(
    *,
    secret: str,
    nonce: str,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    timed_out: bool,
) -> str:
    prefix = "\0".join(
        (
            secret,
            nonce,
            str(returncode),
            str(len(stdout)),
            str(len(stderr)),
            "1" if timed_out else "0",
        )
    ).encode()
    payload = prefix + b"\0" + stdout + b"\0" + stderr + b"\0" + secret.encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class PersistentExecChannel:
    """Serialize commands without replaying requests after uncertain execution."""

    def __init__(
        self,
        protocol_dir: Path,
        command_factory: Callable[[str, str], list[str]],
        env: dict[str, str],
        failure_path: Path | None = None,
        launch_timeout_s: float = 30,
        driver_grace_s: float = 30,
        shutdown_grace_s: float = 3,
    ) -> None:
        self.protocol_dir = protocol_dir
        self._requests_dir = protocol_dir / "requests"
        self._command_factory = command_factory
        self._env = env
        self._failure_path = failure_path
        self._launch_timeout_s = launch_timeout_s
        self._driver_grace_s = driver_grace_s
        self._shutdown_grace_s = shutdown_grace_s
        self._secret = secrets.token_hex(32)
        self._process: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._lock = threading.Lock()
        self._closed = False
        self.commands = 0
        self._requests_dir.mkdir(parents=True, mode=0o700)

    def _fail(self, detail: str) -> RunnerError:
        if self._failure_path is not None:
            self._failure_path.touch()
        return RunnerError(detail)

    def start(self) -> None:
        generation = uuid.uuid4().hex
        try:
            self._log = (self.protocol_dir / "server.log").open("wb")
            self._process = subprocess.Popen(
                self._command_factory(generation, self._secret),
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=self._env,
            )
            deadline = time.monotonic() + self._launch_timeout_s
            while True:
                if _read_step_status(self.protocol_dir / "ready") == generation:
                    return
                if self._process.poll() is not None or time.monotonic() >= deadline:
                    with (self.protocol_dir / "server.log").open("rb") as log:
                        log.seek(max(0, log.seek(0, os.SEEK_END) - 2000))
                        detail = log.read().decode("utf-8", errors="replace")
                    raise self._fail(
                        f"persistent Pyxis server did not become ready: {detail}"
                    )
                time.sleep(_PERSISTENT_POLL_S)
        except (OSError, subprocess.SubprocessError) as exc:
            raise self._fail("could not start persistent Pyxis server") from exc
        finally:
            # The environment owns cleanup even when startup fails.
            if self._log is not None:
                self._log.close()
                self._log = None

    def _read_completion(
        self, request_dir: Path, deadline: float
    ) -> CommandResult | None:
        complete_path = request_dir / "complete"
        try:
            manifest = complete_path.read_text().strip().split()
        except OSError:
            return None
        if (
            len(manifest) != 5
            or not all(re.fullmatch(r"-?[0-9]+", value) for value in manifest[:4])
            or re.fullmatch(r"[0-9a-f]{64}", manifest[4]) is None
        ):
            if time.monotonic() >= deadline:
                raise RunnerError(
                    f"persistent Pyxis completion marker is invalid: {manifest!r}"
                )
            return None
        returncode, stdout_size, stderr_size, timed_out = map(int, manifest[:4])
        digest = manifest[4]
        if (
            not 0 <= returncode <= 255
            or stdout_size < 0
            or stderr_size < 0
            or timed_out not in {0, 1}
        ):
            raise RunnerError("persistent Pyxis completion marker has invalid fields")
        stdout = _read_sized_file(request_dir / "stdout", stdout_size, deadline)
        stderr = _read_sized_file(request_dir / "stderr", stderr_size, deadline)
        expected_digest = _persistent_response_digest(
            secret=self._secret,
            nonce=request_dir.name,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=bool(timed_out),
        )
        if not hmac.compare_digest(digest, expected_digest):
            raise RunnerError("persistent Pyxis completion digest did not verify")
        return CommandResult(
            returncode=returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=bool(timed_out),
        )

    def execute(self, *, command: str, cwd: str, timeout_s: int) -> CommandResult:
        with self._lock:
            if self._closed:
                raise self._fail("persistent Pyxis channel is closed")
            try:
                if self._process is None or self._process.poll() is not None:
                    raise self._fail("persistent Pyxis server is not running")
                nonce = uuid.uuid4().hex
                temporary = self._requests_dir / f".{nonce}.tmp"
                request_dir = self._requests_dir / nonce
                temporary.mkdir(mode=0o700)
                for name, value in (
                    ("command", command),
                    ("cwd", cwd),
                    ("timeout", str(timeout_s)),
                    ("status", "pending\n"),
                ):
                    (temporary / name).write_text(value)
                os.replace(temporary, request_dir)
                self.commands += 1
                deadline = time.monotonic() + timeout_s + self._driver_grace_s
                while True:
                    result = self._read_completion(request_dir, deadline)
                    if result is not None:
                        try:
                            shutil.rmtree(request_dir)
                        except OSError:
                            # Nonces are never reused; environment cleanup removes leftovers.
                            logger.debug(
                                "Could not remove completed request %s",
                                request_dir,
                                exc_info=True,
                            )
                        return result
                    if self._process.poll() is not None:
                        raise self._fail(
                            "persistent Pyxis server died while a request was active; execution is uncertain"
                        )
                    if time.monotonic() >= deadline:
                        raise self._fail(
                            "persistent Pyxis request exceeded its driver deadline; execution is uncertain"
                        )
                    time.sleep(_PERSISTENT_POLL_S)
            except (OSError, RunnerError) as exc:
                # A dead local srun does not prove its remote command stopped.
                # Poison this channel; never restart or replay an accepted request.
                failure = self._fail(f"persistent Pyxis infrastructure failure: {exc}")
                try:
                    self._stop()
                except (OSError, subprocess.SubprocessError):
                    logger.warning("Could not stop failed Pyxis worker", exc_info=True)
                raise failure from exc

    def _stop(self) -> None:
        self._closed = True
        try:
            _atomic_write_text(self.protocol_dir / "stop", "stop\n")
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
