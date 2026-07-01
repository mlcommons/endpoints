# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Power-telemetry sidecar lifecycle.

A plain ``subprocess.Popen`` (NOT the ServiceLauncher — we must never abort the
benchmark if the collector misbehaves) that streams a source's stdout into a
trace file. It runs in its own process group so a hung child and any
grandchildren are torn down together. Every method swallows its errors: power
monitoring is strictly best-effort.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import IO

from inference_endpoint.config.schema import PowerConfig
from inference_endpoint.power.sources import ResolvedSource, resolve

logger = logging.getLogger(__name__)

# Grace period after SIGTERM before the sidecar is force-killed. Not a config
# knob — sampling sidecars die fast; tuning this has no practical value.
_STOP_GRACE_S = 5.0
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """preexec_fn: on Linux, request SIGKILL if the parent process dies.

    Backstop for the case where the parent is SIGKILL'd/OOM-killed before
    stop() can run — otherwise a detached sidecar would poll forever.
    """
    if sys.platform == "linux":
        try:
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(
                _PR_SET_PDEATHSIG, signal.SIGKILL
            )
        except Exception:  # noqa: BLE001 — best-effort; never block the child.
            pass


class PowerCollector:
    """Start/stop a sidecar that writes a power trace to ``trace_path``."""

    def __init__(self, cfg: PowerConfig, out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = out_dir
        self.trace_path = out_dir / "power_trace.log"
        self.stderr_path = out_dir / "power_collect.stderr.log"
        self.resolved: ResolvedSource | None = None
        self.status = "disabled"
        self.error: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._trace_fh: IO[bytes] | None = None
        self._stderr_fh: IO[bytes] | None = None

    def start(self) -> None:
        """Launch the sidecar. Records failure in ``status`` instead of raising."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.resolved = resolve(self.cfg)
            env = {**os.environ, **self.cfg.env}
            self._trace_fh = self.trace_path.open("wb")
            self._stderr_fh = self.stderr_path.open("wb")
            self._proc = subprocess.Popen(
                self.resolved.argv,
                stdout=self._trace_fh,
                stderr=self._stderr_fh,
                env=env,
                # Own session/process-group so stop() can signal the whole tree.
                start_new_session=True,
                # Backstop: die if the parent is SIGKILL'd before stop() runs.
                preexec_fn=_die_with_parent,  # noqa: PLW1509 (intentional, Linux)
            )
            self.status = "ok"
            logger.info(
                "power: started sidecar pid=%d -> %s (%s)",
                self._proc.pid,
                self.trace_path,
                " ".join(self.resolved.argv),
            )
        except Exception as e:  # noqa: BLE001 — collector startup must never abort the run.
            self.status = "failed"
            self.error = f"start failed: {e}"
            logger.warning("power: %s", self.error)
            self._close_files()

    def stop(self) -> None:
        """Terminate the sidecar (SIGTERM → grace → SIGKILL). Never raises."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self._signal_group(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=_STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.warning("power: sidecar did not exit; sending SIGKILL")
                    self._signal_group(proc, signal.SIGKILL)
                    try:
                        proc.wait(timeout=_STOP_GRACE_S)
                    except subprocess.TimeoutExpired:
                        self.error = "sidecar would not die"
            rc = proc.returncode
            # nvidia-smi/poll loops are killed by us, so a signal exit is expected.
            if rc not in (0, None, -signal.SIGTERM, -signal.SIGKILL) and not self.error:
                self.error = f"sidecar exited rc={rc}"
        except Exception as e:  # noqa: BLE001 — stop must never abort finalization.
            self.error = f"stop failed: {e}"
            logger.warning("power: %s", self.error)
        finally:
            self._close_files()

    @staticmethod
    def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), sig)
            else:  # Windows: no process groups
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            proc.send_signal(sig)

    def _close_files(self) -> None:
        for fh in (self._trace_fh, self._stderr_fh):
            try:
                if fh is not None:
                    fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._trace_fh = None
        self._stderr_fh = None
