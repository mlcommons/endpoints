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

"""Integration tests for the aggregator subprocess's signal handlers.

The aggregator's INTERRUPTED-snapshot path is the only mechanism that
produces a ``state=interrupted`` ``final_snapshot.json``, and the
SIGINT-no-op path is the only thing standing between an interactive
^C and silent sample loss. These tests spawn a real subprocess and
exercise both paths end-to-end — the unit tests in ``test_publisher.py``
cover the API surface, not the signal wiring.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest


def _spawn_aggregator(
    socket_dir: Path,
    output_dir: Path,
    *,
    socket_name: str,
    metrics_socket: str,
) -> subprocess.Popen:
    """Launch the metrics-aggregator subprocess in its own process group.

    Own process group is critical for the SIGINT-no-op test — sending
    SIGINT to just this group (not the test runner's group) emulates a
    user Ctrl-C in the foreground process group of the subprocess and
    not the test runner.

    Readiness is gated on the ``<output_dir>/.ready`` marker the aggregator
    touches once its signal handlers are registered.
    """
    cmd = [
        sys.executable,
        "-m",
        "inference_endpoint.async_utils.services.metrics_aggregator",
        "--socket-dir",
        str(socket_dir),
        "--socket-name",
        socket_name,
        "--metrics-socket",
        metrics_socket,
        "--metrics-output-dir",
        str(output_dir),
        # Required by the entrypoint, but inert here: no tokenizer is
        # configured (so no live tokenization) and the run is signalled
        # rather than ENDED, so the drain budget is never reached.
        "--drain-timeout",
        "5",
        "--tokenizer-workers",
        "0",
    ]
    return subprocess.Popen(
        cmd,
        # New process group so we can signal it without disturbing the
        # test runner.
        preexec_fn=os.setsid,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_for_file(path: Path, timeout: float) -> bool:
    """Poll for ``path`` existing within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.integration
class TestAggregatorSignalHandling:
    def test_sigterm_writes_interrupted_final_snapshot(self, tmp_path: Path):
        """SIGTERM to the aggregator MUST produce ``final_snapshot.json``
        with ``state == "interrupted"``. This is the only path that
        produces an INTERRUPTED snapshot — without it, a parent
        ``ServiceLauncher.kill_all`` would leave the Report consumer with
        no final-snapshot file at all.
        """
        socket_dir = tmp_path / "sockets"
        socket_dir.mkdir()
        output_dir = tmp_path / "output"
        # The parent owns directory setup — the aggregator subprocess
        # fail-fasts (SystemExit) on a missing output dir to surface
        # contract violations in its own stderr instead of crashing
        # later on the atomic-write path. Mirror that contract here.
        output_dir.mkdir()
        # Use a unique socket name per test to avoid collisions if a
        # previous test run left an IPC file behind.
        suffix = uuid.uuid4().hex[:8]
        ready_file = output_dir / ".ready"
        proc = _spawn_aggregator(
            socket_dir,
            output_dir,
            socket_name=f"events_{suffix}",
            metrics_socket=f"metrics_{suffix}",
        )
        try:
            # Poll for the ready sentinel instead of sleeping a fixed amount:
            # on network-mounted filesystems (e.g. Lustre) Python import can
            # take several seconds, so a fixed sleep races with signal-handler
            # registration. The aggregator touches <output_dir>/.ready only after
            # loop.add_signal_handler returns, so this is an exact gate.
            ready = _wait_for_file(ready_file, timeout=30.0)
            assert ready, (
                f"aggregator did not become ready within 30 s — "
                f"stderr: {(proc.stderr.read() if proc.stderr else b'').decode()[-2000:]}"
            )
            assert (
                proc.poll() is None
            ), f"aggregator died early: stderr={(proc.stderr.read() if proc.stderr else b'').decode()}"

            # SIGTERM the process group → triggers _signal_finalize.
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10.0)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5.0)

        # The signal handler MUST have written final_snapshot.json
        # before the subprocess exited.
        final = output_dir / "final_snapshot.json"
        assert final.exists(), (
            f"SIGTERM did not produce final_snapshot.json — "
            f"stderr: {(proc.stderr.read() if proc.stderr else b"").decode()[-2000:]}"
        )
        decoded = json.loads(final.read_bytes())
        assert decoded["state"] == "interrupted"

    def test_sigint_does_not_finalize_aggregator(self, tmp_path: Path):
        """SIGINT to the aggregator MUST NOT trigger publish_final.

        On an interactive ^C, the OS sends SIGINT to the whole foreground
        process group; both parent and child receive it. If the
        aggregator finalized eagerly here, samples that completed during
        the parent's clean-shutdown window would never reach the file.
        The aggregator's contract is: SIGINT is a no-op, the parent's
        ENDED-driven path is authoritative.

        Verification: send SIGINT, wait long enough for any naive
        signal-driven write to have happened, then assert the file did
        NOT appear and the subprocess is still alive.
        """
        socket_dir = tmp_path / "sockets"
        socket_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()  # parent owns dir setup (see sibling test)
        suffix = uuid.uuid4().hex[:8]
        ready_file = output_dir / ".ready"
        proc = _spawn_aggregator(
            socket_dir,
            output_dir,
            socket_name=f"events_{suffix}",
            metrics_socket=f"metrics_{suffix}",
        )
        try:
            ready = _wait_for_file(ready_file, timeout=30.0)
            assert ready, "aggregator did not become ready within 30 s"
            assert proc.poll() is None, "aggregator died before signal-handler test"

            os.killpg(proc.pid, signal.SIGINT)
            # Wait a beat — if SIGINT were naively driving publish_final,
            # the file would appear well within this window.
            time.sleep(1.0)

            final = output_dir / "final_snapshot.json"
            assert not final.exists(), (
                "SIGINT must NOT trigger publish_final; the parent's "
                "ENDED-driven path is authoritative on interactive ^C"
            )
            assert proc.poll() is None, (
                "aggregator must remain alive after SIGINT; only "
                "SIGTERM (parent kill) or ENDED should finalize it"
            )
        finally:
            # Use SIGTERM (which is the correct shutdown path) for
            # cleanup, then SIGKILL as belt-and-suspenders.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5.0)
