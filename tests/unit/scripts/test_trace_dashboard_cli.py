# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the /proc-based helpers in ``scripts/trace_dashboard.py``.

The dashboard CLI is a script, not a package module, so it is loaded via
importlib. Both helpers observe live ``/proc``, so the tests exercise them
against a real child process tree and a real loopback connection rather than
synthetic fixtures.
"""

import importlib.util
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "trace_dashboard_cli", Path("scripts/trace_dashboard.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


cli = _load_cli()


class TestBenchmarkPids:
    def test_walks_descendants_via_stat_ppids(self) -> None:
        # `sleep & wait` makes the shell (child) fork a grandchild sleep, so
        # the BFS over /proc/*/stat ppids must surface BOTH levels.
        child = subprocess.Popen(["sh", "-c", "sleep 30 & wait"])
        try:
            time.sleep(0.3)  # let the shell fork the backgrounded sleep
            pids = cli._benchmark_pids(child.pid)
            assert child.pid in pids
            assert len(pids) >= 2, f"expected child + grandchild, got {pids}"
        finally:
            child.terminate()
            child.wait()

    def test_excludes_the_caller(self) -> None:
        # The dashboard must never count its own connections — _benchmark_pids
        # drops os.getpid() even when it is the root.
        assert os.getpid() not in cli._benchmark_pids(os.getpid())

    def test_unknown_pid_yields_no_tree(self) -> None:
        pids = cli._benchmark_pids(2**31 - 1)  # no such process
        assert os.getpid() not in pids


class TestCountEstablishedTcp:
    def test_counts_then_drops_on_close(self) -> None:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        client = socket.create_connection(("127.0.0.1", port))
        accepted, _ = srv.accept()
        try:
            # Both ends of the loopback conn live in this process as
            # ESTABLISHED sockets; the LISTEN socket is not counted.
            n = cli._count_established_tcp([os.getpid()])
            assert n >= 2
        finally:
            client.close()
            accepted.close()
            srv.close()
        # Closing the fds removes them from the count (TIME_WAIT holds no fd).
        assert cli._count_established_tcp([os.getpid()]) < n

    def test_empty_pid_list_is_zero(self) -> None:
        assert cli._count_established_tcp([]) == 0
