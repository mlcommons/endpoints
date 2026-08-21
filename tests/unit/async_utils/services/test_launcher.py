# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from inference_endpoint.async_utils.services.launcher import ServiceLauncher


@pytest.mark.unit
def test_terminate_sigterms_only_exact_module_match():
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    launcher = ServiceLauncher(MagicMock())
    launcher._procs = [target, bystander]
    launcher._modules = ["svc.metrics_aggregator", "prefix.svc.metrics_aggregator"]
    try:
        launcher.terminate_module("svc.metrics_aggregator")
        assert target.wait(timeout=5.0) == -signal.SIGTERM
        assert bystander.poll() is None, (
            "terminate_module() must match the exact module name; a mere suffix match "
            "must stay alive"
        )
    finally:
        for proc in (target, bystander):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5.0)


@pytest.mark.unit
def test_terminate_ignores_already_exited_proc():
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=5.0)
    launcher = ServiceLauncher(MagicMock())
    launcher._procs = [dead]
    launcher._modules = ["svc.metrics_aggregator"]

    launcher.terminate_module("svc.metrics_aggregator")

    assert dead.returncode == 0
