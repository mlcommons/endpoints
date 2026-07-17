# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI flag-surface regression test.

`Settings` carries `@cyclopts.Parameter(name="*")` so its sub-configs flatten to
top-level dotted flags. Misplacing that decorator (e.g. onto a newly inserted config
class above it) silently renames the entire generated surface to `--settings.*` and
double-binds flag names — a regression that only the generated CLI itself can reveal,
so this test inspects real `--help` output (one subprocess, all assertions).
"""

import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_generated_flag_surface():
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference_endpoint.main",
            "benchmark",
            "online",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    # help wraps long flag names mid-token; collapse whitespace to match reliably
    flat = re.sub(r"\s+", "", out.stdout)
    # Settings sub-configs flatten to top level; no --settings.* leak
    assert "--client.num-workers" in flat
    assert "--runtime.min-duration-ms" in flat
    assert "--settings." not in flat
    # warmup (flattened) owns the bare --enabled; early_stopping stays namespaced
    assert "--enabled--warmup" in flat
    assert "--early-stopping.enabled" in flat
    # default-on: --no-early-stopping is the meaningful opt-out
    assert "--no-early-stopping" in flat
