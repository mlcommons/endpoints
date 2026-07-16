# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI flag-surface regression tests.

`Settings` carries `@cyclopts.Parameter(name="*")` so its sub-configs flatten to
top-level dotted flags (`--client.num-workers`, `--runtime.min-duration-ms`, ...).
Misplacing that decorator (e.g. onto a newly inserted config class above it) silently
renames the entire generated surface to `--settings.*` and can double-bind flag names —
these tests pin the surface so decorator placement can't regress unnoticed.
"""

import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def _online_help() -> str:
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
    # help output wraps long flag names mid-token; collapse whitespace to match reliably
    return re.sub(r"\s+", "", out.stdout)


def test_settings_subconfigs_flatten_to_top_level():
    flat = _online_help()
    assert "--client.num-workers" in flat
    assert "--runtime.min-duration-ms" in flat
    assert "--early-stopping" in flat  # the ES opt-in alias
    # the settings prefix must not leak into the generated surface
    assert "--settings." not in flat


def test_early_stopping_does_not_shadow_warmup_enabled():
    # WarmupConfig is itself flattened: its `enabled` owns the bare `--enabled`
    # (aliased `--warmup`). EarlyStoppingConfig must NOT also flatten, or the two
    # would double-bind `--enabled`; it stays namespaced under its field name.
    flat = _online_help()
    assert "--enabled--warmup" in flat  # warmup's pair, whitespace-collapsed
    assert "--early-stopping.enabled" in flat
    assert "--no-early-stopping" in flat
