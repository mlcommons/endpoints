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

"""Shared fixtures + summary table for E2E performance tests.

Tests in this directory inject the ``record_result`` fixture and call it
once per parameterization. After the session finishes,
:func:`pytest_terminal_summary` prints a formatted table of every recorded
row — handy when running roofline + low-QPS together.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import pytest


class _Collected:
    """Module-level singleton holding rows recorded during the session."""

    rows: list[dict[str, Any]] = []


@pytest.fixture
def record_result():
    """Record a result row that will appear in the end-of-session summary.

    Pass keyword fields you want in the table — anything missing renders as
    ``—`` in the output.

    Usage::

        def test_foo(record_result):
            record_result(
                "max_throughput", stream=False,
                qps=44426.0, total=2_000_000, elapsed=45.02, failed=0,
            )
    """

    def _record(label: str, **fields: Any) -> None:
        _Collected.rows.append({"label": label, **fields})

    return _record


# -----------------------------------------------------------------------------
# Host info + table rendering
# -----------------------------------------------------------------------------


def _host_info() -> dict[str, str]:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        # CPU model is informational; missing /proc/cpuinfo (non-Linux,
        # restricted container) just leaves it as "unknown".
        pass
    cores = os.cpu_count() or 0
    return {
        "host": platform.node(),
        "arch": platform.machine(),
        "cpu": cpu,
        "cores": str(cores) if cores else "?",
    }


def _fmt_cell(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "stream":
        return "on " if value else "off"
    # Conversions go through float() first so numeric strings ("100.0")
    # don't crash int(). Any conversion failure falls back to str(value)
    # so the end-of-session summary never blows up the pytest run.
    try:
        if kind == "qps":
            v = float(value)
            return f"{v:>9,.0f}" if v >= 100 else f"{v:>9.2f}"
        if kind == "total":
            return f"{int(float(value)):>10,}"
        if kind == "elapsed":
            return f"{float(value):>7.2f}s"
        if kind == "failed":
            return f"{int(float(value)):>4}"
    except (TypeError, ValueError):
        return str(value)
    return str(value)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    rows = _Collected.rows
    if not rows:
        return

    tr = terminalreporter
    tr.write_sep("=", "E2E Performance Summary")

    info = _host_info()
    tr.write_line(
        f"Host:  {info['host']}    Arch: {info['arch']}    Cores: {info['cores']}"
    )
    tr.write_line(f"CPU:   {info['cpu']}")
    tr.write_line("")

    headers = ["Test", "Stream", "QPS", "Total", "Elapsed", "Failed"]
    kinds = ["label", "stream", "qps", "total", "elapsed", "failed"]
    keys = ["label", "stream", "qps", "total", "elapsed", "failed"]

    body = [
        [_fmt_cell(r.get(k), kind) for k, kind in zip(keys, kinds, strict=False)]
        for r in rows
    ]

    widths = [
        max(len(h), max((len(row[i]) for row in body), default=0))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  ".join("-" * w for w in widths)

    tr.write_line(fmt.format(*headers))
    tr.write_line(sep)
    for row in body:
        tr.write_line(fmt.format(*row))
