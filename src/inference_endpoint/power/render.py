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

"""Render the Power section into report.txt (mirrors _write_profiling_section)."""

from __future__ import annotations

from typing import Any, TextIO


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.3f}{suffix}"
    return f"{v}{suffix}"


def write_power_section(f: TextIO, power: dict[str, Any]) -> None:
    """Append a human-readable Power section to report.txt."""
    f.write("\n------------------- Power -------------------\n")
    f.write(f"Status: {power.get('status', 'unknown')}\n")
    if power.get("error"):
        f.write(f"Error: {power['error']}\n")
        return

    win = power.get("window", {})
    f.write(
        f"Window: {_fmt(win.get('duration_s'), ' s')} "
        f"(basis: {win.get('basis', '?')})\n"
    )

    totals = power.get("totals", {})
    f.write(f"Total energy: {_fmt(totals.get('energy_j'), ' J')}\n")
    f.write(f"Mean power: {_fmt(totals.get('mean_power_w'), ' W')}\n")
    epot = totals.get("energy_per_output_token_j")
    if epot is not None:
        f.write(f"Energy/output token: {_fmt(epot, ' J')}\n")
    elif totals.get("energy_per_output_token_note"):
        f.write(
            f"Energy/output token: n/a ({totals['energy_per_output_token_note']})\n"
        )

    for src in power.get("sources", []):
        pw = src.get("power_w") or {}
        f.write(
            f"  [{src.get('label')}] energy={_fmt(src.get('energy_j'), ' J')} "
            f"mean={_fmt(pw.get('mean'), ' W')} "
            f"p95={_fmt(pw.get('p95'), ' W')} "
            f"max={_fmt(pw.get('max'), ' W')} "
            f"(n={src.get('sample_count')})\n"
        )

    prov = power.get("provenance", {})
    if prov.get("samples_dropped"):
        f.write(f"  (dropped {prov['samples_dropped']} malformed samples)\n")
