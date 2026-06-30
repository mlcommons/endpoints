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

"""Slice a power trace to the measurement window and integrate it into energy.

Produces the ``power.json`` payload: per-series watt statistics, energy over the
window, and an energy-per-output-token figure that is emitted **only** when the
token count and the energy share the same window (otherwise the denominators
would silently mix — the single biggest correctness trap here).
"""

from __future__ import annotations

from typing import Any

from inference_endpoint.power.parse import PowerSample, parse_trace
from inference_endpoint.power.sources import ResolvedSource


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(n * 0.95))],
        "max": s[-1],
        "min": s[0],
    }


def _energy_j(samples: list[PowerSample], value_kind: str) -> float | None:
    """Energy in joules over the samples (already window-filtered, time-sorted).

    ``power_w``: trapezoid integral of watts over seconds.
    ``energy_j``: cumulative counter delta (last - first).
    Returns None when there is too little data to bracket the window.
    """
    if len(samples) < 2:
        return None
    if value_kind == "energy_j":
        delta = samples[-1].value - samples[0].value
        return delta if delta >= 0 else None  # counter reset → unusable
    total = 0.0
    for a, b in zip(samples, samples[1:], strict=False):
        dt = b.ts_epoch_s - a.ts_epoch_s
        if dt <= 0:
            continue
        total += 0.5 * (a.value + b.value) * dt  # ponytail: no edge interpolation
    return total


def build_power_report(
    *,
    resolved: ResolvedSource,
    trace_path: Any,
    window_start_epoch_s: float,
    window_end_epoch_s: float,
    output_tokens: int | None,
    token_window_basis: str,
    consistent_with_window: bool,
    collector_status: str,
    collector_error: str | None,
    interval_s: float,
) -> dict[str, Any]:
    """Assemble the ``power.json`` dict. Pure + total-failure tolerant."""
    parsed = parse_trace(trace_path, resolved)
    in_window = sorted(
        (
            s
            for s in parsed.samples
            if window_start_epoch_s <= s.ts_epoch_s <= window_end_epoch_s
        ),
        key=lambda s: s.ts_epoch_s,
    )

    by_label: dict[str, list[PowerSample]] = {}
    for s in in_window:
        by_label.setdefault(s.label, []).append(s)

    sources: list[dict[str, Any]] = []
    total_energy_j = 0.0
    have_energy = False
    for label, series in sorted(by_label.items()):
        energy = _energy_j(series, resolved.value_kind)
        watts = [s.value for s in series] if resolved.value_kind == "power_w" else None
        entry: dict[str, Any] = {
            "label": label,
            "value_kind": resolved.value_kind,
            "sample_count": len(series),
            "energy_j": energy,
        }
        if watts is not None:
            entry["power_w"] = _stats(watts)
        sources.append(entry)
        if energy is not None:
            total_energy_j += energy
            have_energy = True

    duration_s = max(0.0, window_end_epoch_s - window_start_epoch_s)
    # Status precedence: collector failure > no data > partial (dropped rows or
    # a nonzero sidecar exit that still produced some samples) > ok.
    if collector_status == "failed":
        status = "failed"
    elif not in_window:
        status = "no_data"
    elif parsed.dropped > 0 or collector_error:
        status = "partial"
    else:
        status = "ok"

    energy_total: float | None = total_energy_j if have_energy else None
    epot: float | None = None
    epot_note: str | None = None
    if energy_total is None:
        epot_note = "no energy integrated over window"
    elif not output_tokens:
        epot_note = "output token count unavailable"
    elif not consistent_with_window:
        epot_note = (
            "token count spans a different window than energy "
            f"(token basis: {token_window_basis}); J/token suppressed"
        )
    else:
        epot = energy_total / output_tokens

    return {
        "schema_version": "1.0",
        "status": status,
        "window": {
            "start_epoch_s": window_start_epoch_s,
            "end_epoch_s": window_end_epoch_s,
            "duration_s": duration_s,
            "basis": "performance_phase",
        },
        "totals": {
            "energy_j": energy_total,
            "mean_power_w": (energy_total / duration_s)
            if (energy_total is not None and duration_s > 0)
            else None,
            "output_tokens": output_tokens,
            "token_window_basis": token_window_basis,
            "consistent_with_window": consistent_with_window,
            "energy_per_output_token_j": epot,
            "energy_per_output_token_note": epot_note,
        },
        "sources": sources,
        "provenance": {
            "command": resolved.argv,
            "interval_s": interval_s,
            "value_kind": resolved.value_kind,
            "samples_parsed": len(parsed.samples),
            "samples_in_window": len(in_window),
            "samples_dropped": parsed.dropped,
            "collector_status": collector_status,
            "collector_error": collector_error,
        },
    }


def disabled_report(reason: str) -> dict[str, Any]:
    """Minimal payload when monitoring could not even start."""
    return {"schema_version": "1.0", "status": "failed", "error": reason}
