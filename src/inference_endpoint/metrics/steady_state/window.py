# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

from inference_endpoint.metrics.early_stopping import CONFIDENCE, es_percentile_estimate
from inference_endpoint.metrics.steady_state.series import SuperPassRollup


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    sp_start: int
    sp_end: int
    n_samples: int
    issue_span_ns: int
    qps: float
    token_tps: float | None
    ttft: dict[float, float]
    tpot: dict[float, float] | None
    latency: dict[float, float]
    valid: dict[str, bool]


def percentile_lower(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty series")
    return sorted_values[int(p * (n - 1))]


def _grid(values: list[float], percentiles) -> dict[float, float]:
    s = sorted(values)
    return {p: percentile_lower(s, p) for p in percentiles}


def windowed_metrics(
    series: list[SuperPassRollup],
    sp_start: int,
    sp_end: int,
    percentiles=(0.5, 0.9, 0.99),
    es_percentile: float = 0.99,
    confidence: float = CONFIDENCE,
) -> WindowMetrics:
    if not 0 <= sp_start < sp_end <= len(series):
        raise ValueError(
            f"bad window [{sp_start},{sp_end}) over {len(series)} super-passes"
        )
    window = series[sp_start:sp_end]
    ttft: list[float] = []
    tpot: list[float] = []
    latency: list[float] = []
    out_tokens = 0
    n_samples = 0
    for sp in window:
        ttft.extend(sp.ttft_ns)
        tpot.extend(sp.tpot_ns)
        latency.extend(sp.latency_ns)
        out_tokens += sp.out_tokens
        n_samples += sp.n_issued
    issue_span_ns = window[-1].last_issue_ns - window[0].first_issue_ns
    span_s = issue_span_ns / 1e9
    qps = n_samples / span_s if span_s > 0 else 0.0
    token_tps = (out_tokens / span_s) if (out_tokens and span_s > 0) else None

    def _valid(vals: list[float]) -> bool:
        if not vals:
            return False
        return (
            es_percentile_estimate(sorted(vals), es_percentile, confidence).estimate
            is not None
        )

    return WindowMetrics(
        sp_start=sp_start,
        sp_end=sp_end,
        n_samples=n_samples,
        issue_span_ns=issue_span_ns,
        qps=qps,
        token_tps=token_tps,
        ttft=_grid(ttft, percentiles) if ttft else {},
        tpot=_grid(tpot, percentiles) if tpot else None,
        latency=_grid(latency, percentiles) if latency else {},
        valid={
            "ttft": _valid(ttft),
            "latency": _valid(latency),
            "tpot": _valid(tpot),
        },
    )
