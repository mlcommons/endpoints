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

from __future__ import annotations

from statistics import pstdev

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.window import percentile_lower


def rule_fixed_budget(
    series: list[SuperPassRollup], k: int, warmup: int = 1
) -> tuple[int, int]:
    n = len(series)
    if warmup >= n:
        raise ValueError(f"warmup {warmup} >= {n} super-passes")
    if k < 1:
        raise ValueError("k must be >= 1")
    return (warmup, min(warmup + k, n))


def cov(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    if m == 0:
        return 0.0
    return pstdev(values) / abs(m)


def rule_cov_converged(
    series: list[SuperPassRollup],
    window: int = 3,
    cov_bound: float = 0.05,
    warmup: int = 1,
    percentiles=(0.5, 0.99),
) -> tuple[int, int] | None:
    n = len(series)

    def _pct(sp: SuperPassRollup, source: str, p: float) -> float:
        vals = sorted(getattr(sp, source))
        return percentile_lower(vals, p) if vals else 0.0

    for sp_end in range(warmup + window, n + 1):
        trailing = series[sp_end - window : sp_end]
        converged = True
        for source in ("ttft_ns", "latency_ns"):
            for p in percentiles:
                across = [_pct(sp, source, p) for sp in trailing]
                if cov(across) >= cov_bound:
                    converged = False
                    break
            if not converged:
                break
        if converged:
            return (warmup, sp_end)
    return None
