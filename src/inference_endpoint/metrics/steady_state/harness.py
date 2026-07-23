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

from dataclasses import dataclass

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import (
    rule_cov_converged,
    rule_fixed_budget,
)
from inference_endpoint.metrics.steady_state.window import (
    WindowMetrics,
    windowed_metrics,
)


@dataclass(frozen=True, slots=True)
class RuleScore:
    name: str
    region: tuple[int, int] | None
    super_passes: int
    metrics: WindowMetrics | None
    qps_rel_err: float | None
    ttft_p99_rel_err: float | None


def asymptote(series: list[SuperPassRollup], warmup: int = 1, **kw) -> WindowMetrics:
    return windowed_metrics(series, warmup, len(series), **kw)


def _rel_err(est: float, ref: float) -> float:
    if ref == 0:
        return 0.0
    return abs(est - ref) / abs(ref)


def score_rule(
    name: str,
    region: tuple[int, int] | None,
    series: list[SuperPassRollup],
    ref: WindowMetrics,
    **kw,
) -> RuleScore:
    if region is None:
        return RuleScore(name, None, 0, None, None, None)
    m = windowed_metrics(series, region[0], region[1], **kw)
    ref_p99 = ref.ttft.get(0.99)
    est_p99 = m.ttft.get(0.99)
    p99_err = (
        _rel_err(est_p99, ref_p99)
        if (ref_p99 is not None and est_p99 is not None)
        else None
    )
    return RuleScore(
        name=name,
        region=region,
        super_passes=region[1] - region[0],
        metrics=m,
        qps_rel_err=_rel_err(m.qps, ref.qps),
        ttft_p99_rel_err=p99_err,
    )


def sweep(
    series: list[SuperPassRollup],
    k: int = 3,
    cov_window: int = 3,
    cov_bound: float = 0.05,
    warmup: int = 1,
    **kw,
) -> tuple[WindowMetrics, list[RuleScore]]:
    ref = asymptote(series, warmup=warmup, **kw)
    region_a = rule_fixed_budget(series, k=k, warmup=warmup)
    region_b = rule_cov_converged(
        series, window=cov_window, cov_bound=cov_bound, warmup=warmup
    )
    return ref, [
        score_rule("fixed_budget", region_a, series, ref, **kw),
        score_rule("cov_converged", region_b, series, ref, **kw),
    ]
