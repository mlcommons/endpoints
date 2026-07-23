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

import pytest
from inference_endpoint.metrics.steady_state.harness import asymptote, sweep
from inference_endpoint.metrics.steady_state.series import SuperPassRollup


def _sp(index, ttft, lat, n=2, span_ns=1_000_000_000):
    return SuperPassRollup(
        index=index,
        n_issued=n,
        first_issue_ns=index * span_ns,
        last_issue_ns=index * span_ns + span_ns,
        ttft_ns=list(ttft),
        latency_ns=list(lat),
    )


@pytest.mark.unit
def test_asymptote_covers_all_after_warmup():
    series = [_sp(i, [10.0, 10.0], [50.0, 50.0]) for i in range(5)]
    a = asymptote(series, warmup=1)
    assert a.sp_start == 1 and a.sp_end == 5


@pytest.mark.unit
def test_sweep_returns_two_rule_scores():
    series = [_sp(0, [100.0, 100.0], [500.0, 500.0])]  # noisy warmup
    for i in range(1, 6):
        series.append(_sp(i, [10.0, 10.0], [50.0, 50.0]))  # stable
    ref, scores = sweep(series, k=3, cov_window=3, cov_bound=0.01, warmup=1)
    names = {s.name for s in scores}
    assert names == {"fixed_budget", "cov_converged"}
    for s in scores:
        if s.region is not None:
            assert s.qps_rel_err is not None and s.qps_rel_err >= 0.0
