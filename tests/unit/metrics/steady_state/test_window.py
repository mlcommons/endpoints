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

import pytest

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.window import (
    percentile_lower,
    windowed_metrics,
)


@pytest.mark.unit
def test_percentile_lower():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_lower(vals, 0.5) == 3.0  # index int(0.5*4)=2
    # int(0.99*4)=int(3.96)=3 -> vals[3]=4.0
    assert percentile_lower(vals, 0.99) == 4.0


@pytest.mark.unit
def test_window_issue_span_excludes_drain():
    # Two super-passes. Latencies vary but issue span is issue-time only.
    sp0 = SuperPassRollup(
        index=0,
        n_issued=2,
        first_issue_ns=0,
        last_issue_ns=1_000_000_000,
        ttft_ns=[10.0, 20.0],
        latency_ns=[100.0, 200.0],
        tpot_ns=[],
        out_tokens=0,
    )
    sp1 = SuperPassRollup(
        index=1,
        n_issued=2,
        first_issue_ns=1_000_000_001,
        last_issue_ns=2_000_000_000,
        ttft_ns=[30.0, 40.0],
        latency_ns=[300.0, 400.0],
        tpot_ns=[],
        out_tokens=0,
    )
    m = windowed_metrics([sp0, sp1], 0, 2)
    # span = 2e9 - 0 = 2s; n=4 -> qps = 2.0
    assert m.issue_span_ns == 2_000_000_000
    assert m.qps == pytest.approx(2.0)
    assert m.n_samples == 4
    # ttft over all 4 samples
    assert m.ttft[0.5] == percentile_lower([10.0, 20.0, 30.0, 40.0], 0.5)


@pytest.mark.unit
def test_window_subrange():
    sp0 = SuperPassRollup(
        index=0, n_issued=1, first_issue_ns=0, last_issue_ns=0, ttft_ns=[10.0], latency_ns=[100.0]
    )
    sp1 = SuperPassRollup(
        index=1,
        n_issued=1,
        first_issue_ns=1_000_000_000,
        last_issue_ns=1_000_000_000,
        ttft_ns=[20.0],
        latency_ns=[200.0],
    )
    m = windowed_metrics([sp0, sp1], 1, 2)  # drop warmup sp0
    assert m.n_samples == 1
    assert m.ttft[0.5] == 20.0
