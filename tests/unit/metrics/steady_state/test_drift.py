# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from inference_endpoint.metrics.steady_state.drift import (
    analyze_trend,
    classify_run,
    ensemble_vote,
    super_pass_metric,
)
from inference_endpoint.metrics.steady_state.series import SuperPassRollup


@pytest.mark.unit
def test_analyze_trend_monotonic_rise_is_drifting():
    # 2.9 -> 10.9 over 10 points, near-linear: large rel_drift, high snr.
    vals = [2.9, 3.8, 4.7, 5.6, 6.5, 7.4, 8.3, 9.2, 10.1, 10.9]
    t = analyze_trend(vals)
    assert t.verdict == "drifting"
    assert t.rel_drift > 0.5 and t.snr > 2.0


@pytest.mark.unit
def test_analyze_trend_flat_noisy_is_steady():
    # oscillates around 100 with no trend -> steady despite non-zero scatter
    vals = [100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 99.0]
    t = analyze_trend(vals)
    assert t.verdict == "steady"
    assert abs(t.rel_drift) < 0.15


@pytest.mark.unit
def test_analyze_trend_too_few_points():
    assert analyze_trend([1.0, 2.0, 3.0]).verdict == "insufficient"


def _sp(index, ttft, lat, span_ns=1_000_000_000):
    return SuperPassRollup(
        index=index,
        n_issued=len(ttft),
        first_issue_ns=index * span_ns,
        last_issue_ns=index * span_ns + span_ns,
        ttft_ns=list(ttft),
        latency_ns=list(lat),
    )


@pytest.mark.unit
def test_super_pass_metric_qps_and_percentiles():
    series = [_sp(i, [10.0, 20.0], [100.0, 200.0]) for i in range(3)]
    # warmup=0 -> all 3; each super-pass span 1s, n_issued 2 -> qps 2.0
    assert super_pass_metric(series, "qps", warmup=0) == [2.0, 2.0, 2.0]
    # ttft p99 (method="lower", n=2 -> index int(0.99*1)=0) -> 10.0 each
    assert super_pass_metric(series, "ttft_p99", warmup=0) == [10.0, 10.0, 10.0]


@pytest.mark.unit
def test_classify_run_flags_only_the_drifting_metric():
    # ttft rises hard (each super-pass 10 samples at a rising level so p99 tracks
    # it); latency flat; qps flat.
    series = []
    for i in range(8):
        ttft = [1_000_000_000 + i * 1_000_000_000] * 10  # 1s -> 8s
        series.append(_sp(i, ttft, [5.0e9] * 10))
    verdicts = classify_run(series, warmup=0)
    assert verdicts["ttft_p99"].verdict == "drifting"
    assert verdicts["qps"].verdict == "steady"
    assert verdicts["lat_p50"].verdict == "steady"


@pytest.mark.unit
def test_ensemble_vote_counts_converged():
    # a stable series: detectors should converge; concordance in [0,1]
    series = [_sp(0, [100.0, 100.0], [100.0, 100.0])]
    for i in range(1, 8):
        series.append(_sp(i, [10.0, 10.0], [50.0, 50.0]))
    v = ensemble_vote(series, warmup=1)
    assert v.n_detectors == 6
    assert 0.0 <= v.concordance <= 1.0
