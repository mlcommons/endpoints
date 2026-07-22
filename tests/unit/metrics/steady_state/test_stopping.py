import pytest
from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import (
    cov,
    rule_cov_converged,
    rule_fixed_budget,
)


def _sp(index, ttft_vals, lat_vals):
    return SuperPassRollup(
        index=index, n_issued=len(ttft_vals),
        first_issue_ns=index, last_issue_ns=index + 1,
        ttft_ns=list(ttft_vals), latency_ns=list(lat_vals),
    )


@pytest.mark.unit
def test_cov_zero_for_constant():
    assert cov([5.0, 5.0, 5.0]) == 0.0


@pytest.mark.unit
def test_fixed_budget_region():
    series = [_sp(i, [1.0, 2.0], [1.0, 2.0]) for i in range(6)]
    assert rule_fixed_budget(series, k=3, warmup=1) == (1, 4)
    # clamps to end
    assert rule_fixed_budget(series, k=100, warmup=1) == (1, 6)


@pytest.mark.unit
def test_cov_converges_when_stable():
    # sp0 warmup (noisy), sp1..sp4 stable -> converges at window of 3 stable ones
    series = [_sp(0, [100.0], [100.0])]
    for i in range(1, 5):
        series.append(_sp(i, [10.0, 10.0], [50.0, 50.0]))
    region = rule_cov_converged(series, window=3, cov_bound=0.01, warmup=1)
    assert region == (1, 4)  # first sp_end where trailing 3 (sp1,sp2,sp3) are stable


@pytest.mark.unit
def test_cov_never_converges_returns_none():
    series = [_sp(0, [1.0], [1.0])]
    for i in range(1, 6):
        series.append(_sp(i, [float(10**i)], [float(10**i)]))  # wildly varying
    assert rule_cov_converged(series, window=3, cov_bound=0.01, warmup=1) is None
