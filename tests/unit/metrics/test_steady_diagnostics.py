import pytest

from inference_endpoint.metrics.steady_diagnostics import analyse

pytestmark = pytest.mark.unit

S = 1_000_000_000


def _row(issue_s, complete_s, conv, counted, tokens):
    return [
        issue_s * S,
        issue_s * S + S // 10,
        complete_s * S,
        conv,
        1,
        counted,
        tokens,
        False,
    ]


def test_full_load_throughput_excludes_the_tail_and_uncounted_work() -> None:
    """Full-load throughput counts only counted requests completing between
    measurement start and pool empty; the window figure spans the tail."""
    collected = {
        "requests": {
            "a": _row(0, 10, "repo__issue-1", True, 100),
            "b": _row(0, 10, "sim_1", True, 100),
            # Tail: last measured turn finishes long after the pool emptied.
            "c": _row(10, 40, "repo__issue-2", True, 100),
            # Unmeasured padding inside the full-load span is not counted.
            "d": _row(0, 8, "sim_2__unmeasured_1", False, 1000),
        },
        "start_ns": 0,
        "stop_ns": None,
    }
    plan = {
        "measure_ns": 0,
        "pool_empty_ns": 10 * S,
        "measured_phase_s": 40.0,
        "copies": 1,
    }
    throughput = analyse(collected, plan)["throughput"]
    assert throughput["full_load_tokens_per_s"] == pytest.approx(200 / 10)
    assert throughput["full_load_duration_s"] == pytest.approx(10)
    assert throughput["window_tokens_per_s"] == pytest.approx(300 / 40)


def test_full_load_throughput_absent_without_pool_empty_time() -> None:
    collected = {
        "requests": {"a": _row(0, 10, "sim_1", True, 100)},
        "start_ns": 0,
        "stop_ns": None,
    }
    throughput = analyse(
        collected, {"measure_ns": 0, "pool_empty_ns": None, "copies": 1}
    )["throughput"]
    assert "full_load_tokens_per_s" not in throughput
