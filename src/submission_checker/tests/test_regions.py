"""Tests for region boundary computation (§5.5 reference algorithm)."""

import math

import pytest

from submission_checker.models import RegionBounds, classify_concurrency, compute_regions

# Appendix B reference table — (M, LT_end, MT_end)
APPENDIX_B = [
    (64, 35, 42),
    (128, 37, 53),
    (256, 38, 69),
    (512, 40, 93),  # Appendix B: MT = 41–93 (algorithm gives round(93.30) = 93)
    (1024, 42, 131),
    (2048, 45, 192),
    (4096, 48, 287),
    (8192, 52, 437),
    (16384, 57, 676),
]


@pytest.mark.parametrize("M, expected_lt_end, expected_mt_end", APPENDIX_B)
def test_region_boundaries_match_appendix_b(M, expected_lt_end, expected_mt_end):
    r = compute_regions(M)
    assert r.low_throughput.end == expected_lt_end, f"M={M}: LT end"
    assert r.med_throughput.end == expected_mt_end, f"M={M}: MT end"


def test_low_latency_always_1_to_32():
    for M in [64, 256, 1024, 8192]:
        r = compute_regions(M)
        assert r.low_latency.start == 1
        assert r.low_latency.end == 32


def test_high_throughput_ends_at_margin():
    for M in [64, 512, 2048]:
        r = compute_regions(M)
        assert r.high_throughput.end == math.ceil(M * 1.10)


def test_regions_are_contiguous():
    r = compute_regions(512)
    assert r.low_latency.end + 1 == r.low_throughput.start
    assert r.low_throughput.end + 1 == r.med_throughput.start
    assert r.med_throughput.end + 1 == r.high_throughput.start


def test_m_must_be_greater_than_32():
    with pytest.raises(ValueError):
        compute_regions(32)
    with pytest.raises(ValueError):
        compute_regions(1)


def test_classify_low_latency():
    r = compute_regions(1024)
    assert classify_concurrency(1, r) == "low_latency"
    assert classify_concurrency(32, r) == "low_latency"


def test_classify_throughput_regions():
    r = compute_regions(1024)  # LT=33-42, MT=43-131, HT=132-1024
    assert classify_concurrency(33, r) == "low_throughput"
    assert classify_concurrency(42, r) == "low_throughput"
    assert classify_concurrency(43, r) == "med_throughput"
    assert classify_concurrency(131, r) == "med_throughput"
    assert classify_concurrency(132, r) == "high_throughput"
    assert classify_concurrency(1024, r) == "high_throughput"


def test_classify_above_m_still_high_throughput():
    r = compute_regions(1024)
    # high_throughput now extends to ceil(1024 * 1.10) = 1127
    assert classify_concurrency(1025, r) == "high_throughput"
    assert classify_concurrency(1126, r) == "high_throughput"
    assert classify_concurrency(1127, r) == "high_throughput"


def test_classify_out_of_range_returns_none():
    r = compute_regions(1024)
    assert classify_concurrency(9999, r) is None


def test_region_bounds_contains():
    b = RegionBounds(33, 42)
    assert b.contains(33)
    assert b.contains(42)
    assert not b.contains(32)
    assert not b.contains(43)
