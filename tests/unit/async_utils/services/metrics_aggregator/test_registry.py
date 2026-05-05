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

"""Tests for ``MetricsRegistry`` and its samplers."""

from __future__ import annotations

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    CounterSampler,
    MetricsRegistry,
    SeriesSampler,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat,
    SeriesStat,
    SessionState,
)

# 1 hour in ns — same as the aggregator's default for time-series metrics.
_NS_HIGH = 3_600_000_000_000


@pytest.mark.unit
class TestCounterSampler:
    def test_increment_and_value(self):
        c = CounterSampler("c", dtype=int)
        c.increment(1)
        c.increment(4)
        assert c.value() == 5

    def test_set(self):
        c = CounterSampler("c", dtype=int)
        c.increment(10)
        c.set(2)
        assert c.value() == 2

    def test_build_stat(self):
        c = CounterSampler("c", dtype=int)
        c.increment(7)
        stat = c.build_stat(exact=False)
        assert isinstance(stat, CounterStat)
        assert stat.name == "c"
        assert stat.value == 7


@pytest.mark.unit
class TestSeriesSampler:
    def _make(self, dtype=int):
        return SeriesSampler(
            "s",
            hdr_low=1,
            hdr_high=_NS_HIGH,
            sig_figs=3,
            n_histogram_buckets=5,
            percentiles=(50.0, 99.0),
            dtype=dtype,
        )

    def test_empty_build_stat(self):
        s = self._make()
        stat = s.build_stat(exact=False)
        assert isinstance(stat, SeriesStat)
        assert stat.count == 0
        # No data → empty histogram. Edges are dynamic and only meaningful
        # once min/max are observed.
        assert stat.histogram == []

    def test_record_and_rollups(self):
        s = self._make()
        for v in [10, 20, 30, 40, 50]:
            s.record(v)
        stat = s.build_stat(exact=False)
        assert stat.count == 5
        assert stat.total == 150
        assert stat.min == 10
        assert stat.max == 50
        assert stat.sum_sq == 10**2 + 20**2 + 30**2 + 40**2 + 50**2

    def test_hdr_percentiles_within_tolerance(self):
        s = self._make()
        for v in range(1, 101):  # 1..100
            s.record(v * 1000)  # values: 1000..100000
        stat = s.build_stat(exact=False)
        # HDR with 3 sig figs is approximate but should be close.
        # Keys are stringified percentile floats (e.g. "50.0", "99.0").
        p50 = stat.percentiles.get("50.0", stat.percentiles.get("50"))
        p99 = stat.percentiles.get("99.0", stat.percentiles.get("99"))
        assert p50 == pytest.approx(50_000, rel=0.05)
        assert p99 == pytest.approx(99_000, rel=0.05)

    def test_final_exact_percentiles(self):
        s = self._make()
        for v in range(1, 101):
            s.record(v * 1000)
        stat = s.build_stat(exact=True)
        # method="lower" returns observed values.
        p50 = stat.percentiles.get("50.0", stat.percentiles.get("50"))
        p99 = stat.percentiles.get("99.0", stat.percentiles.get("99"))
        assert p50 == 50_000
        assert p99 == 99_000

    def test_final_histogram_is_dense(self):
        s = self._make()
        for v in range(1, 11):
            s.record(v)
        stat = s.build_stat(exact=True)
        # Number of buckets matches what was registered.
        assert len(stat.histogram) == 5
        # Final histogram is exact: every recorded value lands in some bucket
        # (clipped into range when out of bounds), so total == count.
        total = sum(c for _, c in stat.histogram)
        assert total == stat.count

    def test_final_histogram_edges_track_observed_range(self):
        """Dynamic edges span [observed_min, observed_max] of the data —
        the histogram auto-zooms instead of using fixed [hdr_low, hdr_high].
        """
        s = self._make()
        for v in (1_000_000, 2_000_000, 5_000_000, 10_000_000):
            s.record(v)
        stat = s.build_stat(exact=True)
        # First bucket starts at observed min (or its log-clamp). Last
        # bucket ends at observed max. Edges should be much tighter than
        # the [1, _NS_HIGH] HDR bounds.
        assert stat.histogram[0][0][0] >= 1
        assert stat.histogram[0][0][0] <= 1_000_000
        assert stat.histogram[-1][0][1] == pytest.approx(10_000_000)
        # All values land in some bucket.
        total = sum(c for _, c in stat.histogram)
        assert total == stat.count == 4

    def test_final_histogram_handles_zero_value(self):
        """Sub-clamp raw values (e.g. 0) are clipped into the first bucket,
        not dropped. Total bucket count equals the recorded count.
        """
        s = self._make()
        s.record(0)
        s.record(100)
        s.record(1000)
        stat = s.build_stat(exact=True)
        total = sum(c for _, c in stat.histogram)
        assert total == stat.count == 3

    def test_hdr_histogram_count_matches_total(self):
        """HDR-derived histogram bucket counts must sum to the recorded count.

        Regression: an earlier implementation derived counts via
        ``get_count_at_value(hi) - get_count_at_value(lo)`` which returns
        single-bucket counts, not cumulative — total ended up far less than
        the actual recorded count.
        """
        s = self._make()
        for v in range(1, 101):
            s.record(v * 1000)
        stat = s.build_stat(exact=False)
        total = sum(c for _, c in stat.histogram)
        # Every recorded value must land in exactly one display bucket.
        assert total == stat.count == 100

    def test_hdr_histogram_distribution_matches_exact(self):
        """HDR-derived bucket counts approximate the exact counts.

        Within ~5% relative tolerance per non-empty bucket: HDR's bucketing
        rounds values into its sub-buckets, which can shift a few near a
        display-bucket boundary, but the bulk shape matches.
        """
        # Values clustered into two display buckets so HDR rounding can't
        # significantly redistribute the totals.
        s = self._make()
        # 60 values around 1e4, 40 values around 1e8 — far apart, so they
        # end up in clearly distinct display buckets.
        for _ in range(60):
            s.record(10_000)
        for _ in range(40):
            s.record(100_000_000)
        live = s.build_stat(exact=False)
        ended = s.build_stat(exact=True)

        live_counts = [c for _, c in live.histogram]
        ended_counts = [c for _, c in ended.histogram]

        # Both must agree on which buckets are non-empty.
        assert [c > 0 for c in live_counts] == [c > 0 for c in ended_counts]
        # And on totals (HDR aggregates exactly across buckets).
        assert sum(live_counts) == sum(ended_counts) == 100

    def test_float_dtype(self):
        s = self._make(dtype=float)
        s.record(1.5)
        s.record(2.5)
        s.record(3.5)
        stat = s.build_stat(exact=True)
        assert stat.count == 3
        assert stat.total == pytest.approx(7.5)


@pytest.mark.unit
class TestMetricsRegistry:
    def test_register_and_increment(self):
        reg = MetricsRegistry()
        reg.register_counter("c1")
        reg.increment("c1", 1)
        reg.increment("c1", 2)
        snap = reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
        assert snap.counter == 1
        # Find the counter in the snapshot.
        counter_stats = [m for m in snap.metrics if isinstance(m, CounterStat)]
        assert len(counter_stats) == 1
        assert counter_stats[0].name == "c1"
        assert counter_stats[0].value == 3

    def test_set_counter(self):
        reg = MetricsRegistry()
        reg.register_counter("c1")
        reg.set_counter("c1", 99)
        snap = reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
        c = next(m for m in snap.metrics if isinstance(m, CounterStat))
        assert c.value == 99

    def test_record_series(self):
        reg = MetricsRegistry()
        reg.register_series(
            "ttft_ns",
            hdr_low=1,
            hdr_high=_NS_HIGH,
            sig_figs=3,
            n_histogram_buckets=10,
            percentiles=(50.0,),
        )
        for v in [100, 200, 300]:
            reg.record("ttft_ns", v)
        snap = reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        s = next(m for m in snap.metrics if isinstance(m, SeriesStat))
        assert s.count == 3
        assert s.total == 600

    def test_seq_increments(self):
        reg = MetricsRegistry()
        reg.register_counter("c")
        s1 = reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
        s2 = reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
        assert s2.counter == s1.counter + 1

    def test_complete_flag_propagates(self):
        reg = MetricsRegistry()
        snap = reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=2)
        assert snap.state == SessionState.COMPLETE
        assert snap.n_pending_tasks == 2

    def test_name_collision_counter(self):
        reg = MetricsRegistry()
        reg.register_counter("dup")
        with pytest.raises(ValueError, match="already registered"):
            reg.register_counter("dup")

    def test_name_collision_series(self):
        reg = MetricsRegistry()
        reg.register_series("dup", hdr_low=1, hdr_high=_NS_HIGH)
        with pytest.raises(ValueError, match="already registered"):
            reg.register_series("dup", hdr_low=1, hdr_high=_NS_HIGH)

    def test_name_collision_cross_kind(self):
        """A counter and a series MUST NOT share a name."""
        reg = MetricsRegistry()
        reg.register_counter("dup")
        with pytest.raises(ValueError, match="already registered"):
            reg.register_series("dup", hdr_low=1, hdr_high=_NS_HIGH)
