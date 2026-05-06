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

"""Sampler hierarchy and registry for the metrics aggregator.

A ``MetricsRegistry`` holds one ``CounterSampler`` per counter and one
``SeriesSampler`` per series. The aggregator hot path calls
``registry.increment(...)`` / ``registry.record(...)`` for every event;
the publisher periodically calls ``registry.build_snapshot(...)`` to
materialize a ``MetricsSnapshot``.

Series samplers maintain three parallel views:

1. Cheap exact rollups (count/total/min/max/sum_sq) — O(1), exact.
2. HDR Histogram — supports cheap live percentiles/histogram.
3. ``array.array`` of raw values — supports exact final percentiles.
"""

from __future__ import annotations

import array
import bisect
import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Final

import numpy as np
from hdrh.histogram import HdrHistogram

from .snapshot import CounterStat, MetricsSnapshot, MetricStat, SeriesStat, SessionState

logger = logging.getLogger(__name__)


# array.array typecodes per dtype. 'q' = signed int64, 'd' = float64.
_ARRAY_TYPECODE: Final[dict[type, str]] = {int: "q", float: "d"}
_NUMPY_DTYPE: Final[dict[type, type]] = {int: np.int64, float: np.float64}


class MetricSampler(ABC):
    """A single named sampler that builds a ``MetricStat`` on demand."""

    name: str

    @abstractmethod
    def build_stat(self, *, exact: bool) -> MetricStat:
        """Materialize the current state into a wire ``MetricStat``.

        ``exact=True`` selects the raw-values-driven computation path used
        for the ``COMPLETE`` snapshot (sort + np.percentile/histogram).
        ``exact=False`` selects the cheap HDR-derived path used for ``LIVE``
        and ``DRAINING`` snapshots.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class CounterSampler(MetricSampler):
    """A monotonic (or settable) counter."""

    __slots__ = ("name", "_value", "_dtype")

    def __init__(self, name: str, dtype: type = int) -> None:
        self.name = name
        self._dtype = dtype
        # Use the dtype to seed the zero so we keep int/float identity.
        self._value: int | float = dtype()

    def increment(self, delta: int | float) -> None:
        self._value += delta

    def set(self, value: int | float) -> None:  # noqa: A003 — domain term.
        self._value = value

    def value(self) -> int | float:
        return self._value

    def build_stat(self, *, exact: bool) -> CounterStat:  # noqa: ARG002
        # Counters are exact at every tick — the ``exact`` flag is part of
        # the sampler protocol but has no effect on counter output.
        return CounterStat(name=self.name, value=self._value)


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def _log_spaced_edges(low: float, high: float, n_buckets: int) -> list[float]:
    """Return ``n_buckets+1`` log-spaced edges over ``[low, high]``.

    ``low`` is clamped to ``max(low, 1)`` so the log is well-defined for
    zero-bound metrics (e.g. token counts starting at 1).
    """
    safe_low = max(float(low), 1.0)
    safe_high = max(float(high), safe_low * 10.0)
    log_lo = math.log(safe_low)
    log_hi = math.log(safe_high)
    step = (log_hi - log_lo) / n_buckets
    return [math.exp(log_lo + i * step) for i in range(n_buckets + 1)]


class SeriesSampler(MetricSampler):
    """An append-only series sampler with cheap rollups + HDR + raw values."""

    __slots__ = (
        "name",
        "_dtype",
        "_hdr",
        "_hdr_low",
        "_hdr_high",
        "_raw",
        "_n_histogram_buckets",
        "_percentiles",
        "_count",
        "_total",
        "_sum_sq",
        "_min",
        "_max",
        "_warned_clamp",
    )

    def __init__(
        self,
        name: str,
        *,
        hdr_low: int,
        hdr_high: int,
        sig_figs: int,
        n_histogram_buckets: int,
        percentiles: tuple[float, ...],
        dtype: type,
    ) -> None:
        if dtype not in _ARRAY_TYPECODE:
            raise ValueError(f"Unsupported series dtype: {dtype!r}")
        self.name = name
        self._dtype = dtype
        # HDR low must be >=1; a bound of 0 is rejected by the C library.
        self._hdr_low = max(int(hdr_low), 1)
        self._hdr_high = int(hdr_high)
        self._hdr = HdrHistogram(self._hdr_low, self._hdr_high, sig_figs)
        self._raw: array.array = array.array(_ARRAY_TYPECODE[dtype])
        # Bucket count is fixed; edges are derived per snapshot from the
        # observed [min, max] so the histogram auto-zooms to the data.
        self._n_histogram_buckets = n_histogram_buckets
        self._percentiles: tuple[float, ...] = percentiles

        self._count: int = 0
        zero = dtype()
        self._total: int | float = zero
        self._sum_sq: int | float = zero
        self._min: int | float = math.inf
        self._max: int | float = -math.inf
        self._warned_clamp: bool = False

    # -- hot path ----------------------------------------------------------

    def record(self, value: int | float) -> None:
        # 1. Cheap exact rollups.
        self._count += 1
        self._total += value
        self._sum_sq += value * value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

        # 2. HDR (clamp into [hdr_low, hdr_high]).
        if self._dtype is int:
            clamped: int | float = max(int(value), self._hdr_low)
        else:
            clamped = max(float(value), float(self._hdr_low))
        if clamped > self._hdr_high:
            clamped = self._hdr_high
        if not self._warned_clamp and clamped != value:
            logger.warning(
                "%s: value %r outside HDR bounds [%d, %d]; clamped (warn-once)",
                self.name,
                value,
                self._hdr_low,
                self._hdr_high,
            )
            self._warned_clamp = True
        # HDR API accepts ints; coerce floats to int for the HDR view.
        self._hdr.record_value(int(clamped))

        # 3. Raw values for exact-final percentile/histogram computation.
        self._raw.append(value)

    # -- snapshot construction --------------------------------------------

    def build_stat(self, *, exact: bool) -> SeriesStat:
        if self._count == 0:
            # No data → no histogram. Edges are dynamic and only meaningful
            # once min/max are observed; consumers should treat an empty
            # histogram as "no data yet".
            return SeriesStat(
                name=self.name,
                count=0,
                total=self._dtype(),
                min=0,
                max=0,
                sum_sq=self._dtype(),
                percentiles={str(p): 0.0 for p in self._percentiles},
                histogram=[],
            )

        if exact:
            return self._exact_stat()
        return self._hdr_stat()

    def _hdr_stat(self) -> SeriesStat:
        perc_dict: dict[str, float] = {
            str(p): float(self._hdr.get_value_at_percentile(p))
            for p in self._percentiles
        }

        # Dynamic display edges, log-spaced over the observed [min, max].
        # Re-derived per snapshot: edges auto-zoom to data, no wasted
        # buckets. Consumers must re-render from (lo, hi, count) triples
        # each frame rather than tracking bucket-by-index.
        n_buckets = self._n_histogram_buckets
        edges = _log_spaced_edges(self._min, self._max, n_buckets)
        counts = [0] * n_buckets

        # Bin HDR sub-bucket counts into the display histogram. Walk the
        # recorded iterator (length bounded by distinct sub-buckets,
        # typically hundreds to thousands per series, not millions).
        for it in self._hdr.get_recorded_iterator():
            v = it.value_iterated_to
            c = it.count_added_in_this_iter_step
            # Place v into the display bucket [edges[idx], edges[idx+1]).
            idx = bisect.bisect_right(edges, v) - 1
            if idx < 0:
                idx = 0
            elif idx >= n_buckets:
                idx = n_buckets - 1
            counts[idx] += c

        histogram: list[tuple[tuple[float, float], int]] = [
            ((edges[i], edges[i + 1]), counts[i]) for i in range(n_buckets)
        ]

        return SeriesStat(
            name=self.name,
            count=self._count,
            total=self._total,
            min=self._min,
            max=self._max,
            sum_sq=self._sum_sq,
            percentiles=perc_dict,
            histogram=histogram,
        )

    def _exact_stat(self) -> SeriesStat:
        np_dtype = _NUMPY_DTYPE[self._dtype]
        arr = np.frombuffer(self._raw, dtype=np_dtype)
        # method="lower" returns observed values (not interpolated) so
        # percentiles round-trip through int dtypes cleanly.
        perc_values = np.percentile(arr, self._percentiles, method="lower")
        perc_dict = {
            str(p): float(v)
            for p, v in zip(self._percentiles, perc_values, strict=True)
        }

        # Dynamic edges from observed [min, max], same as the live HDR path,
        # so consumers see consistent edge semantics across LIVE/DRAINING/
        # COMPLETE. ``_log_spaced_edges`` clamps the lower edge to >=1; clip
        # values into the resulting edge range so any value below 1 (rare,
        # but possible for sub-clamp raw recordings) lands in the first
        # bucket instead of being dropped by np.histogram. Total bucket
        # count then equals the recorded count.
        edges = _log_spaced_edges(
            float(self._min), float(self._max), self._n_histogram_buckets
        )
        arr_clipped = np.clip(arr, edges[0], edges[-1])
        counts, _ = np.histogram(arr_clipped, bins=edges)
        histogram: list[tuple[tuple[float, float], int]] = [
            ((float(edges[i]), float(edges[i + 1])), int(counts[i]))
            for i in range(len(edges) - 1)
        ]

        return SeriesStat(
            name=self.name,
            count=self._count,
            total=self._total,
            min=self._min,
            max=self._max,
            sum_sq=self._sum_sq,
            percentiles=perc_dict,
            histogram=histogram,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_DEFAULT_PERCENTILES: Final[tuple[float, ...]] = (
    99.9,
    99.0,
    97.0,
    95.0,
    90.0,
    80.0,
    75.0,
    50.0,
    25.0,
    10.0,
    5.0,
    1.0,
)


class MetricsRegistry:
    """Central registry of all counter and series samplers."""

    def __init__(self) -> None:
        self._counters: dict[str, CounterSampler] = {}
        self._series: dict[str, SeriesSampler] = {}
        self._seen_names: set[str] = set()
        # Monotonic snapshot emit counter; surfaced on the wire as
        # MetricsSnapshot.counter for diagnostic use by consumers.
        self._counter: int = 0

    # -- registration -----------------------------------------------------

    def register_counter(self, name: str, dtype: type = int) -> CounterSampler:
        if name in self._seen_names:
            raise ValueError(f"Metric name already registered: {name}")
        sampler = CounterSampler(name, dtype=dtype)
        self._counters[name] = sampler
        self._seen_names.add(name)
        return sampler

    def register_series(
        self,
        name: str,
        *,
        hdr_low: int,
        hdr_high: int,
        sig_figs: int = 3,
        n_histogram_buckets: int = 30,
        percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
        dtype: type = int,
    ) -> SeriesSampler:
        if name in self._seen_names:
            raise ValueError(f"Metric name already registered: {name}")
        sampler = SeriesSampler(
            name,
            hdr_low=hdr_low,
            hdr_high=hdr_high,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
            percentiles=percentiles,
            dtype=dtype,
        )
        self._series[name] = sampler
        self._seen_names.add(name)
        return sampler

    # -- hot path ---------------------------------------------------------
    # Direct dict lookup, no isinstance dispatch — these are called once per
    # event in the aggregator's process() loop.

    def increment(self, name: str, delta: int | float = 1) -> None:
        """Increment a counter by ``delta`` (default 1)."""
        self._counters[name].increment(delta)

    def set_counter(self, name: str, value: int | float) -> None:
        self._counters[name].set(value)

    def record(self, name: str, value: int | float) -> None:
        self._series[name].record(value)

    # -- snapshot ---------------------------------------------------------

    def build_snapshot(
        self, *, state: SessionState, n_pending_tasks: int
    ) -> MetricsSnapshot:
        # Exact (raw-values) computation is reserved for the COMPLETE snapshot;
        # live and draining snapshots use the cheap HDR path.
        exact = state == SessionState.COMPLETE
        self._counter += 1
        metrics: list[MetricStat] = []
        for c_sampler in self._counters.values():
            metrics.append(c_sampler.build_stat(exact=exact))
        for s_sampler in self._series.values():
            metrics.append(s_sampler.build_stat(exact=exact))
        return MetricsSnapshot(
            counter=self._counter,
            timestamp_ns=time.monotonic_ns(),
            state=state,
            n_pending_tasks=n_pending_tasks,
            metrics=metrics,
        )

    # -- introspection (mostly for tests) --------------------------------

    def has_counter(self, name: str) -> bool:
        return name in self._counters

    def has_series(self, name: str) -> bool:
        return name in self._series
