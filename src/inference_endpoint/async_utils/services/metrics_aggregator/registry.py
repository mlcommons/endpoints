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
from collections.abc import Iterable
from typing import Final

import numpy as np
from hdrh.histogram import HdrHistogram
from inference_endpoint.metrics.early_stopping import (
    EarlyStoppingSpec,
    es_percentile_estimate,
    es_percentiles_from_grid,
    grid_percentile_key,
)

from .snapshot import (
    CounterStat,
    MetricsSnapshot,
    MetricStat,
    SeriesStat,
    SessionState,
    _metric_to_dict,
)

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
        "_es_spec",
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
        es_spec: EarlyStoppingSpec | None = None,
    ) -> None:
        if dtype not in _ARRAY_TYPECODE:
            raise ValueError(f"Unsupported series dtype: {dtype!r}")
        self.name = name
        self._dtype = dtype
        self._es_spec = es_spec
        # HDR low must be >=1; a bound of 0 is rejected by the C library.
        self._hdr_low = max(int(hdr_low), 1)
        self._hdr_high = int(hdr_high)
        # hdrhistogram's C constructor requires `high >= 2*low`; the error
        # it raises is opaque ("ValueError: Could not allocate..."), so
        # validate up front with both values in the message for callers
        # who hit this from a custom registration site.
        if self._hdr_high < self._hdr_low * 2:
            raise ValueError(
                f"{name}: HDR high ({self._hdr_high}) must be >= 2 * low "
                f"({self._hdr_low}); got high/low={self._hdr_high / self._hdr_low:.2f}"
            )
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

    def _es_estimates(self, sorted_values) -> dict[str, float | None] | None:
        # Percentile targets come from this series' own report grid (>= median)
        # unless the spec pins an explicit tuple (tests / offline analysis). The
        # snapshot carries only the compact {grid_key: estimate-or-None} map; the
        # rich detail (empirical/n/min_queries/discarded) is INFO-logged here and
        # recomputable offline via scripts/early_stopping_estimate_from_events.py.
        if self._es_spec is None:
            return None
        spec = self._es_spec
        targets = spec.percentiles or es_percentiles_from_grid(self._percentiles)
        results = [
            es_percentile_estimate(sorted_values, p, spec.confidence) for p in targets
        ]
        logger.info(
            "%s early-stopping detail (confidence %s): %s",
            self.name,
            spec.confidence,
            "; ".join(
                f"p{grid_percentile_key(r.percentile)}: estimate={r.estimate} "
                f"empirical={r.empirical} n={r.n} min_queries={r.min_queries} "
                f"discarded={r.discarded}"
                for r in results
            ),
        )
        return {
            grid_percentile_key(r.percentile): (
                None if r.estimate is None else float(r.estimate)
            )
            for r in results
        }

    def build_stat(self, *, exact: bool) -> SeriesStat:
        if self._count == 0:
            # No data → no histogram. Edges are dynamic and only meaningful
            # once min/max are observed; consumers should treat an empty
            # histogram as "no data yet". Early stopping still self-describes
            # (n=0, sufficient=false) — an empty target series must not look
            # like the feature was disabled.
            early_stopping_percentiles = self._es_estimates(()) if exact else None
            return SeriesStat(
                name=self.name,
                count=0,
                total=self._dtype(),
                min=0,
                max=0,
                sum_sq=self._dtype(),
                percentiles={str(p): 0.0 for p in self._percentiles},
                histogram=[],
                early_stopping_percentiles=early_stopping_percentiles,
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
            sum_sq=float(self._sum_sq),
            percentiles=perc_dict,
            histogram=histogram,
        )

    def _exact_stat(self) -> SeriesStat:
        np_dtype = _NUMPY_DTYPE[self._dtype]
        arr = np.frombuffer(self._raw, dtype=np_dtype)
        # Sort ONCE, in place, and share the sorted array across the percentile
        # grid, the histogram, and the early-stopping estimates. In-place is safe:
        # this is the terminal COMPLETE path, recording has ended, and none of the
        # consumers care about arrival order — while a sorted copy would double the
        # peak memory for multi-million-sample runs.
        arr.sort()
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

        # Early-stopping estimates (COMPLETE path only): conservative confidence-backed
        # bound per configured percentile, all off one sorted raw array. Cold path —
        # the sort is one-time at run end and each estimate is a few beta evaluations.
        early_stopping_percentiles = self._es_estimates(arr)

        return SeriesStat(
            name=self.name,
            count=self._count,
            total=self._total,
            min=self._min,
            max=self._max,
            sum_sq=float(self._sum_sq),
            percentiles=perc_dict,
            histogram=histogram,
            early_stopping_percentiles=early_stopping_percentiles,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


DEFAULT_PERCENTILES: Final[tuple[float, ...]] = (
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

    def __init__(self, early_stopping: EarlyStoppingSpec | None = None) -> None:
        self._counters: dict[str, CounterSampler] = {}
        self._series: dict[str, SeriesSampler] = {}
        self._seen_names: set[str] = set()
        # Optional early-stopping spec; applied to series registered tail_latency=True.
        self._early_stopping = early_stopping
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
        percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
        dtype: type = int,
        tail_latency: bool = False,
    ) -> SeriesSampler:
        """Register a new series.

        ``percentiles`` MUST include ``50.0`` (or ``50``) — median is a
        mandatory metric on every series's display rollup, and
        ``Report._series_to_metric_dict`` reads p50 from this tuple
        rather than recomputing it from raw values. Without p50 the
        median fallback degrades to ``(min + max) / 2`` (midrange),
        which bears no relationship to the actual median; we reject
        such registrations at construction time instead of producing
        misleading reports downstream.
        """
        if name in self._seen_names:
            raise ValueError(f"Metric name already registered: {name}")
        if 50.0 not in percentiles and 50 not in percentiles:
            raise ValueError(
                f"register_series({name!r}): percentiles must include 50.0 — "
                f"median is a mandatory metric on every series. Got: "
                f"{percentiles!r}"
            )
        sampler = SeriesSampler(
            name,
            hdr_low=hdr_low,
            hdr_high=hdr_high,
            sig_figs=sig_figs,
            n_histogram_buckets=n_histogram_buckets,
            percentiles=percentiles,
            dtype=dtype,
            es_spec=self._early_stopping if tail_latency else None,
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


# ---------------------------------------------------------------------------
# Token-count series (OSL / ISL)
# ---------------------------------------------------------------------------

# Canonical parameters for a token-count series, shared by the aggregator's live
# OSL/ISL series and the finalize-side accuracy OSL so both emit an identical
# block from one source. The HDR bounds size the live/HDR percentile view; the
# exact (raw-array) path used for COMPLETE snapshots and accuracy OSL ignores them
# and derives histogram edges from the observed range.
TOKEN_HDR_LOW: Final[int] = 1
TOKEN_HDR_HIGH: Final[int] = 10_000_000  # 10M tokens
TOKEN_SERIES_SIG_FIGS: Final[int] = 3
TOKEN_SERIES_HISTOGRAM_BUCKETS: Final[int] = 30


def build_token_series_dict(
    values: Iterable[int],
    *,
    sig_figs: int = TOKEN_SERIES_SIG_FIGS,
    n_histogram_buckets: int = TOKEN_SERIES_HISTOGRAM_BUCKETS,
) -> dict:
    """Build a token-count series stat dict (exact) from raw ``values``.

    Records ``values`` into a ``SeriesSampler`` configured identically to the
    aggregator's OSL/ISL series, then materializes the exact-path stat (sort +
    np.percentile/np.histogram over the observed range) in the same dict shape
    ``snapshot_to_dict`` produces — so the finalize-side accuracy OSL block
    matches the perf one without duplicating the construction. Off the hot path.
    """
    sampler = SeriesSampler(
        "osl",
        hdr_low=TOKEN_HDR_LOW,
        hdr_high=TOKEN_HDR_HIGH,
        sig_figs=sig_figs,
        n_histogram_buckets=n_histogram_buckets,
        percentiles=DEFAULT_PERCENTILES,
        dtype=int,
    )
    for v in values:
        sampler.record(v)
    return _metric_to_dict(sampler.build_stat(exact=True))
