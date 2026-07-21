# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration: early-stopping estimates flow registry -> snapshot -> report dict.

The pure math is covered in tests/unit/metrics/test_early_stopping.py; these tests
pin only the wiring: which snapshots carry the map, which series get it, its shape,
and the in-place-sort finalize optimization.
"""

import random

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    SessionState,
    snapshot_to_dict,
)
from inference_endpoint.metrics.early_stopping import EarlyStoppingSpec
from inference_endpoint.metrics.report import _series_to_metric_dict

pytestmark = pytest.mark.unit

# ES keys derived from the default report grid (>= median) — mirror the
# `percentiles` grid keys AND its (descending) order so the two overlay 1:1.
_GRID_ES_KEYS = ["99.9", "99.0", "97.0", "95.0", "90.0", "80.0", "75.0", "50.0"]


def _registry(es_spec, series=("ttft_ns",), n=2000, dtype=int):
    reg = MetricsRegistry(early_stopping=es_spec)
    for name in series:
        reg.register_series(
            name, hdr_low=1, hdr_high=10_000_000_000, dtype=dtype, tail_latency=True
        )
    reg.register_series("isl", hdr_low=1, hdr_high=1_000_000)  # not tail latency
    rng = random.Random(0)
    for _ in range(n):
        for name in series:
            value = rng.randint(1_000_000, 2_000_000)
            reg.record(name, float(value) if dtype is float else value)
        reg.record("isl", rng.randint(10, 500))
    return reg


def _snap(reg, state=SessionState.COMPLETE, pending=0):
    return snapshot_to_dict(reg.build_snapshot(state=state, n_pending_tasks=pending))


def _series(snap_dict, name):
    return next(m for m in snap_dict["metrics"] if m.get("name") == name)


def test_config_is_a_single_opt_out_flag():
    # Default-on with one opt-out is the agreed config surface; anything more is
    # a knob that can weaken the statistical claim.
    from inference_endpoint.config.schema import EarlyStoppingConfig

    assert set(EarlyStoppingConfig.model_fields) == {"enabled"}
    assert EarlyStoppingConfig().enabled is True


def test_complete_snapshot_carries_grid_keyed_estimates():
    # The one test of the full output contract: grid-mirrored keys, null when the
    # run is below a percentile's floor, conservative vs the grid's empirical
    # value, non-tail series untouched, and survival through the report dict.
    d = _snap(_registry(EarlyStoppingSpec()))
    ttft, isl = _series(d, "ttft_ns"), _series(d, "isl")
    esp = ttft["early_stopping_percentiles"]
    assert list(esp) == _GRID_ES_KEYS
    assert esp["99.9"] is None  # n=2000 < the p99.9 floor (6636)
    for key, estimate in esp.items():
        if estimate is not None:
            assert estimate >= ttft["percentiles"][key]
    assert "early_stopping_percentiles" not in isl
    # the report dict keeps the map and places it directly after the grid
    md = _series_to_metric_dict(ttft)
    assert md["early_stopping_percentiles"] == esp
    keys = list(md)
    assert keys.index("early_stopping_percentiles") == keys.index("percentiles") + 1


def test_map_only_on_enabled_complete_snapshots():
    # Disabled registries and non-exact snapshots (LIVE mid-run, INTERRUPTED via
    # SIGTERM) must not carry the map — partial data would fabricate a bound.
    assert "early_stopping_percentiles" not in _series(
        _snap(_registry(None)), "ttft_ns"
    )
    enabled = _registry(EarlyStoppingSpec())
    for state, pending in ((SessionState.LIVE, 1), (SessionState.INTERRUPTED, 0)):
        d = _snap(enabled, state=state, pending=pending)
        assert "early_stopping_percentiles" not in _series(d, "ttft_ns")


def test_every_tail_latency_series_and_dtype_gets_the_map():
    # The tail_latency registration flag is the only thing that attaches ES; all
    # three real series (incl. the float-dtype tpot_ns path) must carry the map.
    for dtype in (int, float):
        reg = _registry(
            EarlyStoppingSpec(),
            series=("ttft_ns", "tpot_ns", "sample_latency_ns"),
            n=100,
            dtype=dtype,
        )
        for name in ("ttft_ns", "tpot_ns", "sample_latency_ns"):
            assert (
                list(_series(_snap(reg), name)["early_stopping_percentiles"])
                == _GRID_ES_KEYS
            )


def test_empty_series_reports_all_null():
    # A tail series that recorded nothing (e.g. every request failed) must still
    # self-describe as insufficient — silence would look like feature-off.
    stat = _series(_snap(_registry(EarlyStoppingSpec(), n=0)), "ttft_ns")
    esp = stat["early_stopping_percentiles"]
    assert list(esp) == _GRID_ES_KEYS
    assert all(v is None for v in esp.values())
    # and the report dict must not drop it with the empty rollups
    assert _series_to_metric_dict(stat)["early_stopping_percentiles"] == esp


def test_custom_grid_with_p100_and_int_keys_is_safe():
    # A legal report grid may contain 100.0 (np.percentile accepts it) and int
    # entries; the COMPLETE snapshot must neither crash (p1.0 is outside the
    # binomial domain — and this path runs default-on inside publish_final) nor
    # emit keys that diverge from the grid's own strings.
    reg = MetricsRegistry(early_stopping=EarlyStoppingSpec())
    reg.register_series(
        "ttft_ns",
        hdr_low=1,
        hdr_high=10_000_000_000,
        percentiles=(100.0, 99, 50),
        tail_latency=True,
    )
    for _ in range(1000):
        reg.record("ttft_ns", 1_000_000)
    esp = _series(_snap(reg), "ttft_ns")["early_stopping_percentiles"]
    assert list(esp) == ["99", "50"]  # 100.0 excluded; int keys + grid order kept


def test_repeated_complete_snapshots_are_identical():
    # The exact path sorts the raw array IN PLACE (avoids a full transient copy);
    # a second COMPLETE snapshot must be byte-identical or the mutation leaked.
    reg = _registry(EarlyStoppingSpec())
    assert _series(_snap(reg), "ttft_ns") == _series(_snap(reg), "ttft_ns")
