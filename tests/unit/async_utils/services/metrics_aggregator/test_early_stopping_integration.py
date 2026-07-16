# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: early-stopping estimates flow registry -> snapshot -> report dict."""

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

# ES percentiles derived from the default report grid, at or above the median
_GRID_ES_PERCENTILES = [0.5, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999]


def _registry_with_data(es_spec: EarlyStoppingSpec | None) -> MetricsRegistry:
    reg = MetricsRegistry(early_stopping=es_spec)
    reg.register_series(
        "ttft_ns", hdr_low=1, hdr_high=10_000_000_000, tail_latency=True
    )
    reg.register_series("isl", hdr_low=1, hdr_high=1_000_000)  # not tail latency
    rng = random.Random(0)
    for _ in range(2000):
        reg.record("ttft_ns", rng.randint(1_000_000, 2_000_000))
        reg.record("isl", rng.randint(10, 500))
    return reg


def _series(snap_dict: dict, name: str) -> dict:
    return next(m for m in snap_dict["metrics"] if m.get("name") == name)


@pytest.mark.unit
class TestConfigSurface:
    def test_schema_is_a_single_flag(self):
        # The whole feature is one opt-in switch: percentiles/confidence/tolerance
        # are loadgen-parity constants, not YAML/CLI knobs.
        from inference_endpoint.config.schema import EarlyStoppingConfig

        assert set(EarlyStoppingConfig.model_fields) == {"enabled"}
        assert EarlyStoppingConfig().enabled is False


@pytest.mark.unit
class TestEarlyStoppingIntegration:
    def test_enabled_emits_grid_derived_percentiles(self):
        # Default spec derives the ES set from the series' own report grid
        # (p >= 0.5) — one source of truth, nothing to tune per-yaml.
        reg = _registry_with_data(EarlyStoppingSpec())
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        ttft, isl = _series(d, "ttft_ns"), _series(d, "isl")
        es = ttft["early_stopping"]
        assert isinstance(es, list)
        assert [b["percentile"] for b in es] == _GRID_ES_PERCENTILES
        for block in es:
            assert block["n"] == 2000
            assert "tolerance" not in block  # constant, not reported
            if block["percentile"] <= 0.99:  # p99.9 floor (6636) exceeds n=2000
                assert block["sufficient"] is True
                assert block["estimate"] >= block["empirical"]  # conservative
        assert "early_stopping" not in isl  # not registered tail_latency
        # and it survives the report-dict conversion
        assert _series_to_metric_dict(ttft)["early_stopping"] == es

    def test_custom_single_percentile(self):
        reg = _registry_with_data(EarlyStoppingSpec(percentiles=(0.99,)))
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        es = _series(d, "ttft_ns")["early_stopping"]
        assert [b["percentile"] for b in es] == [0.99]
        assert es[0]["min_queries"] == 662

    def test_disabled_by_default_no_block(self):
        reg = _registry_with_data(None)  # feature off
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        assert "early_stopping" not in _series(d, "ttft_ns")
        assert "early_stopping" not in _series_to_metric_dict(_series(d, "ttft_ns"))

    def test_live_snapshot_has_no_estimate(self):
        # ES is COMPLETE-only (needs the exact sorted raw array); LIVE must not carry it.
        reg = _registry_with_data(EarlyStoppingSpec())
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=1)
        )
        assert "early_stopping" not in _series(d, "ttft_ns")

    def test_interrupted_snapshot_has_no_estimate(self):
        # SIGTERM path publishes a terminal INTERRUPTED snapshot via the non-exact
        # (HDR) path — it must not carry a partial ES block.
        reg = _registry_with_data(EarlyStoppingSpec())
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.INTERRUPTED, n_pending_tasks=0)
        )
        assert "early_stopping" not in _series(d, "ttft_ns")

    def test_empty_target_series_still_reports_blocks(self):
        # A series that recorded nothing (e.g. all requests failed) must still
        # self-describe as insufficient — not silently look feature-off.
        reg = MetricsRegistry(early_stopping=EarlyStoppingSpec())
        reg.register_series(
            "ttft_ns", hdr_low=1, hdr_high=10_000_000_000, tail_latency=True
        )
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        es = _series(d, "ttft_ns")["early_stopping"]
        assert [b["percentile"] for b in es] == _GRID_ES_PERCENTILES
        for b in es:
            assert b["n"] == 0 and b["sufficient"] is False
            assert b["estimate"] is None and b["empirical"] is None

    def test_all_target_series_get_blocks(self):
        # tpot_ns is the only float-dtype tail-latency series; latency and ttft
        # are int — all three must carry blocks on COMPLETE.
        reg = MetricsRegistry(early_stopping=EarlyStoppingSpec())
        reg.register_series(
            "ttft_ns", hdr_low=1, hdr_high=10_000_000_000, tail_latency=True
        )
        reg.register_series(
            "sample_latency_ns", hdr_low=1, hdr_high=10_000_000_000, tail_latency=True
        )
        reg.register_series(
            "tpot_ns",
            hdr_low=1,
            hdr_high=10_000_000_000,
            dtype=float,
            tail_latency=True,
        )
        rng = random.Random(1)
        for _ in range(1000):
            reg.record("ttft_ns", rng.randint(1_000_000, 2_000_000))
            reg.record("sample_latency_ns", rng.randint(5_000_000, 9_000_000))
            reg.record("tpot_ns", rng.uniform(50_000.0, 90_000.0))
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        for name in ("ttft_ns", "sample_latency_ns", "tpot_ns"):
            es = _series(d, name)["early_stopping"]
            assert isinstance(es, list)
            assert len(es) == len(_GRID_ES_PERCENTILES)

    def test_block_empirical_equals_grid_value(self):
        # The block's empirical and the percentile grid come from the same array
        # and must use the same order-statistic convention.
        reg = _registry_with_data(EarlyStoppingSpec())
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        ttft = _series(d, "ttft_ns")
        for p_block, p_grid in ((0.99, "99.0"), (0.9, "90.0"), (0.95, "95.0")):
            blk = next(b for b in ttft["early_stopping"] if b["percentile"] == p_block)
            assert blk["empirical"] == ttft["percentiles"][p_grid]
