# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test: early-stopping estimate flows registry -> snapshot -> report dict."""

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


def _registry_with_data(es_spec: EarlyStoppingSpec | None) -> MetricsRegistry:
    reg = MetricsRegistry(early_stopping=es_spec)
    reg.register_series("ttft_ns", hdr_low=1, hdr_high=10_000_000_000)
    reg.register_series("isl", hdr_low=1, hdr_high=1_000_000)  # non-target series
    rng = random.Random(0)
    for _ in range(2000):
        reg.record("ttft_ns", rng.randint(1_000_000, 2_000_000))
        reg.record("isl", rng.randint(10, 500))
    return reg


def _series(snap_dict: dict, name: str) -> dict:
    return next(m for m in snap_dict["metrics"] if m.get("name") == name)


@pytest.mark.unit
class TestEarlyStoppingIntegration:
    def test_enabled_populates_target_series_only(self):
        reg = _registry_with_data(EarlyStoppingSpec(percentile=0.99))
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        ttft, isl = _series(d, "ttft_ns"), _series(d, "isl")
        es = ttft["early_stopping"]
        assert es["sufficient"] is True
        assert es["estimate"] >= es["empirical"]  # conservative
        assert es["n"] == 2000
        assert "early_stopping" not in isl  # non-target series untouched
        # and it survives the report-dict conversion
        assert _series_to_metric_dict(ttft)["early_stopping"] == es

    def test_disabled_by_default_no_block(self):
        reg = _registry_with_data(None)  # feature off
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.COMPLETE, n_pending_tasks=0)
        )
        assert "early_stopping" not in _series(d, "ttft_ns")
        assert "early_stopping" not in _series_to_metric_dict(_series(d, "ttft_ns"))

    def test_live_snapshot_has_no_estimate(self):
        # ES is COMPLETE-only (needs the exact sorted raw array); LIVE must not carry it.
        reg = _registry_with_data(EarlyStoppingSpec(percentile=0.99))
        d = snapshot_to_dict(
            reg.build_snapshot(state=SessionState.LIVE, n_pending_tasks=1)
        )
        assert "early_stopping" not in _series(d, "ttft_ns")
