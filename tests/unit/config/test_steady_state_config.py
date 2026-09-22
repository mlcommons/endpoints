# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``settings.steady_state`` block and its give-up deadline on
``settings.timeouts``."""

import pytest
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    Settings,
    SteadyStateConfig,
    TestType,
    Timeouts,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit

_MINIMAL_KWARGS = {
    "type": TestType.OFFLINE,
    "model_params": {"name": "M"},
    "endpoint_config": {"endpoints": ["http://x"]},
    "datasets": [{"path": "D"}],
}


class TestSteadyStateConfig:
    def test_disabled_by_default(self):
        """Opt-in: the detector is unvalidated for most workloads and tokenizes
        every response, so a run must ask for it."""
        assert SteadyStateConfig().enabled is False

    def test_settings_exposes_the_block(self):
        assert Settings().steady_state.enabled is False

    def test_opt_in_through_benchmark_config(self):
        cfg = BenchmarkConfig(
            **_MINIMAL_KWARGS,
            settings={"steady_state": {"enabled": True}},
        )
        assert cfg.settings.steady_state.enabled is True

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            SteadyStateConfig(enbaled=False)


class TestSteadyStateTimeout:
    def test_default_is_ten_minutes(self):
        """Detection sits between the metrics drain and the report, so this
        budget is wall-clock the report waits for. Measured at 94s over a 1.0GB
        event log (~57k samples), so this is several times the observed cost
        without leaving a stalled child able to hold a finished run for half an
        hour."""
        assert Timeouts().steady_state_timeout_s == 600.0

    def test_none_means_unlimited(self):
        assert Timeouts(steady_state_timeout_s=None).steady_state_timeout_s is None

    def test_must_be_positive(self):
        with pytest.raises(ValidationError):
            Timeouts(steady_state_timeout_s=0)
