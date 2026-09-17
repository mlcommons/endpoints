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
    def test_enabled_by_default(self):
        assert SteadyStateConfig().enabled is True

    def test_settings_exposes_the_block(self):
        assert Settings().steady_state.enabled is True

    def test_opt_out_through_benchmark_config(self):
        cfg = BenchmarkConfig(
            **_MINIMAL_KWARGS,
            settings={"steady_state": {"enabled": False}},
        )
        assert cfg.settings.steady_state.enabled is False

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            SteadyStateConfig(enbaled=False)


class TestSteadyStateTimeout:
    def test_default_is_thirty_minutes(self):
        assert Timeouts().steady_state_timeout_s == 1800.0

    def test_none_means_unlimited(self):
        assert Timeouts(steady_state_timeout_s=None).steady_state_timeout_s is None

    def test_must_be_positive(self):
        with pytest.raises(ValidationError):
            Timeouts(steady_state_timeout_s=0)
