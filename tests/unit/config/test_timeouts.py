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

"""Tests for the ``settings.timeouts`` block (Timeouts model), the
``runtime.max_issue_duration_ms`` perf-phase cap, and the rejection of config
keys that do not exist (``settings.drain``, top-level ``timeout``,
``settings.service_ready_timeout_s``, ``runtime.min_issue_duration_ms``)."""

import pytest
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    RuntimeConfig,
    TestType,
    Timeouts,
)
from pydantic import ValidationError

_MINIMAL_KWARGS = {
    "type": TestType.OFFLINE,
    "model_params": {"name": "M"},
    "endpoint_config": {"endpoints": ["http://x"]},
    "datasets": [{"path": "D"}],
}


class TestTimeoutsDefaults:
    @pytest.mark.unit
    def test_defaults(self):
        cfg = Timeouts()
        assert cfg.run_timeout_s is None
        assert cfg.service_ready_timeout_s == 30.0
        assert cfg.warmup_drain_timeout_s == 240.0
        assert cfg.interrupted_teardown_grace_s == 30.0
        assert cfg.performance_drain_timeout_s is None
        assert cfg.accuracy_drain_timeout_s is None
        assert cfg.metrics_drain_timeout_s is None

    @pytest.mark.unit
    def test_mounted_on_settings_by_default(self):
        config = BenchmarkConfig(**_MINIMAL_KWARGS)
        assert config.settings.timeouts == Timeouts()


class TestTimeoutsValidation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("run_timeout_s", 0),
            ("run_timeout_s", -1.0),
            ("service_ready_timeout_s", 0),
            ("service_ready_timeout_s", -1.0),
            ("warmup_drain_timeout_s", 0),
            ("warmup_drain_timeout_s", -1.0),
            ("interrupted_teardown_grace_s", 0),
            ("interrupted_teardown_grace_s", -1.0),
            ("interrupted_teardown_grace_s", None),
            ("performance_drain_timeout_s", 0),
            ("performance_drain_timeout_s", -1.0),
            ("accuracy_drain_timeout_s", 0),
            ("accuracy_drain_timeout_s", -1.0),
            ("metrics_drain_timeout_s", 0),
            ("metrics_drain_timeout_s", -1.0),
        ],
    )
    def test_invalid_field_values_are_rejected(self, field, value):
        with pytest.raises(ValidationError, match=field):
            Timeouts(**{field: value})

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field",
        [
            "run_timeout_s",
            "warmup_drain_timeout_s",
            "performance_drain_timeout_s",
            "accuracy_drain_timeout_s",
            "metrics_drain_timeout_s",
        ],
    )
    def test_nullable_deadline_accepts_none(self, field):
        assert getattr(Timeouts(**{field: None}), field) is None


class TestDeletedConfigSurface:
    """Removed config keys must error via extra=forbid, not silently pass."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("settings", "match"),
        [
            ({"drain": {"warmup_timeout_s": 10.0}}, "drain"),
            ({"service_ready_timeout_s": 10.0}, "service_ready_timeout_s"),
        ],
    )
    def test_removed_settings_key_rejected(self, settings, match):
        with pytest.raises(ValidationError, match=match):
            BenchmarkConfig(**_MINIMAL_KWARGS, settings=settings)

    @pytest.mark.unit
    def test_top_level_timeout_rejected(self):
        with pytest.raises(ValidationError, match="timeout"):
            BenchmarkConfig(**_MINIMAL_KWARGS, timeout=42.0)


class TestClientWorkerKnobs:
    @pytest.mark.unit
    def test_worker_lifecycle_knobs_live_on_client(self):
        """Worker lifecycle timeouts are endpoint-client internals and stay on
        settings.client, not in the timeouts block."""
        config = BenchmarkConfig(
            **_MINIMAL_KWARGS,
            settings={"client": {"worker_initialization_timeout": 120.0}},
        )
        assert config.settings.client.worker_initialization_timeout == 120.0
        with pytest.raises(ValidationError):
            Timeouts(worker_initialization_timeout_s=90.0)


class TestTimeoutsYAMLRoundtrip:
    @pytest.mark.unit
    def test_yaml_block_loads(self, tmp_path):
        yaml_content = """
type: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://test:8000"]
datasets:
  - path: "test.jsonl"
settings:
  timeouts:
    run_timeout_s: 900
    warmup_drain_timeout_s: 12.5
    performance_drain_timeout_s: 30.0
    accuracy_drain_timeout_s: null
    metrics_drain_timeout_s: 300.0
"""
        config_file = tmp_path / "timeouts.yaml"
        config_file.write_text(yaml_content)
        config = BenchmarkConfig.from_yaml_file(config_file)
        timeouts = config.settings.timeouts
        assert timeouts.run_timeout_s == 900.0
        assert timeouts.warmup_drain_timeout_s == 12.5
        assert timeouts.performance_drain_timeout_s == 30.0
        assert timeouts.accuracy_drain_timeout_s is None
        assert timeouts.metrics_drain_timeout_s == 300.0


class TestMaxDurationSuffix:
    """max_issue_duration_ms keeps the duration suffix parser (600s, 10m, plain ms)."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value, expected_ms",
        [
            ("600s", 600000),
            ("10m", 600000),
            ("600000ms", 600000),
            ("600000", 600000),
            (600000, 600000),
            ("0.5m", 30000),
            ("1.5s", 1500),
        ],
    )
    def test_suffix_parses(self, value, expected_ms):
        cfg = RuntimeConfig(max_issue_duration_ms=value)
        assert cfg.max_issue_duration_ms == expected_ms

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [0, -1, "0s"])
    def test_zero_and_negative_rejected(self, value):
        # No 0-sentinel: "no cap" is spelled None.
        with pytest.raises(ValidationError):
            RuntimeConfig(max_issue_duration_ms=value)
