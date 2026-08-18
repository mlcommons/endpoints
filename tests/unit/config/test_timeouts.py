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
``runtime.max_duration_ms`` perf-phase cap, and the rejection of config
keys that do not exist (``settings.drain``, top-level ``timeout``,
``settings.service_ready_timeout_s``, ``runtime.min_duration_ms``)."""

import random

import pytest
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    LoadPattern,
    LoadPatternType,
    RuntimeConfig,
    TestType,
    Timeouts,
)
from inference_endpoint.metrics.metric import Throughput
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
        assert cfg.performance_drain_timeout_s is None
        assert cfg.accuracy_drain_timeout_s is None
        assert cfg.metrics_drain_timeout_s is None

    @pytest.mark.unit
    def test_mounted_on_settings_by_default(self):
        config = BenchmarkConfig(**_MINIMAL_KWARGS)
        assert config.settings.timeouts == Timeouts()

    @pytest.mark.unit
    def test_metrics_tokenizer_workers_is_flat_settings_field(self):
        config = BenchmarkConfig(**_MINIMAL_KWARGS)
        assert config.settings.metrics_tokenizer_workers == 4


class TestTimeoutsValidation:
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
    @pytest.mark.parametrize("value", [0, -1.0])
    def test_deadline_must_be_positive_or_none(self, field, value):
        # The 0-sentinel is dead: unlimited is spelled None, never 0.
        with pytest.raises(ValidationError):
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
    def test_deadline_none_means_unlimited(self, field):
        assert getattr(Timeouts(**{field: None}), field) is None

    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["service_ready_timeout_s"])
    def test_ge_zero_fields_accept_zero_reject_negative(self, field):
        assert getattr(Timeouts(**{field: 0}), field) == 0.0
        with pytest.raises(ValidationError):
            Timeouts(**{field: -1.0})

    @pytest.mark.unit
    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Timeouts(unknown_field=1)


class TestDeletedConfigSurface:
    """Removed config keys must error via extra=forbid, not silently pass."""

    @pytest.mark.unit
    def test_settings_drain_block_rejected(self):
        with pytest.raises(ValidationError, match="drain"):
            BenchmarkConfig(
                **_MINIMAL_KWARGS,
                settings={"drain": {"warmup_timeout_s": 10.0}},
            )

    @pytest.mark.unit
    def test_settings_service_ready_timeout_rejected(self):
        with pytest.raises(ValidationError, match="service_ready_timeout_s"):
            BenchmarkConfig(
                **_MINIMAL_KWARGS,
                settings={"service_ready_timeout_s": 10.0},
            )

    @pytest.mark.unit
    def test_runtime_min_duration_rejected(self):
        with pytest.raises(ValidationError, match="min_duration_ms"):
            BenchmarkConfig(
                **_MINIMAL_KWARGS,
                settings={"runtime": {"min_duration_ms": 1000}},
            )

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
    """max_duration_ms keeps the duration suffix parser (600s, 10m, plain ms)."""

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
        cfg = RuntimeConfig(max_duration_ms=value)
        assert cfg.max_duration_ms == expected_ms

    @pytest.mark.unit
    def test_default_is_none(self):
        assert RuntimeConfig().max_duration_ms is None

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [0, -1, "0s"])
    def test_zero_and_negative_rejected(self, value):
        # No 0-sentinel: "no cap" is spelled None.
        with pytest.raises(ValidationError):
            RuntimeConfig(max_duration_ms=value)


class TestDatasetOnceDefault:
    @pytest.mark.unit
    def test_sample_count_defaults_to_dataset_size(self):
        """Without n_samples_to_issue and without any duration knob, a run
        issues the dataset exactly once."""
        config = BenchmarkConfig(**_MINIMAL_KWARGS)
        rt = RuntimeSettings.from_config(config, dataloader_num_samples=123)
        assert rt.n_samples_to_issue is None
        assert rt.total_samples_to_issue() == 123

    @pytest.mark.unit
    def test_explicit_n_samples_still_wins(self):
        rt = RuntimeSettings(
            metric_target=Throughput(10.0),
            reported_metrics=[Throughput(10.0)],
            min_duration_ms=0,
            max_duration_ms=None,
            n_samples_from_dataset=123,
            n_samples_to_issue=7,
            min_sample_count=1,
            rng_sched=random.Random(0),
            rng_sample_index=random.Random(0),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        assert rt.total_samples_to_issue() == 7
