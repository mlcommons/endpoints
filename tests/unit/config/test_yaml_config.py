# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for YAML configuration loading and merging."""

from pathlib import Path

import pytest
from inference_endpoint.config.schema import TestType
from inference_endpoint.config.yaml_config import ConfigError, ConfigLoader


class TestConfigLoader:
    """Test configuration loader."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML config."""
        config_content = """
name: "test-config"
version: "1.0"
type: "offline"

model_params:
  temperature: 0.7
  max_new_tokens: 1024

datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"

settings:
  runtime:
    min_duration_ms: 60000
  load_pattern:
    type: "max_throughput"
  client:
    workers: 4

metrics:
  collect:
    - "throughput"

endpoint_config:
  endpoint: "http://localhost:8000"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = ConfigLoader.load_yaml(config_file)
        assert config.name == "test-config"
        assert config.type == TestType.OFFLINE
        assert len(config.datasets) == 1

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(ConfigError, match="not found"):
            ConfigLoader.load_yaml(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: error:")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            ConfigLoader.load_yaml(config_file)

    def test_create_default_offline_config(self):
        """Test creating default offline config."""
        config = ConfigLoader.create_default_config("offline")
        assert config["load_pattern"]["type"] == "max_throughput"
        assert config["mode"] == "perf"
        assert "client" in config
        assert "runtime" in config

    def test_create_default_online_config(self):
        """Test creating default online config."""
        config = ConfigLoader.create_default_config("online")
        assert config["load_pattern"]["type"] == "poisson"
        assert config["mode"] == "perf"
        assert "qps" in config["load_pattern"]

    def test_merge_with_cli_args(self, tmp_path):
        """Test merging YAML config with CLI arguments."""
        config_content = """
name: "test"
type: "online"

datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"

settings:
  load_pattern:
    type: "poisson"
    qps: 10
  client:
    workers: 4
    max_concurrency: 50

endpoint_config:
  endpoint: "http://localhost:8000"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        yaml_config = ConfigLoader.load_yaml(config_file)

        cli_args = {
            "endpoint": "http://production.com",
            "qps": 100,
            "workers": 8,
        }

        merged = ConfigLoader.merge_with_cli_args(yaml_config, cli_args)

        # CLI should override
        assert merged["endpoint"] == "http://production.com"
        assert merged["load_pattern"]["qps"] == 100
        assert merged["client"]["workers"] == 8
        # YAML values preserved
        assert merged["client"]["max_concurrency"] == 50

    def test_locked_baseline_prevents_override(self, tmp_path):
        """Test that locked baseline cannot be overridden."""
        config_content = """
name: "official"
type: "submission"

baseline:
  locked: true
  model: "llama-2-70b"
  ruleset: "mlperf-inference-v6.0"

datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"
"""
        config_file = tmp_path / "official.yaml"
        config_file.write_text(config_content)

        yaml_config = ConfigLoader.load_yaml(config_file)

        cli_args = {
            "model": "different-model",  # Should fail
        }

        with pytest.raises(ConfigError, match="locked fields"):
            ConfigLoader.merge_with_cli_args(yaml_config, cli_args)

    def test_merge_ignores_none_values(self, tmp_path):
        """Test that None CLI values don't override YAML."""
        config_content = """
name: "test"
type: "offline"

datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"

settings:
  client:
    workers: 4

endpoint_config:
  endpoint: "http://localhost:8000"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        yaml_config = ConfigLoader.load_yaml(config_file)

        cli_args = {
            "endpoint": None,  # Should not override
            "workers": 8,  # Should override
        }

        merged = ConfigLoader.merge_with_cli_args(yaml_config, cli_args)

        # YAML value preserved for None
        assert merged["endpoint"] == "http://localhost:8000"
        # Non-None value overrides
        assert merged["client"]["workers"] == 8

    def test_validate_concurrency_auto_adjust(self, caplog):
        """Test auto-adjustment of max_concurrency when less than target_concurrency."""
        config = {
            "load_pattern": {"type": "concurrency", "target_concurrency": 64},
            "client": {"max_concurrency": 32},  # Too small
            "runtime": {},
        }

        with caplog.at_level("INFO"):  # Need INFO level for auto-adjust message
            ConfigLoader.validate_config(config, benchmark_mode="online")

        # Should auto-adjust
        assert config["client"]["max_concurrency"] == 64
        log_text = caplog.text
        assert "32" in log_text and "64" in log_text  # Shows both values

    def test_validate_concurrency_sufficient(self, caplog):
        """Test no adjustment when max_concurrency >= target_concurrency."""
        config = {
            "load_pattern": {"type": "poisson", "target_concurrency": 32},
            "client": {"max_concurrency": 64},  # OK
            "runtime": {},
        }

        with caplog.at_level("INFO"):
            ConfigLoader.validate_config(config, benchmark_mode="online")

        # Should not adjust
        assert config["client"]["max_concurrency"] == 64
        # Should log info
        assert "OK" in caplog.text or "max_concurrency" in caplog.text
