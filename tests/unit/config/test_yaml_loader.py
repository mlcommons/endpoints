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

"""Tests for YAML configuration loading and merging."""

from pathlib import Path

import pytest
from inference_endpoint.config.schema import BenchmarkConfig, LoadPatternType
from inference_endpoint.config.schema import TestType as BenchmarkTestType
from inference_endpoint.config.yaml_loader import ConfigError, ConfigLoader


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
  endpoint:
    - "http://localhost:8000"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = ConfigLoader.load_yaml(config_file)
        assert config.name == "test-config"
        assert config.type == BenchmarkTestType.OFFLINE
        assert len(config.datasets) == 1

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(ConfigError, match="not found"):
            ConfigLoader.load_yaml(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: error:")

        with pytest.raises(ConfigError, match="Config validation failed"):
            ConfigLoader.load_yaml(config_file)

    def test_create_default_offline_config(self):
        """Test creating default offline config."""
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.OFFLINE)
        assert isinstance(config, BenchmarkConfig)
        assert config.settings.load_pattern.type == LoadPatternType.MAX_THROUGHPUT
        assert config.settings.runtime.min_duration_ms == 600000
        assert config.settings.client.workers == 4

    def test_create_default_online_config(self):
        """Test creating default online config."""
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.ONLINE)
        assert isinstance(config, BenchmarkConfig)
        assert config.settings.load_pattern.type == LoadPatternType.POISSON
        assert config.settings.load_pattern.target_qps == 10.0
        assert config.settings.runtime.min_duration_ms == 600000

    def test_serialize_deserialize_roundtrip(self, tmp_path):
        """Test BenchmarkConfig.to_yaml_file() and from_yaml_file() roundtrip."""
        # Create a config
        original = BenchmarkConfig.create_default_config(BenchmarkTestType.OFFLINE)

        # Save to YAML
        yaml_file = tmp_path / "test_config.yaml"
        original.to_yaml_file(yaml_file)
        assert yaml_file.exists()

        # Load back
        loaded = BenchmarkConfig.from_yaml_file(yaml_file)

        # Verify same
        assert loaded.name == original.name
        assert loaded.type == original.type
        assert loaded.settings.client.workers == original.settings.client.workers
        assert loaded.settings.load_pattern.type == original.settings.load_pattern.type

    def test_to_yaml_file_creates_directory(self, tmp_path):
        """Test that to_yaml_file creates parent directories."""
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.ONLINE)

        # Save to nested path that doesn't exist
        nested_path = tmp_path / "subdir" / "nested" / "config.yaml"
        config.to_yaml_file(nested_path)

        assert nested_path.exists()
        # Verify it loads back
        loaded = BenchmarkConfig.from_yaml_file(nested_path)
        assert loaded.name == config.name
