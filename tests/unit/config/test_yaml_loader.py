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

"""Tests for YAML configuration loading, env var interpolation, and serialization."""

from pathlib import Path

import pytest
from inference_endpoint.config.schema import BenchmarkConfig, LoadPatternType
from inference_endpoint.config.schema import TestType as BenchmarkTestType
from pydantic import ValidationError


class TestYAMLLoading:
    """Test BenchmarkConfig.from_yaml_file()."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML config."""
        config_content = """
name: "test-config"
version: "1.0"
type: "offline"

model_params:
  name: "test-model"
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
    worker_initialization_timeout: 120
    transport:
      type: zmq
      recv_buffer_size: 16777216
      send_buffer_size: 8388608

metrics:
  collect:
    - "throughput"

endpoint_config:
  endpoints:
    - "http://localhost:8000"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = BenchmarkConfig.from_yaml_file(config_file)
        assert config.name == "test-config"
        assert config.type == BenchmarkTestType.OFFLINE
        assert len(config.datasets) == 1
        assert config.settings.client.worker_initialization_timeout == 120.0
        assert config.settings.client.transport.recv_buffer_size == 16777216
        assert config.settings.client.transport.send_buffer_size == 8388608

    def test_load_nonexistent_file(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            BenchmarkConfig.from_yaml_file(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        import yaml

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: error:")

        with pytest.raises((ValidationError, ValueError, yaml.YAMLError)):
            BenchmarkConfig.from_yaml_file(config_file)

    def test_env_var_interpolation(self, tmp_path, monkeypatch):
        """Test ${VAR} interpolation in YAML files."""
        monkeypatch.setenv("TEST_ENDPOINT", "http://prod:8000")
        monkeypatch.setenv("TEST_MODEL", "llama-2-70b")

        config_content = """
name: "env-test"
type: "offline"
model_params:
  name: "${TEST_MODEL}"
endpoint_config:
  endpoints:
    - "${TEST_ENDPOINT}"
  api_key: "${MISSING_KEY:-sk-default}"
datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"
"""
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text(config_content)

        config = BenchmarkConfig.from_yaml_file(config_file)
        assert config.model_params.name == "llama-2-70b"
        assert config.endpoint_config.endpoints == ["http://prod:8000"]
        assert config.endpoint_config.api_key == "sk-default"

    def test_bare_env_var_syntax(self, tmp_path, monkeypatch):
        """Test $VAR bare syntax (without braces)."""
        monkeypatch.setenv("BARE_MODEL", "bare-model-name")

        config_content = """
name: "bare-test"
type: "offline"
model_params:
  name: "$BARE_MODEL"
endpoint_config:
  endpoints:
    - "http://localhost:8000"
datasets:
  - name: "test"
    type: "performance"
    path: "test.pkl"
"""
        config_file = tmp_path / "bare_env.yaml"
        config_file.write_text(config_content)

        config = BenchmarkConfig.from_yaml_file(config_file)
        assert config.model_params.name == "bare-model-name"

    def test_env_var_missing_raises(self, tmp_path):
        """Test that missing env vars without defaults raise ValueError."""
        config_content = """
name: "fail"
type: "offline"
model_params:
  name: "${DEFINITELY_NOT_SET}"
datasets: []
"""
        config_file = tmp_path / "bad_env.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="DEFINITELY_NOT_SET"):
            BenchmarkConfig.from_yaml_file(config_file)

    def test_bare_env_var_missing_raises(self, tmp_path):
        """Test that missing bare $VAR raises ValueError."""
        config_content = """
name: "fail"
type: "offline"
model_params:
  name: "$ALSO_NOT_SET"
datasets: []
"""
        config_file = tmp_path / "bad_bare.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="ALSO_NOT_SET"):
            BenchmarkConfig.from_yaml_file(config_file)


class TestDefaultConfigs:
    """Test BenchmarkConfig.create_default_config()."""

    def test_create_default_offline_config(self):
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.OFFLINE)
        assert isinstance(config, BenchmarkConfig)
        assert config.settings.load_pattern.type == LoadPatternType.MAX_THROUGHPUT
        assert config.settings.runtime.min_duration_ms == 600000
        assert config.settings.client.workers >= 1  # auto-resolved from -1

    def test_create_default_online_config(self):
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.ONLINE)
        assert isinstance(config, BenchmarkConfig)
        assert config.settings.load_pattern.type == LoadPatternType.POISSON
        assert config.settings.load_pattern.target_qps == 10.0
        assert config.settings.runtime.min_duration_ms == 600000

    def test_create_default_eval_not_implemented(self):
        with pytest.raises(NotImplementedError, match="EVAL"):
            BenchmarkConfig.create_default_config(BenchmarkTestType.EVAL)

    def test_create_default_submission_not_implemented(self):
        with pytest.raises(NotImplementedError, match="SUBMISSION"):
            BenchmarkConfig.create_default_config(BenchmarkTestType.SUBMISSION)


class TestSerialization:
    """Test YAML roundtrip serialization."""

    def test_serialize_deserialize_roundtrip(self, tmp_path):
        original = BenchmarkConfig.create_default_config(BenchmarkTestType.OFFLINE)

        yaml_file = tmp_path / "test_config.yaml"
        original.to_yaml_file(yaml_file)
        assert yaml_file.exists()

        loaded = BenchmarkConfig.from_yaml_file(yaml_file)
        assert loaded.name == original.name
        assert loaded.type == original.type
        assert loaded.settings.client.workers == original.settings.client.workers
        assert loaded.settings.load_pattern.type == original.settings.load_pattern.type
        assert (
            loaded.settings.client.worker_initialization_timeout
            == original.settings.client.worker_initialization_timeout
        )
        assert (
            loaded.settings.client.transport.recv_buffer_size
            == original.settings.client.transport.recv_buffer_size
        )
        assert (
            loaded.settings.client.transport.send_buffer_size
            == original.settings.client.transport.send_buffer_size
        )

    def test_to_yaml_file_creates_directory(self, tmp_path):
        config = BenchmarkConfig.create_default_config(BenchmarkTestType.ONLINE)

        nested_path = tmp_path / "subdir" / "nested" / "config.yaml"
        config.to_yaml_file(nested_path)

        assert nested_path.exists()
        loaded = BenchmarkConfig.from_yaml_file(nested_path)
        assert loaded.name == config.name
