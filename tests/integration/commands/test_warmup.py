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

"""Integration tests for warmup phase against echo server."""

import argparse
import textwrap

import pytest
from inference_endpoint.commands.benchmark import run_benchmark_command


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warmup_offline_with_echo_server(
    mock_http_echo_server, ds_pickle_dataset_path, tmp_path, caplog
):
    """Warmup phase runs and completes before the performance test starts."""
    config_yaml = textwrap.dedent(f"""
        name: "warmup-test-offline"
        version: "1.0"
        type: "offline"

        warmup:
          num_samples: 4
          input_seq_length: 16
          output_seq_length: 8
          range_ratio: 0.9
          random_seed: 42

        model_params:
          name: "Qwen/Qwen2.5-0.5B-Instruct"
          temperature: 0.0
          max_new_tokens: 16

        datasets:
          - name: custom
            type: "performance"
            path: "{ds_pickle_dataset_path}"
            parser:
              input: prompt

        settings:
          runtime:
            min_duration_ms: 0
            max_duration_ms: 30000
            n_samples_to_issue: 10
          load_pattern:
            type: "max_throughput"
          client:
            workers: 1
            warmup_connections: 0

        endpoint_config:
          endpoints:
            - "{mock_http_echo_server.url}"
          api_key: null

        report_dir: "{tmp_path}"
    """).strip()

    config_file = tmp_path / "warmup_test.yaml"
    config_file.write_text(config_yaml)

    args = argparse.Namespace(
        benchmark_mode="from-config",
        config=str(config_file),
        output=None,
        mode=None,
        verbose=1,
    )

    with caplog.at_level("INFO"):
        await run_benchmark_command(args)

    log_text = caplog.text
    assert "Warmup: issuing samples" in log_text, "Warmup did not start"
    assert "Warmup complete" in log_text, "Warmup did not complete"
    assert "QPS:" in log_text, "Performance test did not run after warmup"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warmup_online_with_echo_server(
    mock_http_echo_server, ds_pickle_dataset_path, tmp_path, caplog
):
    """Warmup phase runs before the online (Poisson) performance test."""
    config_yaml = textwrap.dedent(f"""
        name: "warmup-test-online"
        version: "1.0"
        type: "online"

        warmup:
          num_samples: 4
          input_seq_length: 16
          output_seq_length: 8
          range_ratio: 0.8
          random_seed: 42

        model_params:
          name: "Qwen/Qwen2.5-0.5B-Instruct"
          temperature: 0.0
          max_new_tokens: 16
          streaming: "on"

        datasets:
          - name: custom
            type: "performance"
            path: "{ds_pickle_dataset_path}"
            parser:
              input: prompt

        settings:
          runtime:
            min_duration_ms: 0
            max_duration_ms: 5000
          load_pattern:
            type: "poisson"
            target_qps: 50.0
          client:
            workers: 1
            warmup_connections: 0

        endpoint_config:
          endpoints:
            - "{mock_http_echo_server.url}"
          api_key: null

        report_dir: "{tmp_path}"
    """).strip()

    config_file = tmp_path / "warmup_online_test.yaml"
    config_file.write_text(config_yaml)

    args = argparse.Namespace(
        benchmark_mode="from-config",
        config=str(config_file),
        output=None,
        mode=None,
        verbose=1,
    )

    with caplog.at_level("INFO"):
        await run_benchmark_command(args)

    log_text = caplog.text
    assert "Warmup: issuing samples" in log_text, "Warmup did not start"
    assert "Warmup complete" in log_text, "Warmup did not complete"
    assert "QPS:" in log_text, "Performance test did not run after warmup"
