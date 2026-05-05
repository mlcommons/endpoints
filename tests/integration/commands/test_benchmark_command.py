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

"""Integration tests for benchmark commands against echo server."""

import json
import re
from pathlib import Path

import pytest
import yaml
from inference_endpoint.commands.benchmark.execute import run_benchmark
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EndpointConfig,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    RuntimeConfig,
    Settings,
    StreamingMode,
    TestMode,
    TestType,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig

_TEST_SETTINGS = Settings(
    runtime=RuntimeConfig(min_duration_ms=0),
    load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    client=HTTPClientConfig(num_workers=1, warmup_connections=0, max_connections=10),
)


def _config(endpoint_url: str, dataset_path: str, **overrides) -> BenchmarkConfig:
    """Build a minimal BenchmarkConfig for testing."""
    defaults = {
        "type": TestType.OFFLINE,
        "endpoint_config": EndpointConfig(endpoints=[endpoint_url]),
        "model_params": ModelParams(name="echo-server", streaming=StreamingMode.OFF),
        "datasets": [Dataset(path=dataset_path, type=DatasetType.PERFORMANCE)],
        "settings": _TEST_SETTINGS,
    }
    return BenchmarkConfig(**(defaults | overrides))


def _poisson_settings(target_qps: float, duration_s: int = 2) -> Settings:
    return Settings(
        runtime=RuntimeConfig(min_duration_ms=duration_s * 1000),
        load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=target_qps),
        client=HTTPClientConfig(
            num_workers=1, warmup_connections=0, max_connections=10
        ),
    )


class TestBenchmarkCommandIntegration:
    """Integration tests for benchmark commands with echo server."""

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_offline_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            model_params=ModelParams(name="echo-server", streaming=streaming),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "QPS:" in caplog.text
        assert "Starting phase:" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_poisson_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            model_params=ModelParams(name="echo-server", streaming=streaming),
            settings=_poisson_settings(target_qps=50),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text
        assert "Starting phase:" in caplog.text

    @pytest.mark.integration
    @pytest.mark.parametrize("streaming", [StreamingMode.OFF, StreamingMode.ON])
    def test_concurrency_benchmark(
        self, mock_http_echo_server, ds_dataset_path, caplog, streaming
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            model_params=ModelParams(name="echo-server", streaming=streaming),
            settings=Settings(
                runtime=RuntimeConfig(min_duration_ms=2000),
                load_pattern=LoadPattern(
                    type=LoadPatternType.CONCURRENCY, target_concurrency=4
                ),
                client=HTTPClientConfig(
                    num_workers=1, warmup_connections=0, max_connections=10
                ),
            ),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Completed in" in caplog.text
        assert "successful" in caplog.text

    @pytest.mark.integration
    def test_results_json_output(
        self, mock_http_echo_server, ds_dataset_path, tmp_path
    ):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            report_dir=tmp_path,
        )
        run_benchmark(config, TestMode.PERF)

        results_path = tmp_path / "results.json"
        assert results_path.exists()
        results = json.loads(results_path.read_text())
        assert "config" in results
        assert results["results"]["total"] > 0
        assert results["results"]["successful"] >= 0

    @pytest.mark.integration
    def test_mode_logging(self, mock_http_echo_server, ds_dataset_path, caplog):
        config = _config(
            mock_http_echo_server.url,
            ds_dataset_path,
            type=TestType.ONLINE,
            settings=_poisson_settings(target_qps=20),
        )
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)

        assert "Mode:" in caplog.text
        assert "QPS: 20" in caplog.text
        assert "Responses: False" in caplog.text


TEMPLATE_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "inference_endpoint"
    / "config"
    / "templates"
)

# Templates generated by regenerate_templates.py (excludes handwritten eval/submission)
_GENERATED_TEMPLATES = sorted(
    p.name
    for p in TEMPLATE_DIR.glob("*_template*.yaml")
    if p.name.startswith(("offline_", "online_", "concurrency_"))
)


# Non-gated tokenizer model used in place of the templates' default
# (which references gated meta-llama/Llama-3.1-*). The echo-server e2e
# path doesn't care about the model identity, only that the tokenizer
# exists for the metrics aggregator's ISL/OSL/TPOT triggers. TinyLlama's
# tokenizer is ~1MB and matches the Llama-family tokenizer the templates
# were written against.
_TEST_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _resolve_template(template_path: Path, server_url: str) -> dict:
    """Load a template YAML, strip <PLACEHOLDER> wrappers, and patch for testing.

    Replaces placeholders with working values, swaps the gated default
    model for a non-gated tokenizer (so tests run without ``HF_TOKEN``),
    and caps ``n_samples_to_issue``. Everything else stays as the template
    defines it.
    """
    raw = template_path.read_text()
    # Strip <PLACEHOLDER eg: value> → value (all templates use eg: form)
    raw = re.sub(r"<[^>]*eg:\s*([^>]+)>", r"\1", raw)
    # Replace endpoint URLs with the test server
    raw = re.sub(r"http://localhost:\d+", server_url, raw)
    data = yaml.safe_load(raw)

    # Swap any gated default model name for a non-gated tokenizer. The
    # generated templates' "eg: meta-llama/Llama-3.1-8B-Instruct" placeholder
    # points at a gated repo; substituting gpt2 lets these tests run in CI
    # without HF_TOKEN.
    if "model_params" in data and isinstance(data["model_params"], dict):
        data["model_params"]["name"] = _TEST_MODEL_NAME

    # Cap total samples so test finishes in seconds
    data.setdefault("settings", {})
    data["settings"].setdefault("runtime", {})
    data["settings"]["runtime"]["n_samples_to_issue"] = 10

    # Accuracy datasets can't run e2e against echo server (no scorer), so keep only performance datasets.
    data["datasets"] = [
        ds for ds in data.get("datasets", []) if ds.get("type") != "accuracy"
    ]
    return data


class TestTemplateIntegration:
    """Verify generated templates run end-to-end against a local server."""

    @pytest.mark.integration
    @pytest.mark.parametrize("template", _GENERATED_TEMPLATES)
    def test_template_runs(self, mock_http_echo_server, tmp_path, caplog, template):
        data = _resolve_template(TEMPLATE_DIR / template, mock_http_echo_server.url)
        tmp_yaml = tmp_path / template
        tmp_yaml.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        config = BenchmarkConfig.from_yaml_file(tmp_yaml)
        with caplog.at_level("INFO"):
            run_benchmark(config, TestMode.PERF)
        assert "Completed in" in caplog.text
