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

"""Tests for benchmark CLI models, config building, and command handlers."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.commands.benchmark.cli import (
    from_config,
    offline,
    online,
)
from inference_endpoint.commands.benchmark.execute import ResponseCollector
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    DatasetType,
    LoadPattern,
    LoadPatternType,
    OfflineSettings,
    OnlineSettings,
    RuntimeConfig,
    StreamingMode,
    TestMode,
    TestType,
)
from inference_endpoint.config.schema import (
    OfflineBenchmarkConfig as OfflineConfig,
)
from inference_endpoint.config.schema import (
    OnlineBenchmarkConfig as OnlineConfig,
)
from inference_endpoint.core.types import QueryResult
from inference_endpoint.exceptions import InputValidationError
from pydantic import ValidationError

TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "inference_endpoint"
    / "config"
    / "templates"
)

# Reusable minimal config kwargs
_OFFLINE_KWARGS = {
    "endpoint_config": {"endpoints": ["http://test:8000"]},
    "model_params": {"name": "test-model"},
    "datasets": [{"path": "test.pkl"}],
}


class TestCLIConfigModels:
    """Test OfflineBenchmarkConfig/OnlineBenchmarkConfig defaults and validation."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cls, extra_kwargs, expected_type, expected_streaming",
        [
            (OfflineConfig, {}, TestType.OFFLINE, StreamingMode.OFF),
            (
                OnlineConfig,
                {
                    "settings": OnlineSettings(
                        load_pattern=LoadPattern(
                            type=LoadPatternType.POISSON, target_qps=100
                        ),
                    ),
                },
                TestType.ONLINE,
                StreamingMode.ON,
            ),
        ],
    )
    def test_mode_defaults(self, cls, extra_kwargs, expected_type, expected_streaming):
        config = cls(**_OFFLINE_KWARGS, **extra_kwargs)
        assert config.type == expected_type
        assert config.model_params.streaming == expected_streaming
        assert config.settings.runtime.min_duration_ms == 600000

    @pytest.mark.unit
    def test_num_samples_override(self):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(
                runtime=RuntimeConfig(min_duration_ms=0, n_samples_to_issue=100)
            ),
        )
        assert config.settings.runtime.n_samples_to_issue == 100

    @pytest.mark.unit
    def test_missing_model_name_raises(self):
        with pytest.raises(ValidationError, match="model"):
            OfflineConfig(
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "test.pkl"}],
            )


class TestDatasetParsing:
    """Test dataset string coercion through BenchmarkConfig construction."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw, path, dtype, samples, parser, acc_eval_method",
        [
            ("test.pkl", "test.pkl", DatasetType.PERFORMANCE, None, None, None),
            ("perf:a.pkl", "a.pkl", DatasetType.PERFORMANCE, None, None, None),
            ("acc:gpqa.pkl", "gpqa.pkl", DatasetType.ACCURACY, None, None, None),
            (
                "data.csv,samples=500,parser.prompt=article,parser.system=inst",
                "data.csv",
                DatasetType.PERFORMANCE,
                500,
                {"prompt": "article", "system": "inst"},  # {target: source}
                None,
            ),
            (
                "perf:d.jsonl,format=jsonl,parser.prompt=text",
                "d.jsonl",
                DatasetType.PERFORMANCE,
                None,
                {"prompt": "text"},  # {target: source}
                None,
            ),
            (
                "acc:eval.pkl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer",
                "eval.pkl",
                DatasetType.ACCURACY,
                None,
                None,
                "pass_at_1",
            ),
        ],
    )
    def test_dataset_string_coercion(
        self, raw, path, dtype, samples, parser, acc_eval_method
    ):
        """Strings passed as datasets are parsed by BeforeValidator into Dataset objects."""
        config = OfflineConfig(**_OFFLINE_KWARGS | {"datasets": [raw]})
        ds = config.datasets[0]
        assert ds.path == path
        assert ds.type == dtype
        assert ds.samples == samples
        assert ds.parser == parser
        if acc_eval_method:
            assert ds.accuracy_config is not None
            assert ds.accuracy_config.eval_method == acc_eval_method


class TestCommandHandlers:
    """Test offline/online/from_config handlers (mock run_benchmark)."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "handler, config, dataset_arg, mode, expected_path, expected_dtype",
        [
            (
                offline,
                OfflineConfig(
                    endpoint_config={"endpoints": ["http://x"]},
                    model_params={"name": "M"},
                ),
                ["data.pkl"],
                TestMode.PERF,
                "data.pkl",
                DatasetType.PERFORMANCE,
            ),
            (
                online,
                OnlineConfig(
                    endpoint_config={"endpoints": ["http://x"]},
                    model_params={"name": "M"},
                    settings={"load_pattern": {"type": "poisson", "target_qps": 10}},
                ),
                ["acc:eval.pkl"],
                TestMode.ACC,
                "eval.pkl",
                DatasetType.ACCURACY,
            ),
        ],
    )
    @patch("inference_endpoint.commands.benchmark.cli.run_benchmark")
    def test_command_handler(
        self,
        mock_run,
        handler,
        config,
        dataset_arg,
        mode,
        expected_path,
        expected_dtype,
    ):
        handler(config=config, dataset=dataset_arg, mode=mode)
        called_config, called_mode = mock_run.call_args[0]
        assert called_config.datasets[0].path == expected_path
        assert called_config.datasets[0].type == expected_dtype
        assert called_mode == mode

    @pytest.mark.unit
    @patch("inference_endpoint.commands.benchmark.cli.run_benchmark")
    def test_from_config_handler(self, mock_run, tmp_path):
        yaml_content = """
type: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://test:8000"]
datasets:
  - path: "test.pkl"
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        from_config(config=config_file, timeout=42.0, mode=TestMode.BOTH)
        called_config, called_mode = mock_run.call_args[0]
        assert called_config.timeout == 42.0
        assert called_mode == TestMode.BOTH

    @pytest.mark.unit
    def test_from_config_bad_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml")
        with pytest.raises(InputValidationError, match="Config error"):
            from_config(config=bad_file)

    @pytest.mark.unit
    @patch("inference_endpoint.commands.benchmark.cli.run_benchmark")
    def test_from_config_submission_defaults_to_both(self, mock_run, tmp_path):
        yaml_content = """
type: "submission"
benchmark_mode: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://test:8000"]
datasets:
  - path: "test.pkl"
submission_ref:
  model: "test-model"
  ruleset: "test"
"""
        config_file = tmp_path / "sub.yaml"
        config_file.write_text(yaml_content)
        from_config(config=config_file)
        _, called_mode = mock_run.call_args[0]
        assert called_mode == TestMode.BOTH


class TestBenchmarkValidation:
    """Test BenchmarkConfig validation paths."""

    @pytest.mark.unit
    def test_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
type: "offline"
model_params:
  name: "test-model"
datasets:
  - path: "tests/datasets/dummy_1k.pkl"
endpoint_config:
  endpoints: ["http://test:8000"]
""")
            config_path = Path(f.name)
        try:
            config = BenchmarkConfig.from_yaml_file(config_path)
            assert config.endpoint_config.endpoints == ["http://test:8000"]
        finally:
            config_path.unlink()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "overrides, match",
        [
            (
                {
                    "type": TestType.ONLINE,
                    "settings": {"load_pattern": {"type": "poisson"}},
                },
                "requires --target-qps",
            ),
            (
                {
                    "type": TestType.ONLINE,
                    "settings": {"load_pattern": {"type": "concurrency"}},
                },
                "requires --concurrency",
            ),
            (
                {"type": TestType.OFFLINE, "settings": {"client": {"workers": 0}}},
                "workers must be",
            ),
            (
                {
                    "type": TestType.SUBMISSION,
                    "submission_ref": {"model": "M", "ruleset": "R"},
                },
                "benchmark_mode",
            ),
        ],
    )
    def test_validation_errors(self, overrides, match):
        with pytest.raises((ValueError, ValidationError), match=match):
            BenchmarkConfig(
                endpoint_config={"endpoints": ["http://x"]},
                model_params={"name": "M"},
                datasets=[{"path": "test.pkl"}],
                **overrides,
            )


class TestYAMLTemplateValidation:
    """Validate all bundled YAML templates parse correctly."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "template",
        [
            "concurrency_template.yaml",
            "eval_template.yaml",
            "offline_template.yaml",
            "online_template.yaml",
            "submission_template.yaml",
        ],
    )
    def test_valid_templates_parse(self, template):
        config = BenchmarkConfig.from_yaml_file(TEMPLATE_DIR / template)
        assert config.model_params.name
        assert config.endpoint_config.endpoints


class TestResponseCollector:
    @pytest.mark.unit
    def test_success_response(self):
        collector = ResponseCollector(collect_responses=True)
        result = QueryResult(id="q1", error=None, response_output="hello")
        collector.on_complete_hook(result)
        assert collector.count == 1
        assert not collector.errors
        assert "q1" in collector.responses

    @pytest.mark.unit
    def test_error_response(self):
        collector = ResponseCollector()
        result = QueryResult(id="q1", error="timeout")
        collector.on_complete_hook(result)
        assert collector.count == 1
        assert len(collector.errors) == 1
        assert "timeout" in collector.errors[0]

    @pytest.mark.unit
    def test_no_collect_skips_responses(self):
        collector = ResponseCollector(collect_responses=False)
        result = QueryResult(id="q1", error=None, response_output="hello")
        collector.on_complete_hook(result)
        assert collector.count == 1
        assert not collector.responses


class TestErrorFormatter:
    """Test _error_formatter in main.py."""

    @pytest.mark.unit
    def test_cyclopts_arg_with_children(self):
        from inference_endpoint.config.utils import (
            cli_error_formatter as _error_formatter,
        )

        child = SimpleNamespace(
            name="--endpoints", names=("--endpoints",), required=True, has_tokens=False
        )
        arg = SimpleNamespace(name="--endpoint-config", children=[child])
        err = MagicMock(spec=["argument"])
        err.argument = arg
        panel = _error_formatter(err)
        assert "Required: --endpoints" in panel.renderable

    @pytest.mark.unit
    def test_cyclopts_leaf_arg(self):
        from inference_endpoint.config.utils import (
            cli_error_formatter as _error_formatter,
        )

        arg = SimpleNamespace(
            name="--model", names=("--model-params.name", "--model"), children=[]
        )
        err = MagicMock(spec=["argument"])
        err.argument = arg
        panel = _error_formatter(err)
        assert "Required: --model" in panel.renderable

    @pytest.mark.unit
    def test_pydantic_validation_error(self):
        from inference_endpoint.config.utils import (
            cli_error_formatter as _error_formatter,
        )

        try:
            BenchmarkConfig(
                type=TestType.OFFLINE,
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
            )
        except Exception as cause:
            err = MagicMock(spec=[])
            err.__cause__ = cause
            panel = _error_formatter(err)
            assert "model" in panel.renderable.lower()

    @pytest.mark.unit
    def test_generic_error_fallback(self):
        from inference_endpoint.config.utils import (
            cli_error_formatter as _error_formatter,
        )

        class FakeError:
            argument = None
            __cause__ = None
            __context__ = None

            def __str__(self):
                return "something went wrong"

        panel = _error_formatter(FakeError())
        assert "something went wrong" in panel.renderable
