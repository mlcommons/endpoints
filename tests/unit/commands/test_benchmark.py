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

import asyncio
import io
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib import error as urllib_error

import pandas as pd
import pytest
from inference_endpoint.commands.benchmark.cli import (
    benchmark_app,
    from_config,
    offline,
    online,
)
from inference_endpoint.commands.benchmark.execute import (
    AccuracyConfiguration,
    BenchmarkContext,
    ResponseCollector,
    _build_phases,
    _derive_profile_urls,
    _post_profile,
    _render_profile_status,
    _run_benchmark_async,
    _write_profiling_section,
    setup_benchmark,
)
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    DatasetType,
    DrainConfig,
    LoadPattern,
    LoadPatternType,
    OfflineSettings,
    OnlineSettings,
    ProfilerEngine,
    RuntimeConfig,
    ScorerMethod,
    StreamingMode,
    TestMode,
    TestType,
    WarmupConfig,
)
from inference_endpoint.config.schema import (
    OfflineBenchmarkConfig as OfflineConfig,
)
from inference_endpoint.config.schema import (
    OnlineBenchmarkConfig as OnlineConfig,
)
from inference_endpoint.config.utils import cli_error_formatter as _error_formatter
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import InputValidationError, SetupError
from inference_endpoint.load_generator.sample_order import create_sample_order
from inference_endpoint.load_generator.session import PhaseType
from inference_endpoint.metrics.metric import Throughput
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
    "datasets": [{"path": "test.jsonl"}],
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
                datasets=[{"path": "test.jsonl"}],
            )


class TestDurationSuffix:
    """Test duration suffix parsing (600s, 10m, 600000ms, plain int)."""

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
    def test_duration_suffix(self, value, expected_ms):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(runtime=RuntimeConfig(min_duration_ms=value)),
        )
        assert config.settings.runtime.min_duration_ms == expected_ms


class TestDatasetParsing:
    """Test dataset string coercion through BenchmarkConfig construction."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw, path, dtype, samples, parser, acc_eval_method",
        [
            ("test.jsonl", "test.jsonl", DatasetType.PERFORMANCE, None, None, None),
            ("perf:a.jsonl", "a.jsonl", DatasetType.PERFORMANCE, None, None, None),
            ("acc:gpqa.jsonl", "gpqa.jsonl", DatasetType.ACCURACY, None, None, None),
            (
                "data.csv,samples=500,parser.prompt=article,parser.system=inst",
                "data.csv",
                DatasetType.PERFORMANCE,
                500,
                {"prompt": "article", "system": "inst"},  # {target: source}
                None,
            ),
            (
                "perf:d.jsonl,format=.jsonl,parser.prompt=text",
                "d.jsonl",
                DatasetType.PERFORMANCE,
                None,
                {"prompt": "text"},  # {target: source}
                None,
            ),
            (
                "acc:eval.jsonl,accuracy_config.eval_method=pass_at_1,accuracy_config.ground_truth=answer",
                "eval.jsonl",
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
                    settings=OfflineSettings(
                        client=HTTPClientConfig(
                            num_workers=1, warmup_connections=0, max_connections=10
                        ),
                    ),
                ),
                ["data.jsonl"],
                TestMode.PERF,
                "data.jsonl",
                DatasetType.PERFORMANCE,
            ),
            (
                online,
                OnlineConfig(
                    endpoint_config={"endpoints": ["http://x"]},
                    model_params={"name": "M"},
                    settings=OnlineSettings(
                        load_pattern=LoadPattern(
                            type=LoadPatternType.POISSON, target_qps=10
                        ),
                        client=HTTPClientConfig(
                            num_workers=1, warmup_connections=0, max_connections=10
                        ),
                    ),
                ),
                ["acc:eval.jsonl"],
                TestMode.ACC,
                "eval.jsonl",
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
    def test_use_legacy_loadgen_qps_metrics_default_and_disable(self):
        """LoadPattern flag defaults True; --no-use-legacy-loadgen-qps-metrics
        sets False (poisson only).
        """
        base = [
            "online",
            "--endpoints",
            "http://h:80",
            "--model",
            "m",
            "--dataset",
            "d.jsonl",
            "--load-pattern",
            "poisson",
            "--target-qps",
            "100",
        ]
        _, bound, _ = benchmark_app.parse_args(base, exit_on_error=False)
        lp = bound.arguments["config"].settings.load_pattern
        assert lp.use_legacy_loadgen_qps_metrics is True

        _, bound, _ = benchmark_app.parse_args(
            [*base, "--no-use-legacy-loadgen-qps-metrics"], exit_on_error=False
        )
        lp = bound.arguments["config"].settings.load_pattern
        assert lp.use_legacy_loadgen_qps_metrics is False

    @pytest.mark.unit
    def test_loadgen_flag_serialized_only_for_poisson(self):
        """``use_legacy_loadgen_qps_metrics`` is dropped from the serialized
        form for non-poisson patterns (so it does not pollute their YAML
        templates), and present for poisson.
        """
        poisson = LoadPattern(type=LoadPatternType.POISSON, target_qps=100)
        assert "use_legacy_loadgen_qps_metrics" in poisson.model_dump()

        for lp in (
            LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            LoadPattern(type=LoadPatternType.CONCURRENCY, target_concurrency=10),
        ):
            assert "use_legacy_loadgen_qps_metrics" not in lp.model_dump()

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
  - path: "test.jsonl"
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
  - path: "test.jsonl"
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
  - path: "tests/assets/datasets/dummy_1k.jsonl"
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
                {"type": TestType.OFFLINE, "settings": {"client": {"num_workers": 0}}},
                "num_workers must be",
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
                datasets=[{"path": "test.jsonl"}],
                **overrides,
            )


class TestYAMLTemplateValidation:
    """Validate all bundled YAML templates parse correctly."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "template",
        sorted(
            p.name
            for p in (
                Path(__file__).parent.parent.parent.parent
                / "src"
                / "inference_endpoint"
                / "config"
                / "templates"
            ).glob("*_template*.yaml")
        ),
    )
    def test_valid_templates_parse(self, template):
        config = BenchmarkConfig.from_yaml_file(TEMPLATE_DIR / template)
        assert config.model_params.name
        assert config.endpoint_config.endpoints


class TestWarmupConfig:
    """Tests for WarmupConfig schema model."""

    @pytest.mark.unit
    def test_defaults(self):
        cfg = WarmupConfig()
        assert cfg.enabled is False
        assert cfg.n_requests is None
        assert cfg.salt is True
        assert cfg.drain is False

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [1, 10, 1000])
    def test_n_requests_valid(self, n):
        cfg = WarmupConfig(n_requests=n)
        assert cfg.n_requests == n

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [0, -1, -100])
    def test_n_requests_must_be_positive(self, n):
        with pytest.raises(ValidationError):
            WarmupConfig(n_requests=n)

    @pytest.mark.unit
    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            WarmupConfig(unknown_field=True)

    @pytest.mark.unit
    def test_immutable(self):
        cfg = WarmupConfig()
        with pytest.raises(ValidationError):
            cfg.enabled = True  # type: ignore[misc]

    @pytest.mark.unit
    def test_all_flags_enabled(self):
        cfg = WarmupConfig(enabled=True, n_requests=50, salt=True, drain=True)
        assert cfg.enabled is True
        assert cfg.n_requests == 50
        assert cfg.salt is True
        assert cfg.drain is True

    @pytest.mark.unit
    def test_yaml_roundtrip(self, tmp_path):
        yaml_content = """
type: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://test:8000"]
datasets:
  - path: "test.jsonl"
settings:
  warmup:
    enabled: true
    n_requests: 20
    salt: true
    drain: true
"""
        config_file = tmp_path / "warmup.yaml"
        config_file.write_text(yaml_content)
        config = BenchmarkConfig.from_yaml_file(config_file)
        warmup = config.settings.warmup
        assert warmup.enabled is True
        assert warmup.n_requests == 20
        assert warmup.salt is True
        assert warmup.drain is True

    @pytest.mark.unit
    def test_warmup_default_in_settings(self):
        config = OfflineConfig(**_OFFLINE_KWARGS)
        warmup = config.settings.warmup
        assert warmup.enabled is False
        assert warmup.n_requests is None


class TestDrainConfig:
    """Tests for DrainConfig schema model."""

    @pytest.mark.unit
    def test_defaults(self):
        cfg = DrainConfig()
        assert cfg.warmup_timeout_s == 240.0
        assert cfg.performance_timeout_s == 240.0
        assert cfg.accuracy_timeout_s is None
        assert cfg.metrics_drain_timeout_s == 60.0
        assert cfg.metrics_tokenizer_workers == 2

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field",
        ["warmup_timeout_s", "performance_timeout_s", "accuracy_timeout_s"],
    )
    @pytest.mark.parametrize("value", [0, -1.0])
    def test_timeout_must_be_positive_or_none(self, field, value):
        with pytest.raises(ValidationError):
            DrainConfig(**{field: value})

    @pytest.mark.unit
    def test_metrics_drain_timeout_zero_is_valid(self):
        cfg = DrainConfig(metrics_drain_timeout_s=0)
        assert cfg.metrics_drain_timeout_s == 0.0

    @pytest.mark.unit
    def test_metrics_drain_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            DrainConfig(metrics_drain_timeout_s=-1.0)

    @pytest.mark.unit
    def test_metrics_tokenizer_workers_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            DrainConfig(metrics_tokenizer_workers=0)

    @pytest.mark.unit
    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DrainConfig(unknown_field=1)

    @pytest.mark.unit
    def test_yaml_roundtrip(self, tmp_path):
        yaml_content = """
type: "offline"
model_params:
  name: "test-model"
endpoint_config:
  endpoints: ["http://test:8000"]
datasets:
  - path: "test.jsonl"
settings:
  drain:
    warmup_timeout_s: 12.5
    performance_timeout_s: 30.0
    accuracy_timeout_s: null
    metrics_drain_timeout_s: 300.0
    metrics_tokenizer_workers: 8
"""
        config_file = tmp_path / "drain.yaml"
        config_file.write_text(yaml_content)
        config = BenchmarkConfig.from_yaml_file(config_file)
        drain = config.settings.drain
        assert drain.warmup_timeout_s == 12.5
        assert drain.performance_timeout_s == 30.0
        assert drain.accuracy_timeout_s is None
        assert drain.metrics_drain_timeout_s == 300.0
        assert drain.metrics_tokenizer_workers == 8


class TestAggregatorArgs:
    """Tests that metrics aggregator subprocess args are correctly forwarded."""

    def _make_ctx(self, config, tmp_path):
        import random

        rt = RuntimeSettings(
            metric_target=Throughput(10.0),
            reported_metrics=[Throughput(10.0)],
            min_duration_ms=0,
            max_duration_ms=None,
            n_samples_from_dataset=1,
            n_samples_to_issue=None,
            min_sample_count=1,
            rng_sched=random.Random(0),
            rng_sample_index=random.Random(0),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        df = pd.DataFrame({"prompt": ["q0"]})
        ds = Dataset(df)
        ds.load()
        return BenchmarkContext(
            config=config,
            test_mode=TestMode.PERF,
            report_dir=tmp_path,
            tokenizer_name=None,
            dataloader=ds,
            rt_settings=rt,
            total_samples=1,
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "timeout_s, expected_flag",
        [(120.0, "120.0"), (0.0, "0.0"), (60.0, "60.0")],
    )
    async def test_drain_timeout_forwarded_to_aggregator_args(
        self, tmp_path, timeout_s, expected_flag
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(
                drain=DrainConfig(metrics_drain_timeout_s=timeout_s)
            ),
        )
        ctx = self._make_ctx(config, tmp_path)

        captured: list = []

        async def _capture_launch(service_configs, *, timeout):
            captured.extend(service_configs)
            raise KeyboardInterrupt("stop after launch")

        mock_zmq = MagicMock()
        mock_zmq.socket_dir = str(tmp_path / "sockets")

        with (
            patch(
                "inference_endpoint.commands.benchmark.execute.ManagedZMQContext"
            ) as MockZMQ,
            patch(
                "inference_endpoint.commands.benchmark.execute.EventPublisherService"
            ) as MockPub,
            patch(
                "inference_endpoint.commands.benchmark.execute.MetricsSnapshotSubscriber"
            ) as MockSub,
            patch(
                "inference_endpoint.commands.benchmark.execute.ServiceLauncher"
            ) as MockLauncher,
            patch("inference_endpoint.commands.benchmark.execute.tqdm"),
        ):
            MockZMQ.scoped.return_value.__enter__ = MagicMock(return_value=mock_zmq)
            MockZMQ.scoped.return_value.__exit__ = MagicMock(return_value=False)
            MockPub.return_value.socket_name = "test_pub"
            MockSub.return_value.start = MagicMock()
            MockLauncher.return_value.launch = _capture_launch

            loop = asyncio.get_event_loop()
            with pytest.raises(KeyboardInterrupt):
                await _run_benchmark_async(ctx, loop)

        aggregator_cfg = next(c for c in captured if "metrics_aggregator" in c.module)
        args = aggregator_cfg.args
        assert "--drain-timeout" in args
        idx = args.index("--drain-timeout")
        assert args[idx + 1] == expected_flag

    @pytest.mark.unit
    @pytest.mark.asyncio
    @pytest.mark.parametrize("workers, expected_flag", [(4, "4"), (8, "8"), (2, "2")])
    async def test_tokenizer_workers_forwarded_to_aggregator_args(
        self, tmp_path, workers, expected_flag
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(
                drain=DrainConfig(metrics_tokenizer_workers=workers)
            ),
        )
        ctx = self._make_ctx(config, tmp_path)

        captured: list = []

        async def _capture_launch(service_configs, *, timeout):
            captured.extend(service_configs)
            raise KeyboardInterrupt("stop after launch")

        mock_zmq = MagicMock()
        mock_zmq.socket_dir = str(tmp_path / "sockets")

        with (
            patch(
                "inference_endpoint.commands.benchmark.execute.ManagedZMQContext"
            ) as MockZMQ,
            patch(
                "inference_endpoint.commands.benchmark.execute.EventPublisherService"
            ) as MockPub,
            patch(
                "inference_endpoint.commands.benchmark.execute.MetricsSnapshotSubscriber"
            ) as MockSub,
            patch(
                "inference_endpoint.commands.benchmark.execute.ServiceLauncher"
            ) as MockLauncher,
            patch("inference_endpoint.commands.benchmark.execute.tqdm"),
        ):
            MockZMQ.scoped.return_value.__enter__ = MagicMock(return_value=mock_zmq)
            MockZMQ.scoped.return_value.__exit__ = MagicMock(return_value=False)
            MockPub.return_value.socket_name = "test_pub"
            MockSub.return_value.start = MagicMock()
            MockLauncher.return_value.launch = _capture_launch

            loop = asyncio.get_event_loop()
            with pytest.raises(KeyboardInterrupt):
                await _run_benchmark_async(ctx, loop)

        aggregator_cfg = next(c for c in captured if "metrics_aggregator" in c.module)
        args = aggregator_cfg.args
        assert "--tokenizer-workers" in args
        idx = args.index("--tokenizer-workers")
        assert args[idx + 1] == expected_flag


class TestBuildPhases:
    """Tests for _build_phases() in execute.py."""

    @pytest.fixture
    def base_rt_settings(self):
        return RuntimeSettings(
            metric_target=Throughput(10.0),
            reported_metrics=[Throughput(10.0)],
            min_duration_ms=600000,
            max_duration_ms=None,
            n_samples_from_dataset=5,
            n_samples_to_issue=None,
            min_sample_count=1,
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )

    @pytest.fixture
    def simple_dataset(self):
        df = pd.DataFrame({"prompt": [f"q{i}" for i in range(5)]})
        ds = Dataset(df)
        ds.load()
        return ds

    def _make_ctx(self, config, rt_settings, dataloader):
        return BenchmarkContext(
            config=config,
            test_mode=TestMode.PERF,
            report_dir=Path("/tmp"),
            tokenizer_name=None,
            dataloader=dataloader,
            rt_settings=rt_settings,
            total_samples=dataloader.num_samples(),
            accuracy_datasets=[],
            eval_configs=[],
        )

    def _make_eval_config(self, dataset):
        return AccuracyConfiguration(
            scorer=Scorer.get("pass_at_1"),
            extractor=None,
            dataset_name="accuracy",
            dataset=dataset,
            report_dir=Path("/tmp"),
            ground_truth_column=None,
            num_repeats=dataset.repeats,
        )

    @pytest.mark.unit
    def test_warmup_disabled_produces_only_perf_phase(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(**_OFFLINE_KWARGS)
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert len(phases) == 1
        assert phases[0].phase_type == PhaseType.PERFORMANCE

    @pytest.mark.unit
    def test_warmup_enabled_produces_two_phases(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert len(phases) == 2
        assert phases[0].phase_type == PhaseType.WARMUP
        assert phases[1].phase_type == PhaseType.PERFORMANCE

    @pytest.mark.unit
    def test_warmup_phase_named_warmup(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].name == "warmup"

    @pytest.mark.unit
    def test_warmup_phase_uses_max_throughput(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        warmup_rt = phases[0].runtime_settings
        assert warmup_rt.load_pattern is not None
        assert warmup_rt.load_pattern.type == LoadPatternType.MAX_THROUGHPUT

    @pytest.mark.unit
    def test_warmup_phase_min_duration_is_zero(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].runtime_settings.min_duration_ms == 0

    @pytest.mark.unit
    def test_warmup_phase_no_max_duration(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].runtime_settings.max_duration_ms is None

    @pytest.mark.unit
    def test_warmup_n_requests_propagated(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, n_requests=7)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].runtime_settings.n_samples_to_issue == 7

    @pytest.mark.unit
    def test_warmup_n_requests_none_when_unset(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(
                warmup=WarmupConfig(enabled=True, n_requests=None)
            ),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].runtime_settings.n_samples_to_issue is None

    @pytest.mark.unit
    def test_warmup_defaults_uses_salt(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].dataset._salt_rng is not None

    @pytest.mark.unit
    def test_warmup_without_salt_uses_raw_dataloader(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, salt=False)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].dataset._salt_rng is None
        assert phases[0].dataset is simple_dataset

    @pytest.mark.unit
    def test_warmup_with_salt_uses_salted_dataset(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, salt=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].dataset._salt_rng is not None

    @pytest.mark.unit
    def test_warmup_drain_false_by_default(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, drain=False)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].drain_after is False

    @pytest.mark.unit
    def test_warmup_drain_true_propagated(self, base_rt_settings, simple_dataset):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, drain=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[0].drain_after is True

    @pytest.mark.unit
    def test_warmup_n_samples_from_dataset_matches_dataloader(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert (
            phases[0].runtime_settings.n_samples_from_dataset
            == simple_dataset.num_samples()
        )

    @pytest.mark.unit
    def test_performance_phase_dataset_is_always_raw_dataloader(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, salt=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        perf_phase = phases[1]
        assert perf_phase.dataset is simple_dataset

    @pytest.mark.unit
    def test_performance_phase_uses_original_rt_settings(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        assert phases[1].runtime_settings is base_rt_settings

    @pytest.mark.unit
    def test_configured_drain_timeouts_propagate_to_phases(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(
                drain=DrainConfig(
                    warmup_timeout_s=7.0,
                    performance_timeout_s=15.0,
                    accuracy_timeout_s=45.0,
                ),
                warmup=WarmupConfig(enabled=True, drain=True),
            ),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        ctx.eval_configs = [self._make_eval_config(simple_dataset)]
        phases = _build_phases(ctx)

        warmup = next(p for p in phases if p.phase_type == PhaseType.WARMUP)
        perf = next(p for p in phases if p.phase_type == PhaseType.PERFORMANCE)
        acc = next(p for p in phases if p.phase_type == PhaseType.ACCURACY)
        assert warmup.drain_timeout == 7.0
        assert perf.drain_timeout == 15.0
        assert acc.drain_timeout == 45.0

    @pytest.mark.unit
    def test_accuracy_drain_timeout_defaults_to_unbounded(
        self, base_rt_settings, simple_dataset
    ):
        config = OfflineConfig(**_OFFLINE_KWARGS)
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        ctx.eval_configs = [self._make_eval_config(simple_dataset)]
        phases = _build_phases(ctx)

        acc = next(p for p in phases if p.phase_type == PhaseType.ACCURACY)
        assert acc.drain_timeout is None

    @pytest.mark.unit
    def test_warmup_uses_independent_rng_instances(
        self, base_rt_settings, simple_dataset
    ):
        """Warmup RuntimeSettings must not share RNG instances with the perf phase.

        Sharing would cause warmup sample-ordering to consume state from the
        perf phase's deterministic random sequence, breaking reproducibility.
        """
        config = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True)),
        )
        ctx = self._make_ctx(config, base_rt_settings, simple_dataset)
        phases = _build_phases(ctx)

        warmup_rt = phases[0].runtime_settings
        perf_rt = phases[1].runtime_settings
        assert warmup_rt.rng_sched is not perf_rt.rng_sched
        assert warmup_rt.rng_sample_index is not perf_rt.rng_sample_index

    @pytest.mark.unit
    def test_performance_sample_order_identical_with_and_without_warmup(
        self, simple_dataset
    ):
        """Warmup must not perturb the performance phase's sample ordering.

        Both runs use separate RuntimeSettings instances seeded identically so
        the comparison is valid. If warmup ever accidentally shared or advanced
        the perf-phase RNG, the two sequences would diverge.
        """
        n_draw = 20

        def make_rt():
            return RuntimeSettings(
                metric_target=Throughput(10.0),
                reported_metrics=[Throughput(10.0)],
                min_duration_ms=0,
                max_duration_ms=None,
                n_samples_from_dataset=simple_dataset.num_samples(),
                n_samples_to_issue=None,
                min_sample_count=1,
                rng_sched=random.Random(99),
                rng_sample_index=random.Random(99),
                load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
            )

        config_with = OfflineConfig(
            **_OFFLINE_KWARGS,
            settings=OfflineSettings(warmup=WarmupConfig(enabled=True, n_requests=5)),
        )
        config_without = OfflineConfig(**_OFFLINE_KWARGS)

        ctx_with = self._make_ctx(config_with, make_rt(), simple_dataset)
        ctx_without = self._make_ctx(config_without, make_rt(), simple_dataset)

        perf_with = next(
            p for p in _build_phases(ctx_with) if p.phase_type == PhaseType.PERFORMANCE
        )
        perf_without = next(
            p
            for p in _build_phases(ctx_without)
            if p.phase_type == PhaseType.PERFORMANCE
        )

        order_with = [
            next(create_sample_order(perf_with.runtime_settings)) for _ in range(n_draw)
        ]
        order_without = [
            next(create_sample_order(perf_without.runtime_settings))
            for _ in range(n_draw)
        ]

        assert order_with == order_without, (
            "Performance sample order changed when warmup is enabled — "
            "warmup may be sharing or advancing the perf-phase RNG."
        )


class TestScorerMethodSync:
    """Ensure ScorerMethod enum stays in sync with the scorer registry."""

    @pytest.mark.unit
    def test_scorer_enum_matches_registry(self):
        enum_values = {m.value for m in ScorerMethod}
        registry_keys = set(Scorer.PREDEFINED.keys())
        assert enum_values == registry_keys, (
            f"ScorerMethod enum out of sync with Scorer registry.\n"
            f"  In enum only: {enum_values - registry_keys}\n"
            f"  In registry only: {registry_keys - enum_values}"
        )


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
        arg = SimpleNamespace(
            name="--model", names=("--model-params.name", "--model"), children=[]
        )
        err = MagicMock(spec=["argument"])
        err.argument = arg
        panel = _error_formatter(err)
        assert "Required: --model" in panel.renderable

    @pytest.mark.unit
    def test_pydantic_validation_error(self):
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
        class FakeError:
            argument = None
            __cause__ = None
            __context__ = None

            def __str__(self):
                return "something went wrong"

        panel = _error_formatter(FakeError())
        assert "something went wrong" in panel.renderable


class TestSetupBenchmarkTokenizer:
    """Tests for tokenizer resolution logic in setup_benchmark."""

    @pytest.fixture()
    def _base_patches(self):
        with (
            patch(
                "inference_endpoint.commands.benchmark.execute.pin_loadgen",
                return_value=None,
            ),
            patch(
                "inference_endpoint.config.schema.BenchmarkConfig.to_yaml_file",
                return_value=None,
            ),
        ):
            yield

    @pytest.fixture()
    def _simple_dataset(self):
        ds = Dataset(pd.DataFrame({"prompt": ["q0"]}))
        ds.load()
        return ds

    @pytest.fixture()
    def _rt_settings(self):
        return RuntimeSettings(
            metric_target=Throughput(10.0),
            reported_metrics=[Throughput(10.0)],
            min_duration_ms=0,
            max_duration_ms=None,
            n_samples_from_dataset=1,
            n_samples_to_issue=None,
            min_sample_count=1,
            rng_sched=random.Random(0),
            rng_sample_index=random.Random(0),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )

    @pytest.mark.unit
    def test_invalid_tokenizer_override_raises(self, tmp_path, _base_patches):
        config = OfflineConfig(
            **_OFFLINE_KWARGS
            | {
                "model_params": {"name": "test-model", "tokenizer_name": "bad/override"}
            },
            report_dir=str(tmp_path),
        )
        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._check_tokenizer_exists",
                return_value=False,
            ),
            pytest.raises(SetupError, match="bad/override"),
        ):
            setup_benchmark(config, TestMode.PERF)

    @pytest.mark.unit
    def test_valid_tokenizer_override_stored_in_context(
        self, tmp_path, _base_patches, _simple_dataset, _rt_settings
    ):
        config = OfflineConfig(
            **_OFFLINE_KWARGS
            | {
                "model_params": {
                    "name": "test-model",
                    "tokenizer_name": "good/override",
                }
            },
            report_dir=str(tmp_path),
        )
        mock_check = MagicMock(return_value=True)
        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._check_tokenizer_exists",
                mock_check,
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute._load_datasets",
                return_value=(_simple_dataset, [], []),
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute.RuntimeSettings.from_config",
                return_value=_rt_settings,
            ),
        ):
            ctx = setup_benchmark(config, TestMode.PERF)

        mock_check.assert_called_once_with("good/override")
        assert ctx.tokenizer_name == "good/override"

    @pytest.mark.unit
    def test_no_override_uses_model_name_when_tokenizer_exists(
        self, tmp_path, _base_patches, _simple_dataset, _rt_settings
    ):
        config = OfflineConfig(**_OFFLINE_KWARGS, report_dir=str(tmp_path))
        mock_check = MagicMock(return_value=True)
        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._check_tokenizer_exists",
                mock_check,
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute._load_datasets",
                return_value=(_simple_dataset, [], []),
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute.RuntimeSettings.from_config",
                return_value=_rt_settings,
            ),
        ):
            ctx = setup_benchmark(config, TestMode.PERF)

        mock_check.assert_called_once_with("test-model")
        assert ctx.tokenizer_name == "test-model"

    @pytest.mark.unit
    def test_no_override_yields_none_when_model_has_no_tokenizer(
        self, tmp_path, _base_patches, _simple_dataset, _rt_settings
    ):
        config = OfflineConfig(**_OFFLINE_KWARGS, report_dir=str(tmp_path))
        with (
            patch(
                "inference_endpoint.commands.benchmark.execute._check_tokenizer_exists",
                return_value=False,
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute._load_datasets",
                return_value=(_simple_dataset, [], []),
            ),
            patch(
                "inference_endpoint.commands.benchmark.execute.RuntimeSettings.from_config",
                return_value=_rt_settings,
            ),
        ):
            ctx = setup_benchmark(config, TestMode.PERF)

        assert ctx.tokenizer_name is None


class TestProfilingHelpers:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("http://h:8000/v1", "http://h:8000/start_profile"),
            ("http://h:8000/v1/", "http://h:8000/start_profile"),
            ("http://h:8000", "http://h:8000/start_profile"),
        ],
    )
    def test_derive_strips_v1(self, endpoint, expected):
        out = _derive_profile_urls([endpoint], ProfilerEngine.VLLM, "start")
        assert out == [expected]

    @pytest.mark.unit
    def test_derive_stop_path_and_fanout(self):
        out = _derive_profile_urls(
            ["http://a/v1", "http://b/v1"], ProfilerEngine.VLLM, "stop"
        )
        assert out == ["http://a/stop_profile", "http://b/stop_profile"]

    @pytest.mark.unit
    def test_derive_empty_endpoints_raises(self):
        with pytest.raises(ValueError):
            _derive_profile_urls([], ProfilerEngine.VLLM, "start")

    @pytest.mark.unit
    def test_post_profile_200(self):
        resp = MagicMock()
        resp.__enter__.return_value.status = 200
        with patch(
            "inference_endpoint.commands.benchmark.execute.urllib_request.urlopen",
            return_value=resp,
        ):
            rec = _post_profile("http://h/start_profile")
        assert rec["status"] == 200
        assert rec["error"] is None
        assert "sent_at_ns" in rec
        assert "sent_at_iso" in rec

    @pytest.mark.unit
    def test_post_profile_http_error(self):
        err = urllib_error.HTTPError("http://h", 404, "Not Found", {}, None)
        with patch(
            "inference_endpoint.commands.benchmark.execute.urllib_request.urlopen",
            side_effect=err,
        ):
            rec = _post_profile("http://h/start_profile")
        assert rec["status"] == 404
        assert "404" in rec["error"]

    @pytest.mark.unit
    def test_post_profile_connection_failure_never_raises(self):
        with patch(
            "inference_endpoint.commands.benchmark.execute.urllib_request.urlopen",
            side_effect=OSError("refused"),
        ):
            rec = _post_profile("http://h/start_profile")
        assert rec["status"] is None
        assert "OSError" in rec["error"]

    @pytest.mark.unit
    def test_render_status_200(self):
        assert _render_profile_status({"status": 200, "error": None}) == "200 OK"

    @pytest.mark.unit
    def test_render_status_404_hint(self):
        out = _render_profile_status({"status": 404, "error": "404 Not Found"})
        assert "profiling not enabled" in out

    @pytest.mark.unit
    def test_write_section_and_json_roundtrip(self):
        payload = {
            "engine": "vllm",
            "starts": [
                {
                    "url": "http://h/start_profile",
                    "status": 200,
                    "error": None,
                    "sent_at_ns": 1,
                    "sent_at_iso": "2026-01-01T00:00:00.000",
                }
            ],
            "stops": [
                {
                    "url": "http://h/stop_profile",
                    "status": 200,
                    "error": None,
                    "stop_reason": "phase_end",
                    "sent_at_ns": 2,
                    "sent_at_iso": "2026-01-01T00:00:01.000",
                }
            ],
        }
        buf = io.StringIO()
        _write_profiling_section(buf, payload)
        text = buf.getvalue()
        assert "Profiling" in text
        assert "http://h/start_profile" in text
        assert "http://h/stop_profile" in text
        assert "Trigger span" in text
        # Mirrors what finalize_benchmark dumps to profiling.json
        assert json.loads(json.dumps(payload))["engine"] == "vllm"
