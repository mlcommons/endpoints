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

"""Tests for configuration schema models and validation."""

import random

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    APIType,
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EvalMethod,
    LoadPattern,
    LoadPatternType,
    ModelParams,
    OSLDistribution,
    OSLDistributionType,
    StreamingMode,
    SubmissionReference,
    TestType,
)
from inference_endpoint.exceptions import CLIError


class TestOSLDistribution:
    @pytest.mark.unit
    def test_fixed_distribution(self):
        osl = OSLDistribution(type=OSLDistributionType.FIXED, max=1024)
        assert osl.type == OSLDistributionType.FIXED
        assert osl.max == 1024

    @pytest.mark.unit
    def test_normal_distribution(self):
        osl = OSLDistribution(
            type=OSLDistributionType.NORMAL, mean=1000, std=200, min=512, max=2048
        )
        assert osl.mean == 1000
        assert osl.std == 200

    @pytest.mark.unit
    def test_partial_construction_preserves_defaults(self):
        osl = OSLDistribution(min=10)
        assert osl.min == 10
        assert osl.type == OSLDistributionType.ORIGINAL
        assert osl.max == 2048
        assert ModelParams().osl_distribution is None


class TestModelParams:
    @pytest.mark.unit
    def test_defaults(self):
        params = ModelParams(name="test")
        assert params.temperature is None
        assert params.max_new_tokens == 1024

    @pytest.mark.unit
    def test_with_osl_distribution(self):
        params = ModelParams(
            name="test",
            temperature=0.5,
            top_k=50,
            top_p=0.9,
            max_new_tokens=2048,
            osl_distribution=OSLDistribution(
                type=OSLDistributionType.NORMAL, mean=1000, std=200
            ),
        )
        assert params.temperature == 0.5
        assert params.osl_distribution.type == OSLDistributionType.NORMAL


class TestAPIType:
    @pytest.mark.unit
    def test_default_routes(self):
        assert APIType.OPENAI.default_route() == "/v1/chat/completions"
        assert APIType.SGLANG.default_route() == "/generate"


class TestDataset:
    @pytest.mark.unit
    def test_performance_dataset(self):
        ds = Dataset(name="perf", type=DatasetType.PERFORMANCE, path="data.jsonl")
        assert ds.eval_method is None

    @pytest.mark.unit
    def test_accuracy_dataset(self):
        ds = Dataset(
            name="gpqa",
            type=DatasetType.ACCURACY,
            path="gpqa.jsonl",
            eval_method=EvalMethod.EXACT_MATCH,
        )
        assert ds.eval_method == EvalMethod.EXACT_MATCH

    @pytest.mark.unit
    def test_auto_derive_name(self):
        ds = Dataset(path="datasets/my_data.jsonl")
        assert ds.name == "my_data"


class TestBenchmarkConfig:
    @pytest.mark.unit
    def test_minimal_offline(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "test"},
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            datasets=[{"path": "test.jsonl"}],
        )
        assert config.type == TestType.OFFLINE

    @pytest.mark.unit
    def test_submission_with_ref(self):
        config = BenchmarkConfig(
            type=TestType.SUBMISSION,
            benchmark_mode=TestType.OFFLINE,
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            submission_ref=SubmissionReference(
                model="llama-2-70b", ruleset="mlperf-inference-v6.0"
            ),
            datasets=[{"path": "perf.jsonl"}],
        )
        assert config.model_params.name == "llama-2-70b"
        assert config.submission_ref.ruleset == "mlperf-inference-v6.0"

    @pytest.mark.unit
    def test_multiple_accuracy_datasets(self):
        config = BenchmarkConfig(
            type=TestType.SUBMISSION,
            benchmark_mode=TestType.OFFLINE,
            model_params={"name": "test"},
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            datasets=[
                {"name": "gpqa", "type": "accuracy", "path": "gpqa.jsonl"},
                {"name": "aime", "type": "accuracy", "path": "aime.jsonl"},
            ],
        )
        acc = [d for d in config.datasets if d.type == DatasetType.ACCURACY]
        assert len(acc) == 2

    @pytest.mark.unit
    def test_duplicate_datasets_rejected(self):
        with pytest.raises(ValueError, match="Duplicate dataset"):
            BenchmarkConfig(
                type=TestType.OFFLINE,
                model_params={"name": "test"},
                endpoint_config={"endpoints": ["http://localhost:8000"]},
                datasets=[{"path": "test.jsonl"}, {"path": "test.jsonl"}],
            )

    @pytest.mark.unit
    def test_explicit_streaming_preserved(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M", "streaming": "on"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
        )
        assert config.model_params.streaming == StreamingMode.ON

    @pytest.mark.unit
    def test_offline_rejects_poisson(self):
        with pytest.raises(ValueError, match="max_throughput"):
            BenchmarkConfig(
                type=TestType.OFFLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={"load_pattern": {"type": "poisson", "target_qps": 10}},
            )

    @pytest.mark.unit
    def test_online_max_throughput_rejected(self):
        with pytest.raises(ValueError, match="Online mode requires"):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={"load_pattern": {"type": "max_throughput"}},
            )

    @pytest.mark.unit
    def test_negative_min_duration_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            BenchmarkConfig(
                type=TestType.OFFLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={"runtime": {"min_duration_ms": -1}},
            )

    @pytest.mark.unit
    def test_max_lt_min_duration_rejected(self):
        with pytest.raises(ValueError, match="max_duration_ms"):
            BenchmarkConfig(
                type=TestType.OFFLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={
                    "runtime": {"min_duration_ms": 5000, "max_duration_ms": 1000}
                },
            )

    @pytest.mark.unit
    def test_max_duration_below_zero_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            BenchmarkConfig(
                type=TestType.OFFLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={"runtime": {"max_duration_ms": -1}},
            )

    @pytest.mark.unit
    def test_submission_bad_benchmark_mode(self):
        with pytest.raises(ValueError, match="benchmark_mode"):
            BenchmarkConfig(
                type=TestType.SUBMISSION,
                benchmark_mode=TestType.EVAL,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                submission_ref={"model": "M", "ruleset": "R"},
            )


class TestBenchmarkConfigMethods:
    @pytest.mark.unit
    def test_get_benchmark_mode_offline(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
        )
        assert config.get_benchmark_mode() == TestType.OFFLINE

    @pytest.mark.unit
    def test_get_benchmark_mode_submission(self):
        config = BenchmarkConfig(
            type=TestType.SUBMISSION,
            benchmark_mode=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
            submission_ref={"model": "M", "ruleset": "R"},
        )
        assert config.get_benchmark_mode() == TestType.OFFLINE

    @pytest.mark.unit
    def test_get_benchmark_mode_eval_returns_none(self):
        config = BenchmarkConfig(
            type=TestType.EVAL,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
        )
        assert config.get_benchmark_mode() is None

    @pytest.mark.unit
    def test_get_single_dataset(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[
                {"name": "acc", "type": "accuracy", "path": "a.jsonl"},
                {"name": "perf", "type": "performance", "path": "p.jsonl"},
            ],
        )
        ds = config.get_single_dataset()
        assert ds.path == "p.jsonl"

    @pytest.mark.unit
    def test_get_single_dataset_empty(self):
        config = BenchmarkConfig(
            type=TestType.EVAL,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
        )
        assert config.get_single_dataset() is None

    @pytest.mark.unit
    def test_get_single_dataset_acc_only(self):
        config = BenchmarkConfig(
            type=TestType.EVAL,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"name": "acc", "type": "accuracy", "path": "a.jsonl"}],
        )
        assert config.get_single_dataset().path == "a.jsonl"

    @pytest.mark.unit
    def test_create_default_offline(self):
        config = BenchmarkConfig.create_default_config(TestType.OFFLINE)
        assert config.type == TestType.OFFLINE
        assert config.model_params.name == "<MODEL_NAME>"

    @pytest.mark.unit
    def test_create_default_online(self):
        config = BenchmarkConfig.create_default_config(TestType.ONLINE)
        assert config.type == TestType.ONLINE
        assert config.settings.load_pattern.target_qps == 10.0

    @pytest.mark.unit
    def test_create_default_eval_not_implemented(self):
        with pytest.raises(CLIError, match="EVAL config not yet implemented"):
            BenchmarkConfig.create_default_config(TestType.EVAL)

    @pytest.mark.unit
    def test_create_default_submission_not_implemented(self):
        with pytest.raises(CLIError, match="SUBMISSION config not yet implemented"):
            BenchmarkConfig.create_default_config(TestType.SUBMISSION)

    @pytest.mark.unit
    def test_to_yaml_file(self, tmp_path):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
        )
        out = tmp_path / "out.yaml"
        config.to_yaml_file(out)
        assert out.exists()
        loaded = BenchmarkConfig.from_yaml_file(out)
        assert loaded.model_params.name == "M"

    @pytest.mark.unit
    def test_max_duration_zero_converts_to_none_in_runtime_settings(self):
        from inference_endpoint.config.runtime_settings import RuntimeSettings

        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D"}],
            settings={"runtime": {"max_duration_ms": 0}},
        )
        rt = RuntimeSettings.from_config(config, dataloader_num_samples=100)
        assert rt.max_duration_ms is None

    @pytest.mark.unit
    def test_from_yaml_file_not_found(self):
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            BenchmarkConfig.from_yaml_file(Path("/nonexistent.yaml"))


class TestClientAPITypePropagation:
    """endpoint_config.api_type must reach client.adapter/accumulator at construction.

    Regression coverage for the SGLang gpt-oss-120b path: prior code left the
    client with the OpenAI adapter unless execute.py patched it via
    ``with_updates(api_type=...)`` at runtime, which silently broke when the
    auto-resolved adapter was already populated.
    """

    @staticmethod
    def _common(api_type: APIType) -> dict:
        return {
            "type": TestType.OFFLINE,
            "model_params": {"name": "M"},
            "datasets": [{"path": "D"}],
            "endpoint_config": {
                "endpoints": ["http://x:8000"],
                "api_type": api_type,
            },
        }

    @pytest.mark.unit
    def test_sglang_endpoint_resolves_sglang_adapter(self):
        from inference_endpoint.sglang.accumulator import SGLangSSEAccumulator
        from inference_endpoint.sglang.adapter import SGLangGenerateAdapter

        config = BenchmarkConfig(**self._common(APIType.SGLANG))
        assert config.settings.client.api_type is APIType.SGLANG
        assert config.settings.client.adapter is SGLangGenerateAdapter
        assert config.settings.client.accumulator is SGLangSSEAccumulator

    @pytest.mark.unit
    def test_openai_endpoint_resolves_openai_adapter(self):
        from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
        from inference_endpoint.openai.openai_msgspec_adapter import (
            OpenAIMsgspecAdapter,
        )

        config = BenchmarkConfig(**self._common(APIType.OPENAI))
        assert config.settings.client.api_type is APIType.OPENAI
        assert config.settings.client.adapter is OpenAIMsgspecAdapter
        assert config.settings.client.accumulator is OpenAISSEAccumulator

    @pytest.mark.unit
    def test_with_updates_runtime_fields_preserves_sglang_adapter(self):
        """execute.py-style hand-off (no api_type kwarg) must not regress adapter."""
        from inference_endpoint.sglang.adapter import SGLangGenerateAdapter

        config = BenchmarkConfig(**self._common(APIType.SGLANG))
        updated = config.settings.client.with_updates(
            endpoint_urls=["http://x:8000/generate"],
            api_key=None,
        )
        assert updated.adapter is SGLangGenerateAdapter

    @pytest.mark.unit
    def test_with_updates_changing_api_type_reresolves_adapter(self):
        """Defensive: direct api_type swap on the client clears stale adapter."""
        from inference_endpoint.openai.openai_msgspec_adapter import (
            OpenAIMsgspecAdapter,
        )
        from inference_endpoint.sglang.adapter import SGLangGenerateAdapter

        config = BenchmarkConfig(**self._common(APIType.SGLANG))
        assert config.settings.client.adapter is SGLangGenerateAdapter

        rebound = config.settings.client.with_updates(api_type=APIType.OPENAI)
        assert rebound.api_type is APIType.OPENAI
        assert rebound.adapter is OpenAIMsgspecAdapter

    @pytest.mark.unit
    def test_explicit_adapter_override_at_construction_survives(self):
        """parse=False contract: callers can inject a custom adapter.

        Pick a concrete adapter that is *not* the auto-resolved default for
        ``api_type`` so we can detect a silent overwrite. ``OpenAIAdapter`` is
        a sibling of the default ``OpenAIMsgspecAdapter`` — both subclass
        ``HttpRequestAdapter`` so Pydantic accepts it.
        """
        from inference_endpoint.endpoint_client.config import HTTPClientConfig
        from inference_endpoint.openai.openai_adapter import OpenAIAdapter
        from inference_endpoint.openai.openai_msgspec_adapter import (
            OpenAIMsgspecAdapter,
        )

        client = HTTPClientConfig(api_type=APIType.OPENAI, adapter=OpenAIAdapter)
        assert client.adapter is OpenAIAdapter
        assert client.adapter is not OpenAIMsgspecAdapter


class TestMultiTurnValidation:
    """Tests for multi-turn config validation and cross-validation."""

    def _make_online_multi_turn(self, concurrency: int | None = 4, **ds_kwargs):
        lp: dict = {"type": "multi_turn"}
        if concurrency is not None:
            lp["target_concurrency"] = concurrency
        return {
            "type": TestType.ONLINE,
            "model_params": {"name": "M"},
            "endpoint_config": {"endpoints": ["http://x"]},
            "datasets": [{"path": "D", "multi_turn": {}, **ds_kwargs}],
            "settings": {"load_pattern": lp},
        }

    @pytest.mark.unit
    def test_multi_turn_valid_config(self):
        config = BenchmarkConfig(**self._make_online_multi_turn(concurrency=16))
        assert config.settings.load_pattern.type == LoadPatternType.MULTI_TURN
        assert config.settings.load_pattern.target_concurrency == 16

    @pytest.mark.unit
    def test_multi_turn_requires_target_concurrency(self):
        with pytest.raises(ValueError, match="Multi-turn requires --concurrency"):
            BenchmarkConfig(**self._make_online_multi_turn(concurrency=None))

    @pytest.mark.unit
    def test_multi_turn_without_multi_turn_dataset_rejected(self):
        with pytest.raises(ValueError, match="requires at least one dataset"):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={
                    "load_pattern": {"type": "multi_turn", "target_concurrency": 4}
                },
            )

    @pytest.mark.unit
    def test_multi_turn_dataset_without_multi_turn_load_pattern_rejected(self):
        with pytest.raises(ValueError, match="require load_pattern.type=multi_turn"):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D", "multi_turn": {}}],
                settings={"load_pattern": {"type": "poisson", "target_qps": 10}},
            )


class TestMultiTurnTotalSamples:
    """Tests for total_samples_to_issue() with multi_turn load pattern."""

    @pytest.mark.unit
    def test_multi_turn_uses_dataset_size_ignoring_duration(self):
        config = BenchmarkConfig(
            type=TestType.ONLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D", "multi_turn": {}}],
            settings={
                "load_pattern": {"type": "multi_turn", "target_concurrency": 4},
                "runtime": {"min_duration_ms": 600000},
            },
        )
        rt = RuntimeSettings.from_config(config, dataloader_num_samples=4316)
        assert rt.total_samples_to_issue() == 4316

    @pytest.mark.unit
    def test_multi_turn_respects_min_sample_count(self):
        lp = LoadPattern(type=LoadPatternType.MULTI_TURN, target_concurrency=4)
        rt = RuntimeSettings(
            metric_target=metrics.Throughput(10.0),
            reported_metrics=[metrics.Throughput(10.0)],
            min_duration_ms=600000,
            max_duration_ms=None,
            n_samples_from_dataset=5,
            n_samples_to_issue=None,
            min_sample_count=100,
            rng_sched=random.Random(0),
            rng_sample_index=random.Random(0),
            load_pattern=lp,
        )
        assert rt.total_samples_to_issue() == 100

    @pytest.mark.unit
    def test_multi_turn_explicit_n_samples_takes_precedence(self):
        lp = LoadPattern(type=LoadPatternType.MULTI_TURN, target_concurrency=4)
        rt = RuntimeSettings(
            metric_target=metrics.Throughput(10.0),
            reported_metrics=[metrics.Throughput(10.0)],
            min_duration_ms=600000,
            max_duration_ms=None,
            n_samples_from_dataset=4316,
            n_samples_to_issue=200,
            min_sample_count=1,
            rng_sched=random.Random(0),
            rng_sample_index=random.Random(0),
            load_pattern=lp,
        )
        assert rt.total_samples_to_issue() == 200
