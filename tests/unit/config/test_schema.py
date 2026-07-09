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
import re

import pytest
from inference_endpoint import metrics
from inference_endpoint.config import ruleset_registry
from inference_endpoint.config.ruleset_registry import register_ruleset
from inference_endpoint.config.rulesets.mlcommons.rules import RoundRuleset
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
    ProfilerEngine,
    ProfilingConfig,
    StreamingMode,
    SubmissionReference,
    TestType,
)
from inference_endpoint.exceptions import CLIError
from pydantic import ValidationError


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
        assert params.tokenizer_name is None

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

    @pytest.mark.unit
    def test_tokenizer_name_override(self):
        params = ModelParams(
            name="qwen/qwen3.6-35b-a3b", tokenizer_name="Qwen/Qwen3.6-35B-A3B"
        )
        assert params.tokenizer_name == "Qwen/Qwen3.6-35B-A3B"
        assert params.name == "qwen/qwen3.6-35b-a3b"

    @pytest.mark.unit
    def test_min_new_tokens_cannot_exceed_max_new_tokens(self):
        with pytest.raises(
            ValidationError, match="min_new_tokens must be less than or equal"
        ):
            ModelParams(name="test", min_new_tokens=2, max_new_tokens=1)

    @pytest.mark.unit
    def test_min_new_tokens_defaults_one(self):
        assert ModelParams(name="test").min_new_tokens == 1

    @pytest.mark.unit
    def test_skip_special_tokens_defaults_true(self):
        assert ModelParams(name="test").skip_special_tokens is True


class TestAPIType:
    @pytest.mark.unit
    def test_default_routes(self):
        assert APIType.OPENAI.default_route() == "v1/chat/completions"
        assert APIType.SGLANG.default_route() == "generate"
        assert APIType.OPENAI_COMPLETIONS.default_route() == "v1/completions"


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

    @pytest.mark.unit
    def test_generation_config_override_accepts_known_keys(self):
        ds = Dataset(
            name="acc",
            type=DatasetType.ACCURACY,
            path="acc.jsonl",
            generation_config_override={"max_new_tokens": 32768, "temperature": 0.0},
        )
        assert ds.generation_config_override == {
            "max_new_tokens": 32768,
            "temperature": 0.0,
        }

    @pytest.mark.unit
    def test_generation_config_override_rejects_unknown_key(self):
        with pytest.raises(
            ValueError, match=r"unknown keys in generation_config_override.*bogus"
        ):
            Dataset(
                name="acc",
                path="a.jsonl",
                generation_config_override={"bogus": 1},
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("key", ["name", "streaming", "tokenizer_name"])
    def test_generation_config_override_rejects_metrics_decoupled_key(self, key):
        """Per-run/identity keys drive the single global tokenizer / aggregator,
        so a per-dataset override is rejected at construction rather than
        silently desyncing metrics accounting.
        """
        with pytest.raises(
            ValueError,
            match=r"not honored per-dataset",
        ):
            Dataset(
                name="acc",
                path="a.jsonl",
                generation_config_override={key: "whatever"},
            )

    @pytest.mark.unit
    def test_generation_config_override_none_is_noop(self):
        base = ModelParams(name="m", max_new_tokens=1024, streaming=StreamingMode.ON)
        ds = Dataset(name="x", path="x.jsonl")
        assert ds.effective_generation_config(base) is base

    @pytest.mark.unit
    def test_effective_generation_config_merges_sparse_dict(self):
        base = ModelParams(name="m", temperature=0.5, top_p=0.9, max_new_tokens=1024)
        ds = Dataset(
            name="x",
            path="x.jsonl",
            generation_config_override={"max_new_tokens": 32768},
        )
        merged = ds.effective_generation_config(base)
        # overridden field changes...
        assert merged.max_new_tokens == 32768
        # ...everything else is preserved from base
        assert merged.name == "m"
        assert merged.temperature == 0.5
        assert merged.top_p == 0.9

    @pytest.mark.unit
    def test_effective_generation_config_validates_value(self):
        """ModelParams.model_validate is invoked on the merged dict, so a
        type-invalid override is rejected (e.g. non-numeric temperature)."""
        base = ModelParams(name="m")
        ds = Dataset(
            name="x",
            path="x.jsonl",
            generation_config_override={"temperature": "hot"},
        )
        with pytest.raises(ValueError):
            ds.effective_generation_config(base)

    @pytest.mark.unit
    def test_effective_generation_config_deep_merges_nested_dict(self):
        """Sparse overrides of nested fields (osl_distribution,
        chat_template_kwargs) preserve sibling defaults from the base rather
        than wholesale-replacing the nested object. Pins the deep-merge
        behavior added in response to PR review feedback.
        """
        base = ModelParams(
            name="m",
            osl_distribution=OSLDistribution(
                type=OSLDistributionType.NORMAL, mean=1000, std=200, min=512, max=2048
            ),
        )
        ds = Dataset(
            name="x",
            path="x.jsonl",
            generation_config_override={"osl_distribution": {"max": 512}},
        )
        merged = ds.effective_generation_config(base)
        # the explicitly overridden nested field changes...
        assert merged.osl_distribution.max == 512
        # ...and the unspecified siblings are preserved from base
        assert merged.osl_distribution.type == OSLDistributionType.NORMAL
        assert merged.osl_distribution.mean == 1000
        assert merged.osl_distribution.std == 200
        assert merged.osl_distribution.min == 512

    @pytest.mark.unit
    def test_effective_generation_config_deep_merges_chat_template_kwargs(self):
        """Deep-merge also applies to free-form nested dicts like
        chat_template_kwargs; sparse overrides preserve sibling entries.
        """
        base = ModelParams(
            name="m", chat_template_kwargs={"enable_thinking": True, "tools": []}
        )
        ds = Dataset(
            name="x",
            path="x.jsonl",
            generation_config_override={
                "chat_template_kwargs": {"enable_thinking": False}
            },
        )
        merged = ds.effective_generation_config(base)
        assert merged.chat_template_kwargs == {
            "enable_thinking": False,
            "tools": [],
        }


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
                model="llama-2-70b", ruleset="mlperf-inference-v6.1"
            ),
            datasets=[{"path": "perf.jsonl"}],
        )
        assert config.model_params.name == "llama-2-70b"
        assert config.submission_ref.ruleset == "mlperf-inference-v6.1"

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
            submission_ref={"model": "M", "ruleset": "mlperf-inference-v6.1"},
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

    @pytest.mark.unit
    def test_openai_completions_endpoint_resolves_adapter(self):
        from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
        from inference_endpoint.openai.completions_adapter import (
            OpenAITextCompletionsAdapter,
        )

        values = self._common(APIType.OPENAI_COMPLETIONS)
        values["model_params"].update(
            min_new_tokens=1,
            skip_special_tokens=False,
        )
        config = BenchmarkConfig(**values)
        assert config.settings.client.api_type is APIType.OPENAI_COMPLETIONS
        assert config.settings.client.adapter is OpenAITextCompletionsAdapter
        assert config.settings.client.accumulator is OpenAISSEAccumulator

    @pytest.mark.unit
    @pytest.mark.parametrize("api_type", [APIType.OPENAI, APIType.SGLANG])
    @pytest.mark.parametrize(
        ("controls", "message"),
        [
            (
                {"min_new_tokens": 0},
                "model_params.min_new_tokens requires "
                "endpoint_config.api_type=openai_completions",
            ),
            (
                {"skip_special_tokens": False},
                "model_params.skip_special_tokens requires "
                "endpoint_config.api_type=openai_completions",
            ),
            (
                {"min_new_tokens": 0, "skip_special_tokens": False},
                "model_params.min_new_tokens and model_params.skip_special_tokens require "
                "endpoint_config.api_type=openai_completions",
            ),
        ],
    )
    def test_completion_generation_controls_reject_other_api_types(
        self, api_type, controls, message
    ):
        values = self._common(api_type)
        values["model_params"].update(controls)
        with pytest.raises(ValidationError, match=re.escape(message)):
            BenchmarkConfig(**values)

    @pytest.mark.unit
    @pytest.mark.parametrize("api_type", [APIType.OPENAI, APIType.SGLANG])
    def test_default_completion_generation_controls_allow_other_api_types(
        self, api_type
    ):
        config = BenchmarkConfig(**self._common(api_type))
        assert config.model_params.min_new_tokens == 1
        assert config.model_params.skip_special_tokens is True


class TestAgenticInferenceValidation:
    """Tests for agentic inference config validation and cross-validation."""

    def _make_online_agentic_inference(self, concurrency: int | None = 4, **ds_kwargs):
        lp: dict = {"type": "agentic_inference"}
        if concurrency is not None:
            lp["target_concurrency"] = concurrency
        return {
            "type": TestType.ONLINE,
            "model_params": {"name": "M"},
            "endpoint_config": {"endpoints": ["http://x"]},
            "datasets": [{"path": "D", "agentic_inference": {}, **ds_kwargs}],
            "settings": {"load_pattern": lp},
        }

    @pytest.mark.unit
    def test_agentic_inference_valid_config(self):
        config = BenchmarkConfig(**self._make_online_agentic_inference(concurrency=16))
        assert config.settings.load_pattern.type == LoadPatternType.AGENTIC_INFERENCE
        assert config.settings.load_pattern.target_concurrency == 16

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "controls", [{"min_new_tokens": 0}, {"skip_special_tokens": False}]
    )
    def test_agentic_inference_rejects_text_completion_generation_controls(
        self, controls
    ):
        values = self._make_online_agentic_inference()
        values["model_params"].update(controls)
        values["endpoint_config"]["api_type"] = APIType.OPENAI_COMPLETIONS
        with pytest.raises(
            ValidationError, match="not supported for agentic inference"
        ):
            BenchmarkConfig(**values)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "override", [{"min_new_tokens": 0}, {"skip_special_tokens": False}]
    )
    def test_agentic_inference_rejects_completion_controls_via_override(self, override):
        """Completion controls reaching an agentic dataset through a per-dataset
        generation_config_override are rejected, exactly as when set on top-level
        model_params."""
        values = self._make_online_agentic_inference(
            generation_config_override=override
        )
        values["endpoint_config"]["api_type"] = APIType.OPENAI_COMPLETIONS
        with pytest.raises(
            ValidationError, match="not supported for agentic inference"
        ):
            BenchmarkConfig(**values)

    @pytest.mark.unit
    def test_agentic_inference_rejects_removed_stop_on_first_empty_slot_as_extra(self):
        # Legacy agentic inference knobs should remain rejected by extra="forbid".
        with pytest.raises(ValueError, match="stop_on_first_empty_slot"):
            BenchmarkConfig(
                **self._make_online_agentic_inference(
                    concurrency=16,
                    agentic_inference={"stop_on_first_empty_slot": True},
                )
            )

    @pytest.mark.unit
    def test_agentic_inference_requires_target_concurrency(self):
        with pytest.raises(
            ValueError, match="Agentic inference requires --concurrency"
        ):
            BenchmarkConfig(**self._make_online_agentic_inference(concurrency=None))

    @pytest.mark.unit
    def test_agentic_inference_without_agentic_inference_dataset_rejected(self):
        with pytest.raises(
            ValueError,
            match="requires the performance dataset to have agentic_inference config",
        ):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D"}],
                settings={
                    "load_pattern": {
                        "type": "agentic_inference",
                        "target_concurrency": 4,
                    }
                },
            )

    @pytest.mark.unit
    def test_agentic_inference_dataset_without_agentic_inference_load_pattern_rejected(
        self,
    ):
        with pytest.raises(
            ValueError, match="requires load_pattern.type=agentic_inference"
        ):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D", "agentic_inference": {}}],
                settings={"load_pattern": {"type": "poisson", "target_qps": 10}},
            )

    @pytest.mark.unit
    def test_agentic_inference_rejects_runtime_num_samples_override(self):
        with pytest.raises(ValueError, match="num_trajectories_to_issue"):
            BenchmarkConfig(
                type=TestType.ONLINE,
                model_params={"name": "M"},
                endpoint_config={"endpoints": ["http://x"]},
                datasets=[{"path": "D", "agentic_inference": {}}],
                settings={
                    "load_pattern": {
                        "type": "agentic_inference",
                        "target_concurrency": 4,
                    },
                    "runtime": {"n_samples_to_issue": 200},
                },
            )


class TestAgenticInferenceTotalSamples:
    """Tests for total_samples_to_issue() with agentic_inference load pattern."""

    @pytest.mark.unit
    def test_agentic_inference_uses_dataset_size_ignoring_duration(self):
        config = BenchmarkConfig(
            type=TestType.ONLINE,
            model_params={"name": "M"},
            endpoint_config={"endpoints": ["http://x"]},
            datasets=[{"path": "D", "agentic_inference": {}}],
            settings={
                "load_pattern": {"type": "agentic_inference", "target_concurrency": 4},
                "runtime": {"min_duration_ms": 600000},
            },
        )
        rt = RuntimeSettings.from_config(config, dataloader_num_samples=4316)
        assert rt.total_samples_to_issue() == 4316

    @pytest.mark.unit
    def test_agentic_inference_clamps_to_dataset_size(self):
        lp = LoadPattern(type=LoadPatternType.AGENTIC_INFERENCE, target_concurrency=4)
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
        assert rt.total_samples_to_issue() == 5

    @pytest.mark.unit
    def test_agentic_inference_explicit_n_samples_takes_precedence(self):
        lp = LoadPattern(type=LoadPatternType.AGENTIC_INFERENCE, target_concurrency=4)
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


class TestProfilingConfig:
    @pytest.mark.unit
    def test_defaults(self):
        cfg = ProfilingConfig()
        assert cfg.engine is None
        assert cfg.urls is None

    @pytest.mark.unit
    def test_engine_enum_coercion(self):
        assert ProfilingConfig(engine="vllm").engine is ProfilerEngine.VLLM

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "ctor",
        [
            lambda u: ProfilingConfig(engine="vllm", urls=u),
            lambda u: ProfilingConfig.model_validate({"engine": "vllm", "urls": u}),
        ],
    )
    def test_url_scheme_rejected_without_scheme(self, ctor):
        with pytest.raises(ValidationError):
            ctor(["localhost:8000"])

    @pytest.mark.unit
    def test_valid_urls_accepted(self):
        cfg = ProfilingConfig(engine="vllm", urls=["http://h:8001/v1"])
        assert cfg.urls == ["http://h:8001/v1"]


# Official v6.1 seeds from loadgen/mlperf.conf L41-43 (schedule / sample_index).
_V6_1_SCHED_SEED = 16159082839903944936
_V6_1_SAMPLE_SEED = 2747215439041700203


class TestRulesetSeedOverride:
    """A submission ruleset pins the runtime RNG seeds at config construction,
    before the config is dumped to the report dir."""

    def _submission(self, ruleset: str, runtime: dict | None = None) -> BenchmarkConfig:
        settings = {"runtime": runtime} if runtime is not None else {}
        return BenchmarkConfig(
            type=TestType.SUBMISSION,
            benchmark_mode=TestType.OFFLINE,
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            submission_ref=SubmissionReference(model="llama-2-70b", ruleset=ruleset),
            datasets=[{"path": "perf.jsonl"}],
            settings=settings,
        )

    @pytest.mark.unit
    def test_registered_ruleset_pins_seeds(self):
        cfg = self._submission("mlperf-inference-v6.1")
        assert cfg.settings.runtime.scheduler_random_seed == _V6_1_SCHED_SEED
        assert cfg.settings.runtime.dataloader_random_seed == _V6_1_SAMPLE_SEED
        # Warmup is reseeded from the sample-index (dataloader) seed so its
        # sample order derives from the same pinned seed as the perf phase.
        assert cfg.settings.warmup.warmup_random_seed == _V6_1_SAMPLE_SEED

    @pytest.mark.unit
    def test_override_precedes_config_dump(self, tmp_path):
        """The dumped config.yaml carries the pinned seeds, so the report dir's
        config is representative of what actually ran."""
        cfg = self._submission("mlperf-inference-v6.1")
        out = tmp_path / "config.yaml"
        cfg.to_yaml_file(out)
        reloaded = BenchmarkConfig.from_yaml_file(out)
        assert reloaded.settings.runtime.scheduler_random_seed == _V6_1_SCHED_SEED
        assert reloaded.settings.runtime.dataloader_random_seed == _V6_1_SAMPLE_SEED

    @pytest.mark.unit
    def test_ruleset_seed_wins_over_user_value(self):
        """Compliance rounds lock the seeds — a user-supplied seed is overridden
        (mirrors LoadGen locking core seeds from user.conf)."""
        cfg = self._submission(
            "mlperf-inference-v6.1",
            runtime={"scheduler_random_seed": 7, "dataloader_random_seed": 9},
        )
        assert cfg.settings.runtime.scheduler_random_seed == _V6_1_SCHED_SEED
        assert cfg.settings.runtime.dataloader_random_seed == _V6_1_SAMPLE_SEED

    @pytest.mark.unit
    def test_unregistered_ruleset_submission_raises(self):
        """A submission naming an unregistered ruleset must fail loudly rather
        than silently falling back to default seeds."""
        with pytest.raises(ValidationError):
            self._submission("does-not-exist")

    @pytest.mark.unit
    def test_unregistered_ruleset_non_submission_is_lenient(self):
        """Non-submission configs are unaffected: an unknown ruleset leaves the
        runtime seeds at their defaults rather than erroring."""
        cfg = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "test"},
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            datasets=[{"path": "test.jsonl"}],
            submission_ref=SubmissionReference(model="test", ruleset="does-not-exist"),
        )
        assert cfg.settings.runtime.scheduler_random_seed == 42
        assert cfg.settings.runtime.dataloader_random_seed == 42

    @pytest.mark.unit
    def test_no_submission_ref_keeps_defaults(self):
        cfg = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "test"},
            endpoint_config={"endpoints": ["http://localhost:8000"]},
            datasets=[{"path": "test.jsonl"}],
        )
        assert cfg.settings.runtime.scheduler_random_seed == 42
        assert cfg.settings.runtime.dataloader_random_seed == 42

    @pytest.fixture
    def register_temp_ruleset(self):
        """Register a throwaway RoundRuleset for the test, then unregister it."""
        registered: list[str] = []

        def _register(name, *, scheduler_rng_seed, sample_index_rng_seed):
            register_ruleset(
                name,
                RoundRuleset(
                    version=name,
                    scheduler_rng_seed=scheduler_rng_seed,
                    sample_index_rng_seed=sample_index_rng_seed,
                    benchmark_rulesets={},
                ),
            )
            registered.append(name)
            return name

        yield _register
        for name in registered:
            ruleset_registry._RULESET_REGISTRY.pop(name, None)

    @pytest.mark.unit
    def test_partial_none_seed_sample_index_rejected(self, register_temp_ruleset):
        """A submission ruleset must pin both seeds. ``None`` (unseeded) is a
        valid value in the general ruleset contract but incoherent for a pinned
        submission, so a partially-unseeded ruleset is rejected loudly."""
        name = register_temp_ruleset(
            "test-partial-seed", scheduler_rng_seed=None, sample_index_rng_seed=999
        )
        with pytest.raises(ValidationError, match="leaves an RNG seed unset"):
            self._submission(name)

    @pytest.mark.unit
    def test_partial_none_seed_scheduler_rejected(self, register_temp_ruleset):
        """Mirror of the scheduler-None case: scheduler set, sample_index None."""
        name = register_temp_ruleset(
            "test-partial-sched", scheduler_rng_seed=777, sample_index_rng_seed=None
        )
        with pytest.raises(ValidationError, match="leaves an RNG seed unset"):
            self._submission(name)

    @pytest.mark.unit
    def test_all_none_seed_rejected(self, register_temp_ruleset):
        """A fully unseeded ruleset (both seeds ``None``) is rejected."""
        name = register_temp_ruleset(
            "test-no-seed", scheduler_rng_seed=None, sample_index_rng_seed=None
        )
        with pytest.raises(ValidationError, match="leaves an RNG seed unset"):
            self._submission(name)

    @pytest.mark.unit
    def test_non_int_seed_rejected(self, register_temp_ruleset):
        """Seed *values* are validated: a wrong-typed ruleset seed fails config
        construction instead of silently landing a ``str`` on the runtime.
        Guards against the model_copy(update=) __dict__ bypass."""
        name = register_temp_ruleset(
            "test-bad-seed", scheduler_rng_seed="oops", sample_index_rng_seed=999
        )
        with pytest.raises(ValidationError):
            self._submission(name)
