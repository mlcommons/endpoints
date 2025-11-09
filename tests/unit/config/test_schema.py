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

"""Tests for configuration schema."""

from inference_endpoint.config.schema import (
    BenchmarkConfig,
    Dataset,
    DatasetType,
    EvalMethod,
    ModelParams,
    OSLDistribution,
    OSLDistributionType,
    SubmissionReference,
)
from inference_endpoint.config.schema import TestType as BenchmarkTestType


class TestOSLDistribution:
    """Test OSL distribution configuration."""

    def test_fixed_distribution(self):
        """Test fixed OSL distribution."""
        osl = OSLDistribution(type=OSLDistributionType.FIXED, max=1024)
        assert osl.type == OSLDistributionType.FIXED
        assert osl.max == 1024

    def test_normal_distribution(self):
        """Test normal OSL distribution with mean and std."""
        osl = OSLDistribution(
            type=OSLDistributionType.NORMAL, mean=1000, std=200, min=512, max=2048
        )
        assert osl.type == OSLDistributionType.NORMAL
        assert osl.mean == 1000
        assert osl.std == 200


class TestModelParams:
    """Test model parameters."""

    def test_default_params(self):
        """Test default model parameters."""
        params = ModelParams()
        assert params.temperature == 0.7
        assert params.max_new_tokens == 1024

    def test_with_osl_distribution(self):
        """Test model params with OSL distribution."""
        params = ModelParams(
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


class TestDataset:
    """Test dataset configuration."""

    def test_performance_dataset(self):
        """Test performance dataset config."""
        dataset = Dataset(
            name="perf-test",
            type=DatasetType.PERFORMANCE,
            path="datasets/openorca.pkl",
            samples=5000,
        )
        assert dataset.type == DatasetType.PERFORMANCE
        assert dataset.eval_method is None

    def test_accuracy_dataset(self):
        """Test accuracy dataset config."""
        dataset = Dataset(
            name="gpqa",
            type=DatasetType.ACCURACY,
            path="datasets/gpqa.pkl",
            samples=500,
            eval_method=EvalMethod.EXACT_MATCH,
        )
        assert dataset.type == DatasetType.ACCURACY
        assert dataset.eval_method == EvalMethod.EXACT_MATCH


class TestBenchmarkConfig:
    """Test complete benchmark configuration."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = BenchmarkConfig(
            name="test",
            type=BenchmarkTestType.OFFLINE,
            datasets=[{"name": "test", "type": "performance", "path": "test.pkl"}],
        )
        assert config.name == "test"
        assert config.type == BenchmarkTestType.OFFLINE
        assert len(config.datasets) == 1

    def test_submission_config(self):
        """Test official submission configuration."""
        config = BenchmarkConfig(
            name="submission",
            version="1.0",
            type=BenchmarkTestType.SUBMISSION,
            submission_ref=SubmissionReference(
                model="llama-2-70b", ruleset="mlperf-inference-v6.0"
            ),
            datasets=[
                {
                    "name": "perf",
                    "type": "performance",
                    "path": "perf.pkl",
                    "samples": 5000,
                },
                {
                    "name": "gpqa",
                    "type": "accuracy",
                    "path": "gpqa.pkl",
                    "samples": 500,
                    "eval_method": "exact_match",
                },
            ],
        )
        assert config.submission_ref is not None
        assert config.submission_ref.model == "llama-2-70b"
        assert config.submission_ref.ruleset == "mlperf-inference-v6.0"
        assert len(config.datasets) == 2

    def test_multiple_accuracy_datasets(self):
        """Test config with multiple accuracy datasets."""
        config = BenchmarkConfig(
            name="multi-acc",
            type=BenchmarkTestType.SUBMISSION,
            datasets=[
                {
                    "name": "gpqa",
                    "type": "accuracy",
                    "path": "gpqa.pkl",
                    "eval_method": "exact_match",
                },
                {
                    "name": "aime",
                    "type": "accuracy",
                    "path": "aime.pkl",
                    "eval_method": "exact_match",
                },
            ],
        )
        accuracy_datasets = [
            d for d in config.datasets if d.type == DatasetType.ACCURACY
        ]
        assert len(accuracy_datasets) == 2
        assert {d.name for d in accuracy_datasets} == {"gpqa", "aime"}
