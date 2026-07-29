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

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.ruleset_registry import (
    get_ruleset,
    list_rulesets,
    register_ruleset,
)
from inference_endpoint.config.rulesets.mlcommons import datasets, models
from inference_endpoint.config.rulesets.mlcommons.rules import (
    ALL_ROUNDS,
    CURRENT,
    EDGE_CURRENT,
    OptimizationPriority,
)
from inference_endpoint.config.user_config import UserConfig


@pytest.mark.unit
def test_apply_user_config():
    user_config = UserConfig(1234.5, max_duration_ms=42 * 60 * 1000)
    rt_settings = CURRENT.apply_user_config(
        model=models.Llama3_1_8b,
        user_config=user_config,
        opt_prio=OptimizationPriority.LOW_LATENCY_INTERACTIVE,
    )

    assert rt_settings.model is models.Llama3_1_8b
    assert (
        rt_settings.optimization_priority
        is OptimizationPriority.LOW_LATENCY_INTERACTIVE
    )

    assert isinstance(rt_settings.metric_target, metrics.Throughput)
    assert rt_settings.metric_target.target == 1234.5

    assert len(rt_settings.reported_metrics) == 3
    assert isinstance(rt_settings.reported_metrics[0], metrics.Throughput)
    assert rt_settings.reported_metrics[0].target == 1234.5
    assert isinstance(rt_settings.reported_metrics[1], metrics.TTFT)
    assert rt_settings.reported_metrics[1].target == 500
    assert isinstance(rt_settings.reported_metrics[2], metrics.TPOT)
    assert rt_settings.reported_metrics[2].target == 30

    assert rt_settings.min_duration_ms == 10 * 60 * 1000
    assert rt_settings.max_duration_ms == 42 * 60 * 1000
    assert rt_settings.n_samples_from_dataset == 13368
    assert rt_settings.n_samples_to_issue is None
    assert rt_settings.min_sample_count == 270336
    assert (
        rt_settings.rules
        is CURRENT.benchmark_rulesets[models.Llama3_1_8b][
            OptimizationPriority.LOW_LATENCY_INTERACTIVE
        ]
    )

    # Metric type should be throughput
    expected_sample_count = int(1234.5 * 10 * 60)
    assert (
        rt_settings.total_samples_to_issue(
            padding_factor=1.0, align_to_dataset_size=False
        )
        == expected_sample_count
    )

    if (rem := expected_sample_count % rt_settings.n_samples_from_dataset) != 0:
        expected_sample_count += rt_settings.n_samples_from_dataset - rem
    assert (
        rt_settings.total_samples_to_issue(padding_factor=1.0) == expected_sample_count
    )


@pytest.mark.unit
def test_apply_user_config_insufficient_qps():
    user_config = UserConfig(2, max_duration_ms=42 * 60 * 1000)
    rt_settings = CURRENT.apply_user_config(
        model=models.Llama3_1_8b,
        user_config=user_config,
        opt_prio=OptimizationPriority.LOW_LATENCY_INTERACTIVE,
    )

    # Expected is 270336 padded up to multiple of dataset size, which is 13368
    assert rt_settings.total_samples_to_issue(padding_factor=1.0) == 280728
    assert (
        rt_settings.total_samples_to_issue(
            padding_factor=1.0, align_to_dataset_size=False
        )
        == 270336
    )


@pytest.mark.unit
def test_apply_user_config_min_sample_count_override():
    user_config = UserConfig(2, max_duration_ms=42 * 60 * 1000, min_sample_count=1)
    rt_settings = CURRENT.apply_user_config(
        model=models.Llama3_1_8b,
        user_config=user_config,
        opt_prio=OptimizationPriority.LOW_LATENCY_INTERACTIVE,
    )
    assert rt_settings.total_samples_to_issue(padding_factor=1.0) == 13368
    assert (
        rt_settings.total_samples_to_issue(
            padding_factor=1.0, align_to_dataset_size=False
        )
        == 2 * 10 * 60
    )


@pytest.mark.unit
def test_current_round_is_v6_1():
    assert CURRENT.version == "v6.1"


@pytest.mark.unit
def test_v6_1_official_seeds():
    """Seeds are the schedule/sample_index values from loadgen/mlperf.conf."""
    v6_1 = get_ruleset("mlperf-inference-v6.1")
    assert v6_1.scheduler_rng_seed == 16159082839903944936
    assert v6_1.sample_index_rng_seed == 2747215439041700203


@pytest.mark.unit
def test_all_rounds_registered_by_version():
    names = list_rulesets()
    for ruleset in ALL_ROUNDS:
        assert f"mlperf-inference-{ruleset.version}" in names
    # Both the prior and current round remain resolvable.
    assert "mlperf-inference-v5.1" in names
    assert "mlperf-inference-v6.1" in names
    assert get_ruleset("mlcommons-current") is CURRENT


@pytest.mark.unit
def test_round_versions_are_unique():
    """Each registered round must own a distinct version string, else the
    registry would silently overwrite one round's entry with another's."""
    assert len(ALL_ROUNDS) == len({r.version for r in ALL_ROUNDS})


@pytest.mark.unit
def test_register_ruleset_rejects_duplicate():
    """Re-registering an existing name raises rather than clobbering it."""
    with pytest.raises(ValueError):
        register_ruleset("mlperf-inference-v6.1", CURRENT)


@pytest.mark.unit
def test_v6_1_latency_targets_match_v5_1():
    """Only the round seeds rotate; per-model targets are identical."""
    v5_1 = get_ruleset("mlperf-inference-v5.1")
    v6_1 = get_ruleset("mlperf-inference-v6.1")
    assert v6_1.benchmark_rulesets == v5_1.benchmark_rulesets
    # v6.1 holds an independent copy, not a shared reference to v5.1's dict.
    assert v6_1.benchmark_rulesets is not v5_1.benchmark_rulesets
    # Anchor on a concrete target so the equality above verifies real structure.
    assert (
        v6_1.benchmark_rulesets[models.Llama3_1_405b][
            OptimizationPriority.THROUGHPUT
        ].max_tpot_latency_ms
        == 175
    )
    assert v6_1.scheduler_rng_seed != v5_1.scheduler_rng_seed
    assert v6_1.sample_index_rng_seed != v5_1.sample_index_rng_seed


def test_edge_ruleset_registered():
    # Resolvable by version-specific name and the "current" alias.
    assert get_ruleset("mlperf-edge-v0.1") is EDGE_CURRENT
    assert get_ruleset("mlperf-edge-current") is EDGE_CURRENT
    assert "mlperf-edge-v0.1" in list_rulesets()
    assert EDGE_CURRENT.version == "edge-v0.1"


@pytest.mark.unit
def test_edge_model_accuracy_gate():
    model = models.Qwen3_6_27B
    assert model.dataset is datasets.BFCLv4SingleTurn
    assert model.dataset.size == 995

    precision, golden = model.golden_accuracy
    assert precision == "q4_k_m-reference"
    assert golden["bfcl_overall_accuracy"] == pytest.approx(86.23)
    assert golden["bfcl_normalized_accuracy"] == pytest.approx(87.96)

    # 3% one-sided band: pass if score >= 0.97 x reference -> overall gate ~83.64%.
    (settings,) = model.accuracy_target_settings
    (overall_factor,) = settings["bfcl_overall_accuracy"]
    assert overall_factor == 0.97
    assert golden["bfcl_overall_accuracy"] * overall_factor == pytest.approx(83.6431)


@pytest.mark.unit
def test_edge_ruleset_model_lookup():
    model = models.Qwen3_6_27B
    assert model in EDGE_CURRENT.benchmark_rulesets
    rules = EDGE_CURRENT.benchmark_rulesets[model][
        OptimizationPriority.EDGE_SINGLE_STREAM
    ]
    assert rules.metric is metrics.Throughput
    assert rules.min_sample_count_valid == 995
    assert rules.max_duration_ms_valid == 4 * 60 * 60 * 1000


@pytest.mark.unit
def test_edge_ruleset_apply_user_config():
    # Single-stream edge perf: aggregate throughput target (tokens/s) supplied by
    # the user; min duration 0, 4 h safety cap.
    user_config = UserConfig(11.8)
    rt_settings = EDGE_CURRENT.apply_user_config(
        model=models.Qwen3_6_27B,
        user_config=user_config,
        opt_prio=OptimizationPriority.EDGE_SINGLE_STREAM,
    )
    assert rt_settings.model is models.Qwen3_6_27B
    assert isinstance(rt_settings.metric_target, metrics.Throughput)
    assert rt_settings.metric_target.target == pytest.approx(11.8)
    assert rt_settings.min_duration_ms == 0
    assert rt_settings.max_duration_ms == 4 * 60 * 60 * 1000
    assert rt_settings.n_samples_from_dataset == 995
