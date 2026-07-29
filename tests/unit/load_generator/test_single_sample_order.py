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

import random

import pytest
from inference_endpoint.config.runtime_settings import RuntimeSettings, SampleOrderSpec
from inference_endpoint.load_generator.sample_order import (
    SingleSampleOrder,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
    create_sample_order,
)
from inference_endpoint.metrics.metric import Throughput


@pytest.mark.unit
def test_single_yields_fixed_index_forever():
    order = SingleSampleOrder(sample_index=3, n_samples_in_dataset=10)
    assert [next(order) for _ in range(5)] == [3, 3, 3, 3, 3]


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, 10, 99])
def test_single_rejects_out_of_range_index(bad: int):
    with pytest.raises(ValueError, match="sample_index"):
        SingleSampleOrder(sample_index=bad, n_samples_in_dataset=10)


def _settings(spec: SampleOrderSpec, n: int = 10) -> RuntimeSettings:
    return RuntimeSettings(
        metric_target=Throughput(1.0),
        reported_metrics=[],
        min_duration_ms=0,
        max_duration_ms=None,
        n_samples_from_dataset=n,
        n_samples_to_issue=None,
        min_sample_count=1,
        rng_sched=random.Random(0),
        rng_sample_index=random.Random(0),
        load_pattern=None,
        sample_order=spec,
    )


@pytest.mark.unit
def test_create_dispatches_single():
    order = create_sample_order(_settings(SampleOrderSpec.single(2)))
    assert isinstance(order, SingleSampleOrder)
    assert next(order) == 2


@pytest.mark.unit
def test_create_defaults_without_replacement():
    order = create_sample_order(_settings(SampleOrderSpec()))
    assert isinstance(order, WithoutReplacementSampleOrder)


@pytest.mark.unit
def test_create_dispatches_with_replacement():
    order = create_sample_order(_settings(SampleOrderSpec.with_replacement()))
    assert isinstance(order, WithReplacementSampleOrder)
