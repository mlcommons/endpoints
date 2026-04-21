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

"""Tests for sample_order.py."""

import random

import pytest
from inference_endpoint.load_generator.sample_order import (
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
)

# Exercise small/medium/large dataset sizes so shuffle-buffer behavior is
# covered for inputs both much smaller and much larger than typical batches.
_DATASET_SIZES = [3, 100, 10_000]


@pytest.mark.unit
class TestWithoutReplacementSampleOrder:
    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_yields_all_indices(self, n_samples: int):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        indices = [next(order) for _ in range(n_samples)]
        assert sorted(indices) == list(range(n_samples))

    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_reshuffles_after_exhaustion(self, n_samples: int):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        first_pass = [next(order) for _ in range(n_samples)]
        second_pass = [next(order) for _ in range(n_samples)]
        assert sorted(first_pass) == list(range(n_samples))
        assert sorted(second_pass) == list(range(n_samples))

    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_never_raises_stop_iteration(self, n_samples: int):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        # Should be able to draw far more than dataset size
        draws = max(100, n_samples * 3)
        indices = [next(order) for _ in range(draws)]
        assert len(indices) == draws
        assert all(0 <= i < n_samples for i in indices)

    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_reproducible_with_seed(self, n_samples: int):
        order1 = WithoutReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        order2 = WithoutReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        seq1 = [next(order1) for _ in range(n_samples * 2)]
        seq2 = [next(order2) for _ in range(n_samples * 2)]
        assert seq1 == seq2

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="n_samples_in_dataset must be > 0"):
            WithoutReplacementSampleOrder(n_samples_in_dataset=0)


@pytest.mark.unit
class TestWithReplacementSampleOrder:
    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_yields_valid_indices(self, n_samples: int):
        order = WithReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        indices = [next(order) for _ in range(max(100, n_samples))]
        assert all(0 <= i < n_samples for i in indices)

    @pytest.mark.parametrize("n_samples", _DATASET_SIZES)
    def test_reproducible_with_seed(self, n_samples: int):
        order1 = WithReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        order2 = WithReplacementSampleOrder(
            n_samples_in_dataset=n_samples, rng=random.Random(42)
        )
        seq1 = [next(order1) for _ in range(n_samples * 2)]
        seq2 = [next(order2) for _ in range(n_samples * 2)]
        assert seq1 == seq2
