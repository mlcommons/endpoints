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


@pytest.mark.unit
class TestWithoutReplacementSampleOrder:
    def test_yields_all_indices(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=5, rng=random.Random(42)
        )
        indices = [next(order) for _ in range(5)]
        assert sorted(indices) == [0, 1, 2, 3, 4]

    def test_reshuffles_after_exhaustion(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=3, rng=random.Random(42)
        )
        first_pass = [next(order) for _ in range(3)]
        second_pass = [next(order) for _ in range(3)]
        assert sorted(first_pass) == [0, 1, 2]
        assert sorted(second_pass) == [0, 1, 2]

    def test_never_raises_stop_iteration(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=2, rng=random.Random(42)
        )
        # Should be able to draw far more than dataset size
        indices = [next(order) for _ in range(100)]
        assert len(indices) == 100
        assert all(0 <= i < 2 for i in indices)

    def test_reproducible_with_seed(self):
        order1 = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        order2 = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        seq1 = [next(order1) for _ in range(20)]
        seq2 = [next(order2) for _ in range(20)]
        assert seq1 == seq2

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="n_samples_in_dataset must be > 0"):
            WithoutReplacementSampleOrder(n_samples_in_dataset=0)


@pytest.mark.unit
class TestWithReplacementSampleOrder:
    def test_yields_valid_indices(self):
        order = WithReplacementSampleOrder(
            n_samples_in_dataset=5, rng=random.Random(42)
        )
        indices = [next(order) for _ in range(100)]
        assert all(0 <= i < 5 for i in indices)

    def test_reproducible_with_seed(self):
        order1 = WithReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        order2 = WithReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        seq1 = [next(order1) for _ in range(20)]
        seq2 = [next(order2) for _ in range(20)]
        assert seq1 == seq2
