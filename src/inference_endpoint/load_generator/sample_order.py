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

"""Sample ordering strategies for benchmark dataset traversal.

SampleOrder is an infinite iterator yielding dataset indices. It never raises
StopIteration — termination is controlled by BenchmarkSession._make_stop_check().
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..config.runtime_settings import RuntimeSettings, SampleOrderKind


class SampleOrder(ABC):
    """Abstract base class for sample ordering strategies.

    Yields dataset sample indices indefinitely. Different strategies enable
    different testing scenarios (balanced coverage vs random sampling).

    Attributes:
        n_samples_in_dataset: Number of unique samples available in dataset.
        rng: Random number generator for reproducible randomness.
    """

    def __init__(
        self,
        n_samples_in_dataset: int,
        rng: random.Random = random,  # type: ignore[assignment]
    ):
        if n_samples_in_dataset <= 0:
            raise ValueError(
                f"n_samples_in_dataset must be > 0, got {n_samples_in_dataset}"
            )
        self.n_samples_in_dataset = n_samples_in_dataset
        self.rng = rng

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        return self.next_sample_index()

    @abstractmethod
    def next_sample_index(self) -> int:
        """Get the next sample index to issue.

        Returns:
            Sample index (0 to n_samples_in_dataset-1).
        """
        raise NotImplementedError


class WithoutReplacementSampleOrder(SampleOrder):
    """Shuffle dataset, use all samples before repeating.

    Ensures balanced coverage: shuffles all dataset indices, issues them one
    by one until exhausted, then reshuffles and repeats (infinite cycle).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index_order = list(range(self.n_samples_in_dataset))
        # Force initial shuffle on first call
        self._curr_idx = self.n_samples_in_dataset + 1

    def _reset(self) -> None:
        self.rng.shuffle(self.index_order)
        self._curr_idx = 0

    def next_sample_index(self) -> int:
        if self._curr_idx >= len(self.index_order):
            self._reset()
        retval = self.index_order[self._curr_idx]
        self._curr_idx += 1
        return retval


class SequentialSampleOrder(SampleOrder):
    """Sequential ordering: 0, 1, 2, ..., n-1, 0, 1, ...

    Used for accuracy evaluation to ensure deterministic sample ordering
    that matches reference implementations (e.g. evalscope).
    """

    def __init__(self, n_samples_in_dataset: int):
        super().__init__(n_samples_in_dataset=n_samples_in_dataset)
        self._curr_idx = 0

    def next_sample_index(self) -> int:
        idx = self._curr_idx
        self._curr_idx = (self._curr_idx + 1) % self.n_samples_in_dataset
        return idx


class WithReplacementSampleOrder(SampleOrder):
    """Truly random sampling from dataset with replacement.

    Each sample is chosen uniformly at random, independent of previous choices.
    """

    def next_sample_index(self) -> int:
        return self.rng.randint(0, self.n_samples_in_dataset - 1)


class SingleSampleOrder(SampleOrder):
    """Always yield one fixed dataset index (issue the same sample every call).

    The index is fixed at construction and bounds-checked against the dataset
    size; the rng is unused.
    """

    def __init__(self, sample_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= sample_index < self.n_samples_in_dataset:
            raise ValueError(
                f"sample_index must be in [0, {self.n_samples_in_dataset}), "
                f"got {sample_index}"
            )
        self.sample_index = sample_index

    def next_sample_index(self) -> int:
        return self.sample_index


def create_sample_order(
    settings: RuntimeSettings, sequential: bool = False
) -> SampleOrder:
    """Create a SampleOrder from RuntimeSettings, switching on sample_order spec.

    Args:
        settings: Runtime configuration.
        sequential: If True, use sequential ordering (for accuracy evaluation).
    """
    if sequential:
        return SequentialSampleOrder(
            n_samples_in_dataset=settings.n_samples_from_dataset,
        )
    spec = settings.sample_order
    if spec.kind == SampleOrderKind.SINGLE:
        assert spec.fixed_index is not None, "SINGLE kind requires fixed_index"
        return SingleSampleOrder(
            sample_index=spec.fixed_index,
            n_samples_in_dataset=settings.n_samples_from_dataset,
            rng=settings.rng_sample_index,
        )
    if spec.kind == SampleOrderKind.WITH_REPLACEMENT:
        return WithReplacementSampleOrder(
            n_samples_in_dataset=settings.n_samples_from_dataset,
            rng=settings.rng_sample_index,
        )
    return WithoutReplacementSampleOrder(
        n_samples_in_dataset=settings.n_samples_from_dataset,
        rng=settings.rng_sample_index,
    )
