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

"""Inter-arrival delay functions for timed load strategies.

Each function returns a callable that produces delay values in nanoseconds.
Used by TimedIssueStrategy for Poisson and other time-based load patterns.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from ..config.schema import LoadPattern, LoadPatternType


def poisson_delay_fn(target_qps: float, rng: random.Random) -> Callable[[], int]:
    """Create a Poisson-distributed delay function.

    Returns inter-arrival delays following an exponential distribution
    (Poisson process). Models realistic client behavior where requests
    arrive independently at a target rate.

    How it works:

    ``expovariate(lambd)`` draws from the exponential distribution with rate
    ``lambd``. Critically, the return value is in units of ``1 / lambd`` —
    NOT in units of ``lambd``. So if ``lambd`` is expressed in
    events-per-nanosecond, the return value is in nanoseconds.

    Step by step for target_qps = 50,000:
        1. lambd = 50,000 / 1e9 = 5e-5 events per nanosecond
        2. expovariate(5e-5) returns values with mean = 1 / 5e-5 = 20,000 ns
        3. So the average inter-arrival delay is 20,000 ns = 20 us
        4. This matches 50,000 QPS: 1 second / 20 us = 50,000 queries

    The return value is cast to int (nanoseconds). The ``max(1, ...)`` guard
    prevents zero-delay at extreme QPS (> 500M), where the mean approaches
    1 ns and the exponential distribution produces sub-1 values ~63% of the
    time. In practice, no system can issue > 500M QPS, so the guard is
    purely defensive.

    Reference: https://docs.python.org/3/library/random.html#random.Random.expovariate

    Args:
        target_qps: Target queries per second.
        rng: Seeded random number generator for reproducibility.

    Returns:
        Callable returning delay in nanoseconds (int, always >= 1).
    """
    if target_qps <= 0:
        raise ValueError(f"target_qps must be > 0, got {target_qps}")
    lambd = target_qps / 1_000_000_000  # events per nanosecond
    return lambda: max(1, int(rng.expovariate(lambd)))


def make_delay_fn(load_pattern: LoadPattern, rng: random.Random) -> Callable[[], int]:
    """Create a delay function from a LoadPattern config.

    Only used by TimedIssueStrategy. MAX_THROUGHPUT uses BurstStrategy,
    CONCURRENCY uses ConcurrencyStrategy — neither needs a delay function.

    Args:
        load_pattern: LoadPattern config from schema.py.
        rng: Seeded random number generator for reproducibility.

    Returns:
        Callable returning delay in nanoseconds.

    Raises:
        ValueError: If load pattern type has no delay function.
    """
    if load_pattern.type == LoadPatternType.POISSON:
        if load_pattern.target_qps is None or load_pattern.target_qps <= 0:
            raise ValueError("Poisson load pattern requires target_qps > 0")
        return poisson_delay_fn(load_pattern.target_qps, rng)

    raise ValueError(
        f"No delay function for load pattern type: {load_pattern.type}. "
        f"MAX_THROUGHPUT uses BurstStrategy, CONCURRENCY uses ConcurrencyStrategy."
    )
