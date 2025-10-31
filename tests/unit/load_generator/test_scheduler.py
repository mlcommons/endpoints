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

import math
import random
import threading
import time

from inference_endpoint.load_generator.sample import SampleEventHandler
from inference_endpoint.load_generator.scheduler import (
    ConcurrencyScheduler,
    MaxThroughputScheduler,
    PoissonDistributionScheduler,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
)


def test_without_replacement_sample_order():
    ordering = WithoutReplacementSampleOrder(12345, 100)
    indices = list(iter(ordering))
    for i in range(0, 12345, 100):
        assert len(set(indices[i : i + 100])) == min(
            100, 12345 - i
        ), "Indices should be unique, and occur at least once"

    # Assert that order is different in each pass of the dataset
    assert (
        indices[:100] != indices[100:200]
    ), "Order should be different in each pass of the dataset"


def test_with_replacement_sample_order():
    ordering = WithReplacementSampleOrder(12345, 100, rng=random.Random(42))
    indices = list(iter(ordering))

    # With Python random.Random(42), the order can be deterministic
    assert indices[:10] == [
        81,
        14,
        3,
        94,
        35,
        31,
        28,
        17,
        94,
        13,
    ], "Order does not match expected deterministic order"
    # Note with this specific seed and order, 94 occurs twice in the first 10 indices
    assert indices[:10].count(94) == 2, "94 should occur twice in the first 10 indices"


def test_max_throughput_scheduler(max_throughput_runtime_settings):
    scheduler = MaxThroughputScheduler(
        max_throughput_runtime_settings, WithReplacementSampleOrder
    )
    indices = list(iter(scheduler))
    assert len(indices) == 100
    for _, delay in indices:
        assert delay == 0
    assert [s_idx for s_idx, _ in indices[:10]] == [
        81,
        14,
        3,
        94,
        35,
        31,
        28,
        17,
        94,
        13,
    ], "Order does not match expected deterministic order"


def test_concurrency_scheduler(concurrency_runtime_settings):
    """Test ConcurrencyScheduler with parallel query processing (concurrency=2, 10ms/query)."""
    scheduler = ConcurrencyScheduler(
        concurrency_runtime_settings, WithReplacementSampleOrder
    )

    # Mimic real system with concurrency=2, 10ms processing time per query
    def complete_queries():
        for _ in range(5):  # 5 batches × 2 = 10 completions
            time.sleep(0.01)  # 10ms between batches
            scheduler._on_query_complete(None)
            scheduler._on_query_complete(None)

    threading.Thread(target=complete_queries, daemon=True).start()

    # Track issue times
    issue_times = []
    start_time = time.perf_counter()

    for _, _ in scheduler:
        issue_times.append(time.perf_counter())

    elapsed = time.perf_counter() - start_time

    # Verify all samples issued
    assert len(issue_times) == 10

    # Expected timeline with concurrency=2, parallel processing at 10ms/query: 40ms
    assert 0.037 < elapsed < 0.043, f"Expected ~40ms, got {elapsed*1000:.1f}ms"

    SampleEventHandler.clear_hooks()


def test_poisson_scheduler_distribution(poisson_runtime_settings):
    """Test PoissonDistributionScheduler produces exponentially distributed inter-arrival times.

    For a Poisson process with rate λ (1000 QPS), inter-arrival times must follow
    exponential distribution with mean = 1/λ = 1ms.

    Key properties tested:
    1. Sample mean ≈ expected mean (1ms)
    2. Coefficient of Variation (CV) ≈ 1.0 (exponential property: std = mean)
    3. Kolmogorov-Smirnov test confirms exponential distribution
    """
    scheduler = PoissonDistributionScheduler(
        poisson_runtime_settings, WithReplacementSampleOrder
    )

    # Collect delays from scheduler (in seconds) for statistical analysis
    # No sleep overhead - just test the generated distribution
    delays_s = []
    sample_count = 5000  # Large sample for robust statistical testing

    for i, (_, delay_ns) in enumerate(scheduler):
        if i >= sample_count:
            break
        delays_s.append(delay_ns / 1e9)  # Convert ns to seconds

    # Expected mean for 1000 QPS = 1/1000 = 0.001s = 1ms
    expected_mean_s = 1.0 / 1000.0

    # Calculate sample statistics
    sample_mean = sum(delays_s) / len(delays_s)
    sample_variance = sum((x - sample_mean) ** 2 for x in delays_s) / len(delays_s)
    sample_std = sample_variance**0.5

    # Coefficient of Variation = std/mean (should be ~1.0 for exponential)
    cv = sample_std / sample_mean

    # Test 1: Mean should match expected (±10% tolerance for large sample)
    assert (
        abs(sample_mean - expected_mean_s) / expected_mean_s < 0.10
    ), f"Mean {sample_mean*1000:.3f}ms deviates >10% from expected {expected_mean_s*1000:.3f}ms"

    # Test 2: CV should be close to 1.0 (exponential property: std = mean)
    # Allow 10% tolerance due to finite sampling
    assert (
        0.90 < cv < 1.10
    ), f"Coefficient of Variation {cv:.3f} not close to 1.0 (exponential signature)"

    # Test 3: Kolmogorov-Smirnov test for exponential distribution
    # Sort delays and compute empirical CDF
    sorted_delays = sorted(delays_s)
    n = len(sorted_delays)

    # Compute KS statistic: max distance between empirical and theoretical CDF
    # Theoretical CDF for exponential: F(x) = 1 - exp(-λx) where λ = 1/mean
    lambda_param = 1.0 / sample_mean

    max_distance = 0
    for i, x in enumerate(sorted_delays):
        # Empirical CDF at x
        ecdf = (i + 1) / n
        # Theoretical exponential CDF at x: F(x) = 1 - exp(-λx)
        theoretical_cdf = 1 - math.exp(-lambda_param * x)
        distance = abs(ecdf - theoretical_cdf)
        max_distance = max(max_distance, distance)

    # From: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Kolmogorov-Smirnov_statistic
    # use α=0.001 strict test: 1.949 / sqrt(n)
    ks_critical = 1.949 / (n**0.5)

    assert max_distance < ks_critical, (
        f"KS test failed: D={max_distance:.4f} > critical={ks_critical:.4f} "
        f"(distribution not exponential at α=0.001 significance level)"
    )
