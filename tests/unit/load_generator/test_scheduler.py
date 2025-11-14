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

import pytest
from inference_endpoint.load_generator.sample import SampleEventHandler
from inference_endpoint.load_generator.scheduler import (
    ConcurrencyScheduler,
    MaxThroughputScheduler,
    PoissonDistributionScheduler,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
)
from scipy import stats


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


def test_with_replacement_sample_order(random_seed):
    ordering = WithReplacementSampleOrder(12345, 100, rng=random.Random(random_seed))
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


@pytest.mark.parametrize("target_concurrency", [1, 2, 100, 1000], indirect=True)
def test_concurrency_scheduler(concurrency_runtime_settings, target_concurrency):
    """Test ConcurrencyScheduler properly gates issuance by completions."""
    total_samples = concurrency_runtime_settings.n_samples_to_issue

    scheduler = ConcurrencyScheduler(
        concurrency_runtime_settings, WithReplacementSampleOrder
    )

    # State tracking
    state_lock = threading.RLock()
    issued_count = 0
    completed_count = 0
    current_inflight = 0
    max_inflight = 0

    # Synchronization: signal when queries can complete and when they're done
    can_complete = [threading.Event() for _ in range(total_samples)]
    completed = [threading.Event() for _ in range(total_samples)]
    # Signal when each query is issued
    issued = [threading.Event() for _ in range(total_samples)]

    def completion_worker():
        """Waits for signals to complete queries."""
        nonlocal completed_count, current_inflight

        for position in range(total_samples):
            can_complete[position].wait()

            with state_lock:
                completed_count += 1
                current_inflight -= 1
                assert current_inflight >= 0, "Inflight count went negative"

            scheduler._release_slot()
            completed[position].set()

    threading.Thread(target=completion_worker, daemon=True).start()

    def issue_worker():
        """Issues queries through scheduler."""
        nonlocal issued_count, current_inflight, max_inflight

        for position, _ in enumerate(scheduler):
            with state_lock:
                issued_count += 1
                current_inflight += 1
                max_inflight = max(max_inflight, current_inflight)
                assert (
                    current_inflight <= target_concurrency
                ), f"Concurrency {current_inflight} exceeded limit {target_concurrency}"
            issued[position].set()

    issue_thread = threading.Thread(target=issue_worker, daemon=True)
    issue_thread.start()

    try:
        # Phase 1: First target_concurrency queries issue immediately
        for position in range(target_concurrency):
            issued[position].wait()

        with state_lock:
            assert issued_count == target_concurrency
            assert completed_count == 0
            assert current_inflight == target_concurrency

        # Phase 2: Verify scheduler blocks when at capacity, unblocks on completion
        for position in range(target_concurrency, total_samples):
            position_to_complete = position - target_concurrency

            # Verify next query hasn't issued yet (scheduler is blocking)
            assert not issued[
                position
            ].is_set(), f"Query {position} issued before slot was freed"

            # Free a slot
            can_complete[position_to_complete].set()
            completed[position_to_complete].wait()

            # Verify next query now issues
            issued[position].wait()

            with state_lock:
                assert current_inflight == target_concurrency

        # Phase 3: Complete remaining queries and cleanup
        for position in range(target_concurrency, total_samples):
            can_complete[position].set()
            completed[position].wait()

        issue_thread.join()

        # Final validation
        with state_lock:
            assert issued_count == total_samples
            assert completed_count == total_samples
            assert current_inflight == 0
            assert max_inflight == target_concurrency

    finally:
        SampleEventHandler.clear_hooks()


@pytest.mark.parametrize("target_qps", [50.0, 100.0, 500.0, 1000.0], indirect=True)
def test_poisson_scheduler_distribution(poisson_runtime_settings, target_qps):
    """Test PoissonDistributionScheduler produces exponentially distributed inter-arrival times.

    For a Poisson process with rate λ (target QPS), inter-arrival times must follow
    exponential distribution with mean = 1/λ.

    Three-tier validation:
    1. Mean with 99.9% confidence interval
    2. Coefficient of Variation (CV) ≈ 1.0 (exponential signature)
    3. Kolmogorov-Smirnov test for distribution shape
    """
    scheduler = PoissonDistributionScheduler(
        poisson_runtime_settings, WithReplacementSampleOrder
    )

    # Test configuration
    TARGET_QPS = target_qps
    expected_mean_s = 1.0 / TARGET_QPS

    # Collect delays from scheduler (in seconds) for statistical analysis
    delays_s = []
    for _, delay_ns in scheduler:
        delays_s.append(delay_ns / 1e9)  # Convert ns to seconds

    # Validate sufficient sample size
    n = len(delays_s)

    # Calculate sample statistics using Bessel's correction for unbiased variance (whitened)
    sample_mean = sum(delays_s) / n
    sample_variance = sum((x - sample_mean) ** 2 for x in delays_s) / (n - 1)
    sample_std = math.sqrt(sample_variance)
    cv = sample_std / sample_mean

    # Test 1: Mean with statistical confidence interval (99.9% CI)
    # For exponential: std(X̄) = sigma/√n = mu/√n
    z_critical = 3.29  # 99.9% two-tailed
    margin_of_error = z_critical * (sample_std / math.sqrt(n))
    assert abs(sample_mean - expected_mean_s) < margin_of_error, (
        f"Mean {sample_mean*1000:.3f}ms outside 99.9% CI: "
        f"[{(expected_mean_s - margin_of_error)*1000:.3f}, "
        f"{(expected_mean_s + margin_of_error)*1000:.3f}] ms"
    )

    # Test 2: CV should be close to 1.0 (exponential property: std = mean)
    # Use adaptive tolerance based on sample size, max(10%, 1 std. error)
    cv_tolerance = max(0.10, 1.0 / math.sqrt(n))
    assert (
        abs(cv - 1.0) < cv_tolerance
    ), f"CV {cv:.3f} deviates from 1.0 by more than {cv_tolerance:.3f}"

    # Test 3: Kolmogorov-Smirnov test for exponential distribution
    # kstest compares data against exponential CDF with scale parameter = mean
    ks_statistic, p_value = stats.kstest(
        delays_s,
        "expon",
        args=(0, sample_mean),  # loc=0 (no shift), scale=mean
        alternative="two-sided",
    )

    # Reject if p-value < 0.0001 (99.99% confidence that distribution is NOT exponential)
    ALPHA = 0.0001
    assert p_value > ALPHA, (
        f"KS test rejected exponential distribution: "
        f"p-value={p_value:.4f} < alpha={ALPHA} (D={ks_statistic:.4f})"
    )
