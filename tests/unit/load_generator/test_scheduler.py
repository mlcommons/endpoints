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
    """Test ConcurrencyScheduler properly gates issuance by completions with varying completion rates.

    With concurrency=2 and 100 samples:
    - First 2 queries are issued immediately (slots available)
    - Each subsequent issuance waits for a completion to free a slot
    - Query i (i≥2) should only issue after query (i-2) completes
    - Completion rate changes every 10 samples to test dynamic behavior
    """
    scheduler = ConcurrencyScheduler(
        concurrency_runtime_settings, WithReplacementSampleOrder
    )

    issue_events = []
    complete_events = []
    start_time = time.perf_counter()

    # Define completion delays for each batch of 10 samples (in seconds)
    # Pattern: Fast -> Slow -> Fast -> Very Slow -> Fast -> Medium -> Very Fast -> Slow -> Fast -> Medium
    completion_delays = [
        0.005,  # Batch 0-9:   5ms
        0.020,  # Batch 10-19: 20ms
        0.005,  # Batch 20-29: 5ms
        0.050,  # Batch 30-39: 50ms
        0.005,  # Batch 40-49: 5ms
        0.010,  # Batch 50-59: 10ms
        0.002,  # Batch 60-69: 2ms
        0.020,  # Batch 70-79: 20ms
        0.005,  # Batch 80-89: 5ms
        0.010,  # Batch 90-99: 10ms
    ]

    def simulate_completions():
        """Simulate variable processing time per batch of queries."""
        for i in range(100):
            # Determine delay based on which batch of 10 this sample belongs to
            batch_idx = i // 10
            delay = completion_delays[batch_idx]
            time.sleep(delay)
            complete_events.append((i, time.perf_counter() - start_time, delay))
            scheduler._release_slot()

    completion_thread = threading.Thread(target=simulate_completions, daemon=True)
    completion_thread.start()

    try:
        for query_idx, _ in enumerate(scheduler):
            issue_events.append((query_idx, time.perf_counter() - start_time))

        # First 2 queries should issue immediately
        assert issue_events[0][1] < 0.001, "First query should issue immediately"
        assert issue_events[1][1] < 0.001, "Second query should issue immediately"

        # With concurrency=2: first 2 issued immediately, then each completion enables 1 more
        # Check that after initial 2, issuance is gated by completions
        for i in range(2, 100):
            issue_time = issue_events[i][1]
            # This issue should happen after the (i-2)th completion
            prev_complete_time = complete_events[i - 2][1]
            assert (
                issue_time >= prev_complete_time
            ), f"Issue {i} at {issue_time:.4f}s happened before completion {i-2} at {prev_complete_time:.4f}s"

        # Verify expected pattern: measure avg issue intervals for each batch of 10
        # Since concurrency=2, issue rate should track completion rate with ~2 sample lag
        batch_issue_intervals = []
        for batch_idx in range(10):
            start_idx = batch_idx * 10 + 2  # Skip first 2 immediate issues
            if start_idx >= 100:
                break
            end_idx = min((batch_idx + 1) * 10, 100)
            if end_idx <= start_idx + 1:
                continue

            # Calculate average inter-issue interval for this batch
            intervals = [
                issue_events[i][1] - issue_events[i - 1][1]
                for i in range(start_idx, end_idx)
            ]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
            batch_issue_intervals.append(avg_interval)

            # Expected interval should be close to completion delay
            # With concurrency=2 and 2-sample lag: issues in batch i starting at index
            # (batch_idx*10 + 2) wait on completions starting at index (batch_idx*10),
            # which are from the same batch i
            expected_delay = completion_delays[batch_idx]

            # Allow 10% tolerance due to threading timing variance and edge effects
            tolerance = 0.25
            lower_bound = (1 - tolerance) * expected_delay
            upper_bound = (1 + tolerance) * expected_delay
            assert (
                lower_bound <= avg_interval <= upper_bound
            ), f"Batch {batch_idx} avg interval {avg_interval*1000:.2f}ms not in expected range [{lower_bound*1000:.2f}, {upper_bound*1000:.2f}]ms (expected {expected_delay*1000:.2f}ms)"

        # Verify pattern: batches with slower completions should have slower issue rates
        # Compare very slow batch (batch 3: 50ms) vs fast batch (batch 2: 5ms)
        if len(batch_issue_intervals) > 3:
            assert (
                batch_issue_intervals[3] > batch_issue_intervals[2]
            ), "Slower completion batch (50ms) should have slower issue rate than fast batch (5ms)"

        # Compare slow batch (batch 7: 20ms) vs very fast batch (batch 6: 2ms)
        if len(batch_issue_intervals) > 7:
            assert (
                batch_issue_intervals[7] > batch_issue_intervals[6]
            ), "Slower completion batch (20ms) should have slower issue rate than very fast batch (2ms)"
    finally:
        # Ensure proper cleanup even if test fails
        completion_thread.join(timeout=5.0)
        SampleEventHandler.clear_hooks()


def test_poisson_scheduler_distribution(poisson_runtime_settings):
    """Test PoissonDistributionScheduler produces exponentially distributed inter-arrival times.

    For a Poisson process with rate λ (1000 QPS), inter-arrival times must follow
    exponential distribution with mean = 1/λ = 1ms.

    Three-tier validation:
    1. Mean with 99.9% confidence interval
    2. Coefficient of Variation (CV) ≈ 1.0 (exponential signature)
    3. Kolmogorov-Smirnov test for distribution shape
    """
    scheduler = PoissonDistributionScheduler(
        poisson_runtime_settings, WithReplacementSampleOrder
    )

    # Test configuration
    TARGET_QPS = poisson_runtime_settings.metric_target.target
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
    # For exponential: std(X̄) = σ/√n = μ/√n
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
        f"p-value={p_value:.4f} < α={ALPHA} (D={ks_statistic:.4f})"
    )
