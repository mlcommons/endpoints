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


def test_concurrency_scheduler(concurrency_runtime_settings, clean_sample_event_hooks):
    """Test ConcurrencyScheduler properly gates issuance by completions.
    Tests that concurrency is properly limited and queries are gated by completion events.
    """
    target_concurrency = concurrency_runtime_settings.load_pattern.target_concurrency
    total_samples = concurrency_runtime_settings.n_samples_to_issue

    scheduler = ConcurrencyScheduler(
        concurrency_runtime_settings, WithReplacementSampleOrder
    )

    # Track events with synchronization primitives instead of timing
    issue_lock = threading.Lock()
    complete_lock = threading.Lock()
    issue_events = []  # List of issued query indices in order
    complete_events = []  # List of completed query indices in order

    # Use events for deterministic synchronization
    issue_gates = [threading.Event() for _ in range(total_samples)]
    completion_gates = [threading.Event() for _ in range(total_samples)]

    # Track concurrency level
    concurrency_lock = threading.Lock()
    current_inflight = 0
    max_inflight = 0

    def simulate_completions():
        """Simulate query completions with event-based synchronization."""
        nonlocal current_inflight, max_inflight

        for i in range(total_samples):
            # Wait for this query to be issued before completing it
            issue_gates[i].wait(timeout=5.0)

            # Simulate small variable processing time
            time.sleep(0.001 * (1 + i % 3))  # 1-3ms pattern

            with complete_lock:
                complete_events.append(i)

            # Decrease inflight count
            with concurrency_lock:
                current_inflight -= 1

            # Signal completion to scheduler
            scheduler._release_slot()
            completion_gates[i].set()

    completion_thread = threading.Thread(target=simulate_completions, daemon=True)
    completion_thread.start()

    try:
        # Issue queries through scheduler
        for query_idx, _ in enumerate(scheduler):
            with issue_lock:
                issue_events.append(query_idx)

            # Track peak concurrency
            with concurrency_lock:
                current_inflight += 1
                max_inflight = max(max_inflight, current_inflight)

            # Signal that this query has been issued
            issue_gates[query_idx].set()

        # Wait for all completions to finish
        for i in range(total_samples):
            assert completion_gates[i].wait(
                timeout=5.0
            ), f"Query {i} completion timed out"

        # === Deterministic Verification ===

        # Validation: All queries were issued in sequential order
        assert (
            len(issue_events) == total_samples
        ), f"Expected {total_samples} issues, got {len(issue_events)}"
        for i, query_idx in enumerate(issue_events):
            assert (
                query_idx == i
            ), f"Issue order violated: position {i} has query {query_idx}"

        # Validation: All queries completed
        assert (
            len(complete_events) == total_samples
        ), f"Expected {total_samples} completions, got {len(complete_events)}"

        # Validation: Peak concurrency actually reached target
        assert (
            max_inflight == target_concurrency
        ), f"Max concurrent ({max_inflight}) never reached target ({target_concurrency})"

        # Validation: gating behavior
        # For query i where i >= target_concurrency, query (i - target_concurrency)
        # must have completed before query i could issue.
        #
        # We can verify this by checking that when we issued query i,
        # query (i - target_concurrency) had already been issued AND the scheduler
        # had received its completion event.
        #
        # Since the scheduler blocks until a slot is free, and slots are freed by
        # completions, if query i issued, then at least (i - target_concurrency + 1)
        # completions must have occurred (to free up a slot).
        for i in range(target_concurrency, total_samples):
            expected_completed_query = i - target_concurrency
            assert (
                expected_completed_query in complete_events
            ), f"Query {i} issued but query {expected_completed_query} not in completions yet"

    finally:
        # Ensure proper cleanup
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
