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

import random
import threading
import time

from inference_endpoint.load_generator.sample import SampleEventHandler
from inference_endpoint.load_generator.scheduler import (
    ConcurrencyScheduler,
    MaxThroughputScheduler,
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

    # Simulate parallel query processing: 2 complete every 10ms
    # (mimics real system with concurrency=2, 10ms processing time per query)
    def complete_queries():
        for _ in range(5):  # 5 batches × 2 = 10 completions
            time.sleep(0.01)  # 10ms between batches
            scheduler._on_query_complete(None)  # First completion in batch
            scheduler._on_query_complete(None)  # Second completion in batch

    threading.Thread(target=complete_queries, daemon=True).start()

    # Track issue times
    issue_times = []
    start_time = time.perf_counter()

    for _, _ in scheduler:
        issue_times.append(time.perf_counter())

    elapsed = time.perf_counter() - start_time

    # Verify all samples issued
    assert len(issue_times) == 10

    # Expected timeline with concurrency=2, parallel processing at 10ms/query:
    # t=0ms:  Issue Q1, Q2 (initial 2 slots)
    # t=10ms: Q1,Q2 complete → Issue Q3, Q4
    # t=20ms: Q3,Q4 complete → Issue Q5, Q6
    # t=30ms: Q5,Q6 complete → Issue Q7, Q8
    # t=40ms: Q7,Q8 complete → Issue Q9, Q10
    # Total: ~40ms (allow 30-60ms for thread scheduling)
    assert 0.03 < elapsed < 0.06, f"Expected ~40ms, got {elapsed*1000:.1f}ms"

    SampleEventHandler.clear_hooks()
