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

"""Tests for load strategies."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from time import monotonic_ns

import pytest
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.load_generator.delay import make_delay_fn, poisson_delay_fn
from inference_endpoint.load_generator.sample_order import WithoutReplacementSampleOrder
from inference_endpoint.load_generator.strategy import (
    BurstStrategy,
    ConcurrencyStrategy,
    TimedIssueStrategy,
    create_load_strategy,
)
from inference_endpoint.metrics.metric import Throughput


def _constant_delay(ns: int = 1_000) -> Callable[[], int]:
    return lambda: ns


# ---------------------------------------------------------------------------
# Mock PhaseIssuer
# ---------------------------------------------------------------------------


class MockPhaseIssuer:
    """Minimal PhaseIssuer for strategy tests."""

    def __init__(self, max_issues: int = 100):
        self.issued_indices: list[int] = []
        self.issued_count: int = 0
        self._max = max_issues

    def issue(self, sample_index: int) -> str | None:
        if self.issued_count >= self._max:
            return None
        self.issued_indices.append(sample_index)
        self.issued_count += 1
        return f"q{self.issued_count}"


# ---------------------------------------------------------------------------
# TimedIssueStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedIssueStrategyCallAt:
    @pytest.mark.asyncio
    async def test_issues_correct_count(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        delay_fn = _constant_delay(1_000)
        strategy = TimedIssueStrategy(delay_fn, order, loop, use_executor=False)

        issuer = MockPhaseIssuer(max_issues=20)
        count = await strategy.execute(issuer)
        assert count == 20
        assert issuer.issued_count == 20

    @pytest.mark.asyncio
    async def test_stops_on_none(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=5, rng=random.Random(42)
        )
        delay_fn = _constant_delay(1_000)
        strategy = TimedIssueStrategy(delay_fn, order, loop, use_executor=False)

        issuer = MockPhaseIssuer(max_issues=3)
        count = await strategy.execute(issuer)
        assert count == 3

    @pytest.mark.asyncio
    async def test_timing_precision(self):
        """call_at should achieve sub-ms precision for moderate delays."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=100, rng=random.Random(42)
        )
        delay_fn = _constant_delay(1_000_000)

        timestamps: list[int] = []

        class TimingIssuer:
            issued_count = 0

            def issue(self, idx):
                timestamps.append(monotonic_ns())
                self.issued_count += 1
                if self.issued_count >= 10:
                    return None
                return f"q{self.issued_count}"

        strategy = TimedIssueStrategy(delay_fn, order, loop, use_executor=False)
        await strategy.execute(TimingIssuer())

        # Check inter-arrival times are positive (callbacks fire in order)
        for i in range(1, len(timestamps)):
            delta_ns = timestamps[i] - timestamps[i - 1]
            assert delta_ns > 0, f"Issue {i}: non-monotonic timestamps"
        # Total elapsed should be roughly 9ms (9 delays of 1ms)
        total_ns = timestamps[-1] - timestamps[0]
        assert (
            total_ns > 5_000_000
        ), f"Total elapsed {total_ns}ns too small for 9x1ms delays"


@pytest.mark.unit
class TestTimedIssueStrategyExecutor:
    @pytest.mark.asyncio
    async def test_issues_correct_count(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        delay_fn = _constant_delay(1_000)
        strategy = TimedIssueStrategy(delay_fn, order, loop, use_executor=True)

        issuer = MockPhaseIssuer(max_issues=20)
        count = await strategy.execute(issuer)
        assert count == 20


# ---------------------------------------------------------------------------
# BurstStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBurstStrategy:
    @pytest.mark.asyncio
    async def test_issues_all(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)

        issuer = MockPhaseIssuer(max_issues=50)
        count = await strategy.execute(issuer)
        assert count == 50

    @pytest.mark.asyncio
    async def test_stops_on_none(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=5, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)

        issuer = MockPhaseIssuer(max_issues=7)
        count = await strategy.execute(issuer)
        assert count == 7

    @pytest.mark.asyncio
    async def test_does_not_starve_event_loop(self):
        """Verify other coroutines get to run during burst issuance.

        We schedule a coroutine that increments a counter each time it wakes.
        If burst issuance yields properly, the counter should be > 0 before
        issuance completes.
        """
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=200, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)

        wakeup_count = 0
        stop = asyncio.Event()

        async def competing_task():
            nonlocal wakeup_count
            while not stop.is_set():
                await asyncio.sleep(0)
                wakeup_count += 1

        task = asyncio.create_task(competing_task())
        issuer = MockPhaseIssuer(max_issues=200)
        await strategy.execute(issuer)
        stop.set()
        await task
        # The competing task should have woken up multiple times during issuance
        assert wakeup_count > 1, f"Competing task only woke {wakeup_count} times"


# ---------------------------------------------------------------------------
# ConcurrencyStrategy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConcurrencyStrategy:
    @pytest.mark.asyncio
    async def test_issues_up_to_concurrency_then_waits(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = ConcurrencyStrategy(target_concurrency=3, sample_order=order)
        issuer = MockPhaseIssuer(max_issues=10)

        # Start strategy but don't await — it should block after 3 issues
        task = asyncio.create_task(strategy.execute(issuer))
        await asyncio.sleep(0.01)  # let it run
        assert issuer.issued_count == 3

        # Simulate completions
        for i in range(1, 4):
            strategy.on_query_complete(f"q{i}")
        await asyncio.sleep(0.01)
        assert issuer.issued_count == 6

        # Complete remaining
        for i in range(4, 11):
            strategy.on_query_complete(f"q{i}")
        count = await asyncio.wait_for(task, timeout=2.0)
        assert count == 10

    @pytest.mark.asyncio
    async def test_stops_on_none(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=100, rng=random.Random(42)
        )
        strategy = ConcurrencyStrategy(target_concurrency=5, sample_order=order)
        issuer = MockPhaseIssuer(max_issues=3)

        # Complete queries as they arrive so strategy doesn't block
        async def completer():
            while True:
                await asyncio.sleep(0.005)
                for i in range(1, issuer.issued_count + 1):
                    strategy.on_query_complete(f"q{i}")

        completer_task = asyncio.create_task(completer())
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=2.0)
        completer_task.cancel()
        try:
            await completer_task
        except asyncio.CancelledError:
            pass
        assert count == 3

    def test_invalid_concurrency_raises(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        with pytest.raises(ValueError, match="target_concurrency must be > 0"):
            ConcurrencyStrategy(target_concurrency=0, sample_order=order)


# ---------------------------------------------------------------------------
# Delay functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDelayFunctions:
    def test_poisson_delay_positive(self):
        fn = poisson_delay_fn(1000.0, random.Random(42))
        delays = [fn() for _ in range(100)]
        assert all(d >= 1 for d in delays)

    def test_poisson_delay_mean(self):
        """Mean delay should be close to 1/target_qps in ns."""
        target_qps = 10_000.0
        fn = poisson_delay_fn(target_qps, random.Random(42))
        delays = [fn() for _ in range(10_000)]
        mean_ns = sum(delays) / len(delays)
        expected_ns = 1e9 / target_qps  # 100_000 ns
        assert abs(mean_ns - expected_ns) / expected_ns < 0.1  # within 10%

    def test_poisson_delay_invalid_qps(self):
        with pytest.raises(ValueError, match="target_qps must be > 0"):
            poisson_delay_fn(0, random.Random(42))

    def test_make_delay_fn_unsupported_pattern(self):
        lp = LoadPattern(type=LoadPatternType.MAX_THROUGHPUT)
        with pytest.raises(ValueError, match="No delay function"):
            make_delay_fn(lp, random.Random(42))


# ---------------------------------------------------------------------------
# create_load_strategy factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateLoadStrategy:
    def test_max_throughput(self):
        loop = asyncio.new_event_loop()
        try:
            settings = _make_settings(LoadPattern(type=LoadPatternType.MAX_THROUGHPUT))
            strategy = create_load_strategy(settings, loop)
            assert isinstance(strategy, BurstStrategy)
        finally:
            loop.close()

    def test_poisson_default(self):
        loop = asyncio.new_event_loop()
        try:
            settings = _make_settings(
                LoadPattern(type=LoadPatternType.POISSON, target_qps=1000.0)
            )
            strategy = create_load_strategy(settings, loop)
            assert isinstance(strategy, TimedIssueStrategy)
            assert not strategy._use_executor
        finally:
            loop.close()

    def test_poisson_executor(self):
        loop = asyncio.new_event_loop()
        try:
            settings = _make_settings(
                LoadPattern(type=LoadPatternType.POISSON, target_qps=1000.0)
            )
            strategy = create_load_strategy(settings, loop, use_executor=True)
            assert isinstance(strategy, TimedIssueStrategy)
            assert strategy._use_executor
        finally:
            loop.close()

    def test_concurrency(self):
        loop = asyncio.new_event_loop()
        try:
            settings = _make_settings(
                LoadPattern(type=LoadPatternType.CONCURRENCY, target_concurrency=32)
            )
            strategy = create_load_strategy(settings, loop)
            assert isinstance(strategy, ConcurrencyStrategy)
            assert strategy._target == 32
        finally:
            loop.close()

    def test_no_load_pattern_raises(self):
        loop = asyncio.new_event_loop()
        try:
            settings = _make_settings(None)
            with pytest.raises(ValueError, match="load_pattern must not be None"):
                create_load_strategy(settings, loop)
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_burst_single_sample(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=1, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)
        issuer = MockPhaseIssuer(max_issues=1)
        count = await strategy.execute(issuer)
        assert count == 1

    @pytest.mark.asyncio
    async def test_burst_stop_immediately(self):
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)
        issuer = MockPhaseIssuer(max_issues=0)
        count = await strategy.execute(issuer)
        assert count == 0

    @pytest.mark.asyncio
    async def test_burst_exception_in_issue_does_not_hang(self):
        """If issue() raises, strategy should not hang forever."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)

        class FailingIssuer:
            issued_count = 0

            def issue(self, idx: int) -> str | None:
                self.issued_count += 1
                if self.issued_count == 3:
                    raise RuntimeError("load_sample failed")
                return f"q{self.issued_count}"

        issuer = FailingIssuer()
        # Must not hang — should complete (with error) within timeout
        with pytest.raises(RuntimeError, match="load_sample failed"):
            await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)

    @pytest.mark.asyncio
    async def test_timed_call_at_exception_in_issue_does_not_hang(self):
        """If issue() raises in call_at callback, strategy should not hang."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = TimedIssueStrategy(
            _constant_delay(1_000), order, loop, use_executor=False
        )

        class FailingIssuer:
            issued_count = 0

            def issue(self, idx: int) -> str | None:
                self.issued_count += 1
                if self.issued_count == 3:
                    raise RuntimeError("load_sample failed")
                return f"q{self.issued_count}"

        issuer = FailingIssuer()
        with pytest.raises(RuntimeError, match="load_sample failed"):
            await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)

    def test_sample_order_single_element(self):
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=1, rng=random.Random(42)
        )
        indices = [next(order) for _ in range(10)]
        assert all(i == 0 for i in indices)


# ---------------------------------------------------------------------------
# Executor mode exceptions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedIssueStrategyExecutorExceptions:
    @pytest.mark.asyncio
    async def test_executor_issue_raises(self):
        """If issue() raises inside run_in_executor path, exception propagates."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        strategy = TimedIssueStrategy(
            _constant_delay(1_000), order, loop, use_executor=True
        )

        call_count = 0

        class FailingIssuer:
            issued_count = 0

            def issue(self, idx: int) -> str | None:
                nonlocal call_count
                call_count += 1
                self.issued_count += 1
                if call_count == 3:
                    raise ValueError("executor callback failed")
                return f"q{call_count}"

        with pytest.raises(ValueError, match="executor callback failed"):
            await asyncio.wait_for(strategy.execute(FailingIssuer()), timeout=5.0)

    @pytest.mark.asyncio
    async def test_executor_delay_fn_raises(self):
        """If delay_fn raises inside executor path, exception propagates."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=10, rng=random.Random(42)
        )
        call_count = 0

        def bad_delay():
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("delay computation failed")
            return 1_000

        strategy = TimedIssueStrategy(bad_delay, order, loop, use_executor=True)
        issuer = MockPhaseIssuer(max_issues=100)

        with pytest.raises(RuntimeError, match="delay computation failed"):
            await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)


# ---------------------------------------------------------------------------
# Concurrent on_query_complete calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConcurrencyStrategyConcurrentCompletions:
    @pytest.mark.asyncio
    async def test_multiple_completions_simultaneously(self):
        """Multiple on_query_complete calls arriving at the same time."""
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=20, rng=random.Random(42)
        )
        strategy = ConcurrencyStrategy(target_concurrency=5, sample_order=order)
        issuer = MockPhaseIssuer(max_issues=20)

        task = asyncio.create_task(strategy.execute(issuer))

        # Let strategy issue initial batch of 5
        await asyncio.sleep(0.02)
        assert issuer.issued_count == 5

        # Release all 5 at once
        for i in range(1, 6):
            strategy.on_query_complete(f"q{i}")
        await asyncio.sleep(0.02)
        assert issuer.issued_count == 10

        # Release next batch all at once
        for i in range(6, 11):
            strategy.on_query_complete(f"q{i}")
        await asyncio.sleep(0.02)
        assert issuer.issued_count == 15

        # Release rest
        for i in range(11, 21):
            strategy.on_query_complete(f"q{i}")
        count = await asyncio.wait_for(task, timeout=2.0)
        assert count == 20

    @pytest.mark.asyncio
    async def test_completions_interleaved_with_issues(self):
        """Completions arriving while new issues are being scheduled."""
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=50, rng=random.Random(42)
        )
        strategy = ConcurrencyStrategy(target_concurrency=2, sample_order=order)
        issuer = MockPhaseIssuer(max_issues=10)

        task = asyncio.create_task(strategy.execute(issuer))
        await asyncio.sleep(0.01)
        assert issuer.issued_count == 2

        # Alternate: complete one, let it issue one more
        for i in range(1, 11):
            strategy.on_query_complete(f"q{i}")
            await asyncio.sleep(0.005)

        count = await asyncio.wait_for(task, timeout=2.0)
        assert count == 10


# ---------------------------------------------------------------------------
# Near-zero delay (high QPS poisson)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimedIssueStrategyNearZeroDelay:
    @pytest.mark.asyncio
    async def test_very_high_qps(self):
        """Poisson with extremely high QPS should still issue all samples."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=50, rng=random.Random(42)
        )
        # 1ns delay -- essentially zero
        strategy = TimedIssueStrategy(
            _constant_delay(1), order, loop, use_executor=False
        )
        issuer = MockPhaseIssuer(max_issues=50)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)
        assert count == 50

    @pytest.mark.asyncio
    async def test_very_high_qps_executor(self):
        """Near-zero delay in executor mode."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=50, rng=random.Random(42)
        )
        strategy = TimedIssueStrategy(
            _constant_delay(1), order, loop, use_executor=True
        )
        issuer = MockPhaseIssuer(max_issues=50)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)
        assert count == 50

    @pytest.mark.asyncio
    async def test_poisson_high_qps_statistical(self):
        """Real poisson distribution at 1M QPS should complete quickly."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=100, rng=random.Random(42)
        )
        delay_fn = poisson_delay_fn(1_000_000.0, random.Random(42))
        strategy = TimedIssueStrategy(delay_fn, order, loop, use_executor=False)
        issuer = MockPhaseIssuer(max_issues=100)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)
        assert count == 100


# ---------------------------------------------------------------------------
# Large-scale burst
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBurstStrategyLargeScale:
    @pytest.mark.asyncio
    async def test_burst_1000_samples(self):
        """BurstStrategy should handle 1000+ samples without issues."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=200, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)
        issuer = MockPhaseIssuer(max_issues=1000)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=10.0)
        assert count == 1000

    @pytest.mark.asyncio
    async def test_burst_5000_samples(self):
        """BurstStrategy at 5000 samples -- verify count and no event loop starvation."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=500, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)

        wakeups = 0
        stop = asyncio.Event()

        async def observer():
            nonlocal wakeups
            while not stop.is_set():
                await asyncio.sleep(0)
                wakeups += 1

        obs_task = asyncio.create_task(observer())
        issuer = MockPhaseIssuer(max_issues=5000)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=10.0)
        stop.set()
        await obs_task

        assert count == 5000
        assert wakeups > 10, f"Event loop starved: observer only ran {wakeups} times"

    @pytest.mark.asyncio
    async def test_burst_indices_wrap_around(self):
        """With dataset_size < issue_count, indices should wrap around."""
        loop = asyncio.get_running_loop()
        order = WithoutReplacementSampleOrder(
            n_samples_in_dataset=3, rng=random.Random(42)
        )
        strategy = BurstStrategy(order, loop)
        issuer = MockPhaseIssuer(max_issues=10)
        count = await asyncio.wait_for(strategy.execute(issuer), timeout=5.0)
        assert count == 10
        # All indices should be 0, 1, or 2
        assert all(0 <= idx <= 2 for idx in issuer.issued_indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(load_pattern):
    """Create minimal RuntimeSettings for factory tests."""
    return RuntimeSettings(
        metric_target=Throughput(100),
        reported_metrics=[],
        min_duration_ms=0,
        max_duration_ms=None,
        n_samples_from_dataset=10,
        n_samples_to_issue=10,
        min_sample_count=10,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=load_pattern,
    )
