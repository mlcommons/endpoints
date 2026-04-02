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

"""Load strategies: controls the pacing of sample issuance.

Three implementations, each optimized for a different load pattern:
- TimedIssueStrategy: Poisson (loop.call_at or run_in_executor)
- BurstStrategy: Max throughput (loop.call_soon)
- ConcurrencyStrategy: Fixed concurrency (asyncio.Semaphore)

See docs/load_generator/design.md for benchmark data and design rationale.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterator
from time import monotonic_ns
from typing import Protocol

from ..config.runtime_settings import RuntimeSettings
from ..config.schema import LoadPatternType
from .delay import make_delay_fn
from .sample_order import SampleOrder, create_sample_order

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoadStrategy Protocol
# ---------------------------------------------------------------------------


class PhaseIssuerProtocol(Protocol):
    """Minimal interface that strategies see for issuing samples."""

    def issue(self, sample_index: int) -> str | None: ...

    issued_count: int


class LoadStrategy(Protocol):
    """Controls the pacing strategy for issuing requests.

    Strategies call phase_issuer.issue(sample_index) to issue each sample.
    issue() returns query_id on success, None when the session should stop.
    """

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        """Drive sample issuance. Returns count of samples issued."""
        ...

    def on_query_complete(self, query_id: str) -> None:
        """Called by session on each QueryResult. Default: no-op.

        Used by ConcurrencyStrategy to release semaphore slots.
        """
        ...


# ---------------------------------------------------------------------------
# TimedIssueStrategy (Poisson)
# ---------------------------------------------------------------------------


def _busy_wait_until(target_ns: int) -> None:
    """Busy-wait in a thread pool thread until target timestamp."""
    while monotonic_ns() < target_ns:
        pass


class TimedIssueStrategy:
    """Schedule-driven load strategy with inter-arrival delays.

    Default mode (call_at): schedules each issue as an event loop callback
    at the precise target time. Zero GIL contention, sub-ms precision.
    Good for <= 50k QPS.

    Executor mode (opt-in): offloads busy-wait to thread pool for sub-100us
    precision. Introduces GIL contention that adds latency at low QPS.
    """

    def __init__(
        self,
        delay_fn: Callable[[], int],
        sample_order: Iterator[int],
        loop: asyncio.AbstractEventLoop,
        use_executor: bool = False,
    ):
        self._delay_fn = delay_fn
        self._sample_order = sample_order
        self._loop = loop
        self._use_executor = use_executor

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        if self._use_executor:
            return await self._execute_executor(phase_issuer)
        return await self._execute_call_at(phase_issuer)

    def on_query_complete(self, query_id: str) -> None:
        pass

    async def _execute_call_at(self, phase_issuer: PhaseIssuerProtocol) -> int:
        done = asyncio.Event()
        start_time = self._loop.time()
        cumulative_s = 0.0

        def schedule_next():
            nonlocal cumulative_s
            idx = next(self._sample_order, None)
            if idx is None:
                done.set()
                return
            cumulative_s += self._delay_fn() / 1e9
            self._loop.call_at(start_time + cumulative_s, fire, idx)

        def fire(idx: int):
            if phase_issuer.issue(idx) is None:
                done.set()
                return
            schedule_next()

        schedule_next()
        await done.wait()
        return phase_issuer.issued_count

    async def _execute_executor(self, phase_issuer: PhaseIssuerProtocol) -> int:
        start = monotonic_ns()
        cumulative = 0
        for idx in self._sample_order:
            cumulative += self._delay_fn()
            target = start + cumulative
            now = monotonic_ns()
            if target > now:
                await self._loop.run_in_executor(None, _busy_wait_until, target)
            if phase_issuer.issue(idx) is None:
                break
        return phase_issuer.issued_count


# ---------------------------------------------------------------------------
# BurstStrategy (Max Throughput)
# ---------------------------------------------------------------------------


class BurstStrategy:
    """Fire-as-fast-as-possible strategy using loop.call_soon.

    Each issue is scheduled as an event loop callback, yielding between
    issues so the receiver coroutine can process responses. Achieves
    100k+ QPS without starving the event loop.
    """

    def __init__(
        self,
        sample_order: Iterator[int],
        loop: asyncio.AbstractEventLoop,
    ):
        self._sample_order = sample_order
        self._loop = loop

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        done = asyncio.Event()

        def issue_next():
            idx = next(self._sample_order, None)
            if idx is None or phase_issuer.issue(idx) is None:
                done.set()
                return
            self._loop.call_soon(issue_next)

        self._loop.call_soon(issue_next)
        await done.wait()
        return phase_issuer.issued_count

    def on_query_complete(self, query_id: str) -> None:
        pass


# ---------------------------------------------------------------------------
# ConcurrencyStrategy
# ---------------------------------------------------------------------------


class ConcurrencyStrategy:
    """Completion-driven strategy maintaining fixed concurrent requests.

    Uses asyncio.Semaphore for gating: acquire before issue, release on
    completion via on_query_complete(). With eager_task_factory, the woken
    waiter executes synchronously within release(), minimizing jitter.
    """

    def __init__(
        self,
        target_concurrency: int,
        sample_order: Iterator[int],
    ):
        if target_concurrency <= 0:
            raise ValueError(
                f"target_concurrency must be > 0, got {target_concurrency}"
            )
        self._target = target_concurrency
        self._sem = asyncio.Semaphore(target_concurrency)
        self._sample_order = sample_order

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        for idx in self._sample_order:
            await self._sem.acquire()
            if phase_issuer.issue(idx) is None:
                self._sem.release()
                break
        return phase_issuer.issued_count

    def on_query_complete(self, query_id: str) -> None:
        self._sem.release()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_load_strategy(
    runtime_settings: RuntimeSettings,
    loop: asyncio.AbstractEventLoop,
    sample_order: SampleOrder | None = None,
    use_executor: bool = False,
) -> LoadStrategy:
    """Create a LoadStrategy from RuntimeSettings.

    Args:
        runtime_settings: Runtime configuration with load_pattern.
        loop: Event loop for scheduling callbacks.
        sample_order: Sample ordering iterator. If None, created from settings.
        use_executor: For Poisson, use run_in_executor for sub-100us precision.

    Returns:
        LoadStrategy implementation for the configured load pattern.
    """
    lp = runtime_settings.load_pattern
    if lp is None:
        raise ValueError("RuntimeSettings.load_pattern must not be None")

    if sample_order is None:
        sample_order = create_sample_order(runtime_settings)

    match lp.type:
        case LoadPatternType.MAX_THROUGHPUT:
            return BurstStrategy(sample_order, loop)

        case LoadPatternType.POISSON:
            delay_fn = make_delay_fn(lp, runtime_settings.rng_sched)
            return TimedIssueStrategy(
                delay_fn, sample_order, loop, use_executor=use_executor
            )

        case LoadPatternType.CONCURRENCY:
            if lp.target_concurrency is None or lp.target_concurrency <= 0:
                raise ValueError(
                    "Concurrency load pattern requires target_concurrency > 0"
                )
            return ConcurrencyStrategy(lp.target_concurrency, sample_order)

        case _:
            raise ValueError(f"Unsupported load pattern type: {lp.type}")
