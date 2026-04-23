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

"""Async benchmark session: orchestrates phases, issues samples, receives responses.

See docs/load_generator/DESIGN.md for the full design.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..config.runtime_settings import RuntimeSettings
from ..core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from ..core.types import PromptData, Query, QueryResult, StreamChunk
from ..dataset_manager.dataset import Dataset
from .sample_order import create_sample_order
from .strategy import LoadStrategy, create_load_strategy

logger = logging.getLogger(__name__)

_WARMUP_ENABLED = os.environ.get("ENABLE_WARMUP") == "1"


# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------


class PhaseType(str, Enum):
    """Phase types control tracking and reporting behavior."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    WARMUP = "warmup"


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for a single benchmark phase."""

    name: str
    runtime_settings: RuntimeSettings
    dataset: Dataset
    phase_type: PhaseType = PhaseType.PERFORMANCE
    strategy: LoadStrategy | None = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseResult:
    """Result of a single benchmark phase."""

    name: str
    phase_type: PhaseType
    uuid_to_index: dict[str, int]
    issued_count: int
    start_time_ns: int
    end_time_ns: int


@dataclass(frozen=True)
class SessionResult:
    """Combined results from all phases in a session."""

    session_id: str
    phase_results: list[PhaseResult]
    start_time_ns: int
    end_time_ns: int

    @property
    def perf_results(self) -> list[PhaseResult]:
        return [r for r in self.phase_results if r.phase_type == PhaseType.PERFORMANCE]

    @property
    def accuracy_results(self) -> list[PhaseResult]:
        return [r for r in self.phase_results if r.phase_type == PhaseType.ACCURACY]


# ---------------------------------------------------------------------------
# SampleIssuer protocol
# ---------------------------------------------------------------------------


class SampleIssuer(Protocol):
    """Sends queries to an endpoint and receives responses.

    Matches HTTPEndpointClient's interface: issue (sync ZMQ push),
    recv (async ZMQ recv), shutdown.
    """

    def issue(self, query: Query) -> None: ...
    async def recv(self) -> QueryResult | StreamChunk | None: ...
    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# EventRecordPublisher protocol
# ---------------------------------------------------------------------------


class EventPublisher(Protocol):
    """Publishes EventRecords to the metrics pipeline."""

    def publish(self, event_record: EventRecord) -> None: ...
    def flush(self) -> None: ...


# ---------------------------------------------------------------------------
# PhaseIssuer
# ---------------------------------------------------------------------------


class PhaseIssuer:
    """Per-phase state holder that wraps the issue logic.

    Created fresh for each phase. Holds the phase-scoped uuid_to_index map,
    inflight counter, and issued count. Strategies call issue(sample_index)
    to load data, build a Query, publish ISSUED, and send to the endpoint.
    """

    __slots__ = (
        "_dataset",
        "_issuer",
        "_publisher",
        "_stop_check",
        "uuid_to_index",
        "inflight",
        "issued_count",
    )

    def __init__(
        self,
        dataset: Dataset,
        issuer: SampleIssuer,
        publisher: EventPublisher,
        stop_check: Callable[[], bool],
    ):
        self._dataset = dataset
        self._issuer = issuer
        self._publisher = publisher
        self._stop_check = stop_check
        self.uuid_to_index: dict[str, int] = {}
        self.inflight: int = 0
        self.issued_count: int = 0

    def issue(
        self, sample_index: int, data_override: dict[str, Any] | None = None
    ) -> str | None:
        """Load data, build Query, publish ISSUED, send to endpoint.

        Returns query_id on success, None if session is stopping.

        Args:
            sample_index: Index into the dataset.
            data_override: If provided, merged over the loaded sample data.
                Keys in data_override take precedence. Used by MultiTurnStrategy
                to substitute live-accumulated message history.

        Note: load_sample() runs synchronously before the ISSUED timestamp.
        For accurate timing, datasets MUST be pre-loaded into memory.
        Disk-backed datasets will inflate timing and delay subsequent issues.
        """
        if self._stop_check():
            return None
        query_id = uuid.uuid4().hex
        data = self._dataset.load_sample(sample_index)
        if data_override is not None:
            data = {**data, **data_override}
        query = Query(id=query_id, data=data)
        self.uuid_to_index[query_id] = sample_index
        ts = time.monotonic_ns()
        prompt_data: PromptData
        if isinstance(data, dict):
            token_ids = data.get("input_tokens") or data.get("token_ids")
            prompt_data = PromptData(
                text=data.get("prompt"),
                token_ids=tuple(token_ids) if token_ids is not None else None,
            )
        else:
            prompt_data = PromptData()
        self._publisher.publish(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=ts,
                sample_uuid=query_id,
                data=prompt_data,
            )
        )
        self._issuer.issue(query)
        self.inflight += 1
        self.issued_count += 1
        return query_id


# ---------------------------------------------------------------------------
# BenchmarkSession
# ---------------------------------------------------------------------------


class BenchmarkSession:
    """Async benchmark orchestrator. Single thread, single event loop.

    Runs phases sequentially. Each phase gets its own PhaseIssuer and
    LoadStrategy. The receiver coroutine runs concurrently throughout,
    processing responses and routing completions to the active strategy.
    """

    def __init__(
        self,
        issuer: SampleIssuer,
        event_publisher: EventPublisher,
        loop: asyncio.AbstractEventLoop,
        on_sample_complete: Callable[[QueryResult], None] | None = None,
        session_id: str | None = None,
    ):
        self._issuer = issuer
        self._publisher = event_publisher
        self._loop = loop
        self._on_sample_complete = on_sample_complete
        self.session_id = session_id or uuid.uuid4().hex

        # Mutable state
        self._stop_requested = False
        self._done = False
        self._current_phase_issuer: PhaseIssuer | None = None
        self._current_strategy: LoadStrategy | None = None
        self._recv_task: asyncio.Task | None = None
        self._strategy_task: asyncio.Task | None = None
        self._drain_event = asyncio.Event()

    def stop(self) -> None:
        """Signal early termination. Safe to call from signal handler.

        Cancels the running strategy task to unblock strategies that may be
        waiting on semaphores or other async primitives. Also sets the drain
        event to unblock _drain_inflight if it's waiting for responses.
        """
        self._stop_requested = True
        self._drain_event.set()
        if self._strategy_task and not self._strategy_task.done():
            self._strategy_task.cancel()

    async def run(self, phases: list[PhaseConfig]) -> SessionResult:
        """Run all benchmark phases sequentially.

        Returns SessionResult with per-phase results.
        """
        session_start = time.monotonic_ns()
        self._publish_session_event(SessionEventType.STARTED)

        self._recv_task = asyncio.create_task(self._receive_responses())
        phase_results: list[PhaseResult] = []

        try:
            for phase in phases:
                if self._stop_requested:
                    break
                if phase.phase_type == PhaseType.WARMUP and not _WARMUP_ENABLED:
                    logger.info(
                        "Skipping warmup phase %s (set ENABLE_WARMUP=1 to enable)",
                        phase.name,
                    )
                    continue
                result = await self._run_phase(phase)
                if result is not None:
                    phase_results.append(result)
        finally:
            self._done = True
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
                try:
                    await self._recv_task
                except asyncio.CancelledError:
                    pass
            self._publish_session_event(SessionEventType.ENDED)

        return SessionResult(
            session_id=self.session_id,
            phase_results=phase_results,
            start_time_ns=session_start,
            end_time_ns=time.monotonic_ns(),
        )

    async def _run_phase(self, phase: PhaseConfig) -> PhaseResult | None:
        """Run a single phase. Returns PhaseResult or None for warmup."""
        logger.info("Starting phase: %s (%s)", phase.name, phase.phase_type.value)
        phase_start = time.monotonic_ns()

        # Create per-phase state
        if phase.strategy is not None:
            strategy = phase.strategy
        else:
            sample_order = create_sample_order(phase.runtime_settings)
            strategy = create_load_strategy(
                phase.runtime_settings, self._loop, sample_order
            )
        phase_issuer = PhaseIssuer(
            dataset=phase.dataset,
            issuer=self._issuer,
            publisher=self._publisher,
            stop_check=self._make_stop_check(phase.runtime_settings, phase_start),
        )

        self._current_phase_issuer = phase_issuer
        self._current_strategy = strategy

        # Performance phases get tracking events
        if phase.phase_type == PhaseType.PERFORMANCE:
            self._publish_session_event(SessionEventType.START_PERFORMANCE_TRACKING)

        # Execute the strategy as a task so it can be cancelled on transport close
        self._strategy_task = asyncio.create_task(strategy.execute(phase_issuer))
        try:
            await self._strategy_task
        except asyncio.CancelledError:
            logger.info("Strategy cancelled for phase %s", phase.name)
        finally:
            self._strategy_task = None

        # Drain in-flight (skip for warmup — keep concurrency hot)
        if phase.phase_type != PhaseType.WARMUP:
            await self._drain_inflight(phase_issuer)

        if phase.phase_type == PhaseType.PERFORMANCE:
            self._publish_session_event(SessionEventType.STOP_PERFORMANCE_TRACKING)

        phase_end = time.monotonic_ns()
        logger.info(
            "Phase %s complete: %d samples issued",
            phase.name,
            phase_issuer.issued_count,
        )

        # Saturation phases produce no result
        if phase.phase_type == PhaseType.WARMUP:
            return None

        return PhaseResult(
            name=phase.name,
            phase_type=phase.phase_type,
            uuid_to_index=phase_issuer.uuid_to_index,
            issued_count=phase_issuer.issued_count,
            start_time_ns=phase_start,
            end_time_ns=phase_end,
        )

    async def _drain_inflight(self, phase_issuer: PhaseIssuer) -> None:
        """Wait for all in-flight responses from this phase to complete.

        Currently, there is no timeout for the drain step. In the future,
        we can possibly add a dynamic timeout based on the rate of completion
        throughout the current phase."""
        if phase_issuer.inflight <= 0 or self._stop_requested:
            return
        logger.info("Draining %d in-flight responses...", phase_issuer.inflight)
        self._drain_event.clear()
        await self._drain_event.wait()

    async def _receive_responses(self) -> None:
        """Receive responses from the issuer. Runs as a concurrent task."""
        while not self._done:
            resp = await self._issuer.recv()
            if resp is None:
                # Transport closed unexpectedly — trigger stop so strategy
                # and drain don't hang waiting for responses that will never arrive.
                logger.warning("Issuer recv() returned None — transport closed")
                self._stop_requested = True
                self._drain_event.set()  # Unblock _drain_inflight
                # Cancel the strategy task if it's blocked (e.g., ConcurrencyStrategy
                # awaiting sem.acquire() that will never be released).
                if self._strategy_task and not self._strategy_task.done():
                    self._strategy_task.cancel()
                break
            self._handle_response(resp)

    def _handle_response(self, resp: QueryResult | StreamChunk) -> None:
        """Route a response to the appropriate handler.

        Transport contract for streaming: the worker sends intermediate
        StreamChunk messages for timing events, then a final QueryResult
        with accumulated output for completion.
        """
        phase_issuer = self._current_phase_issuer

        if isinstance(resp, QueryResult):
            query_id = resp.id
            self._publisher.publish(
                EventRecord(
                    event_type=SampleEventType.COMPLETE,
                    timestamp_ns=resp.completed_at
                    if isinstance(resp.completed_at, int)
                    else time.monotonic_ns(),
                    sample_uuid=query_id,
                    data=resp.response_output,
                )
            )
            if resp.error is not None:
                self._publisher.publish(
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=time.monotonic_ns(),
                        sample_uuid=query_id,
                        data=resp.error,
                    )
                )
            if phase_issuer is not None and query_id in phase_issuer.uuid_to_index:
                phase_issuer.inflight -= 1
                if phase_issuer.inflight <= 0:
                    self._drain_event.set()
                if self._current_strategy:
                    self._current_strategy.on_query_complete(query_id)
                if self._on_sample_complete:
                    self._on_sample_complete(resp)

        elif isinstance(resp, StreamChunk):
            is_first = resp.metadata.get("first_chunk", False)
            event_type = (
                SampleEventType.RECV_FIRST
                if is_first
                else SampleEventType.RECV_NON_FIRST
            )
            self._publisher.publish(
                EventRecord(
                    event_type=event_type,
                    timestamp_ns=time.monotonic_ns(),
                    sample_uuid=resp.id,
                )
            )

    def _make_stop_check(
        self, settings: RuntimeSettings, phase_start_ns: int
    ) -> Callable[[], bool]:
        """Create a stop-check closure for a phase.

        Reads self._current_phase_issuer at call time (not creation time).
        Invariant: _current_phase_issuer must not change while a phase's
        strategy is executing. This is guaranteed by sequential phase execution.
        """
        max_duration_ns = (
            settings.max_duration_ms * 1_000_000
            if settings.max_duration_ms is not None
            else 0
        )
        total_samples = settings.total_samples_to_issue()

        def check() -> bool:
            if self._stop_requested:
                return True
            if (
                self._current_phase_issuer
                and self._current_phase_issuer.issued_count >= total_samples
            ):
                return True
            if (
                max_duration_ns > 0
                and (time.monotonic_ns() - phase_start_ns) >= max_duration_ns
            ):
                return True
            return False

        return check

    def _publish_session_event(self, event_type: SessionEventType) -> None:
        """Publish a session event and flush the publisher immediately.

        Session events are control signals (STARTED, ENDED, START/STOP
        PERFORMANCE_TRACKING) that subscribers must receive promptly for
        correct state transitions. Flushing ensures any buffered sample
        events are sent first, followed by the session event, so ordering
        is preserved and the signal is not delayed by batching.
        """
        self._publisher.publish(
            EventRecord(event_type=event_type, timestamp_ns=time.monotonic_ns())
        )
        self._publisher.flush()
