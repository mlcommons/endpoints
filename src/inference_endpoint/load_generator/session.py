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
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import msgspec

from ..config.runtime_settings import RuntimeSettings
from ..config.schema import LoadPatternType
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


class EndpointResponseIdleTimeoutError(RuntimeError):
    """An endpoint stopped returning response progress while work was in flight."""


# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------


class PhaseType(str, Enum):
    """Phase types control tracking and reporting behavior."""

    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    WARMUP = "warmup"


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    """Configuration for a single benchmark phase."""

    name: str
    runtime_settings: RuntimeSettings
    dataset: Dataset
    phase_type: PhaseType = PhaseType.PERFORMANCE
    drain_after: bool = True
    drain_timeout: float | None = None
    strategy: LoadStrategy | None = field(default=None, compare=False)
    routing_headers: tuple[str, ...] = ()


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
# EventPublisher protocol
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
        "_on_inflight_drained",
        "_on_inflight_started",
        "_performance_tracking_stopped",
        "_prompt_warning_reasons",
        "_publisher",
        "_routing_headers",
        "_stop_check",
        "uuid_to_index",
        "uuid_to_conv_info",
        "completed_uuids",
        "inflight",
        "inflight_started_ns",
        "issued_count",
    )

    def __init__(
        self,
        dataset: Dataset,
        issuer: SampleIssuer,
        publisher: EventPublisher,
        stop_check: Callable[[], bool],
        on_inflight_started: Callable[[], None] | None = None,
        on_inflight_drained: Callable[[], None] | None = None,
        routing_headers: tuple[str, ...] = (),
    ):
        self._dataset = dataset
        self._issuer = issuer
        self._publisher = publisher
        self._stop_check = stop_check
        self._on_inflight_started = on_inflight_started
        self._on_inflight_drained = on_inflight_drained or (lambda: None)
        self._routing_headers = routing_headers
        self.uuid_to_index: dict[str, int] = {}
        self.uuid_to_conv_info: dict[str, tuple[str, int | None]] = {}
        self.completed_uuids: set[str] = set()
        self.inflight: int = 0
        # Set on the 0 -> 1 transition and cleared when the phase drains.
        self.inflight_started_ns: int | None = None
        self.issued_count: int = 0
        self._performance_tracking_stopped = False
        self._prompt_warning_reasons: set[str] = set()

    def _warn_prompt_once(self, reason: str, message: str) -> None:
        """Warn once per phase when ISL cannot be derived from a sample."""
        if reason in self._prompt_warning_reasons:
            return
        self._prompt_warning_reasons.add(reason)
        logger.warning(message)

    def mark_inflight_complete(self) -> None:
        if self.inflight <= 0:
            logger.warning("Ignoring completion with no in-flight request")
            return
        self.inflight -= 1
        if self.inflight == 0:
            self.inflight_started_ns = None
            self._on_inflight_drained()

    def stop_performance_tracking(self) -> None:
        """Publish STOP_PERFORMANCE_TRACKING once for this phase."""
        if self._performance_tracking_stopped:
            return
        self._performance_tracking_stopped = True
        self._publisher.publish(
            EventRecord(
                event_type=SessionEventType.STOP_PERFORMANCE_TRACKING,
                timestamp_ns=time.monotonic_ns(),
            )
        )
        self._publisher.flush()

    def issue(
        self,
        sample_index: int,
        data_override: dict[str, Any] | None = None,
        conversation_id: str = "",
        turn: int | None = None,
    ) -> str | None:
        """Load data, build Query, publish ISSUED, send to endpoint.

        Returns query_id on success, None if session is stopping.

        Args:
            sample_index: Index into the dataset.
            data_override: If provided, merged over the loaded sample data.
                Keys in data_override take precedence. Used by AgenticInferenceStrategy
                to override pre-baked messages when trajectory salting is enabled.

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
        headers = (
            dict.fromkeys(self._routing_headers, conversation_id)
            if conversation_id
            else {}
        )
        query = Query(id=query_id, data=data, headers=headers)
        self.uuid_to_index[query_id] = sample_index
        self.uuid_to_conv_info[query_id] = (conversation_id, turn)
        ts = time.monotonic_ns()
        prompt_data: PromptData
        if isinstance(data, dict):
            input_tokens = data.get("input_tokens")
            token_ids = data.get("token_ids")
            messages = data.get("messages")
            prompt = data.get("prompt")
            if input_tokens is not None and token_ids is not None:
                raise ValueError("sample contains both input_tokens and token_ids")

            if input_tokens is not None:
                prompt_data = PromptData(token_ids=tuple(input_tokens))
            elif token_ids is not None:
                prompt_data = PromptData(token_ids=tuple(token_ids))
            elif isinstance(messages, list | tuple) and messages:
                tools = data.get("tools")
                chat_template_kwargs = data.get("chat_template_kwargs")
                prompt_data = PromptData(
                    messages=tuple(messages),
                    tools=tuple(tools) if isinstance(tools, list | tuple) else None,
                    chat_template_kwargs=(
                        dict(chat_template_kwargs)
                        if isinstance(chat_template_kwargs, dict)
                        else None
                    ),
                    chat_template=(
                        data["chat_template"]
                        if isinstance(data.get("chat_template"), str)
                        else None
                    ),
                    tool_choice=(
                        data["tool_choice"]
                        if isinstance(data.get("tool_choice"), str | dict)
                        else None
                    ),
                )
            elif isinstance(prompt, str):
                prompt_data = PromptData(text=prompt)
            else:
                if not isinstance(prompt, list):
                    self._warn_prompt_once(
                        "unsupported_mapping",
                        "Samples without token IDs, non-empty messages, or a string "
                        "prompt are issued normally, but ISL is unavailable",
                    )
                prompt_data = PromptData()
        else:
            self._warn_prompt_once(
                "non_mapping",
                "Non-mapping samples are issued normally, but ISL is unavailable",
            )
            prompt_data = PromptData()
        self._publisher.publish(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=ts,
                sample_uuid=query_id,
                conversation_id=conversation_id,
                turn=turn,
                data=prompt_data,
            )
        )
        self._issuer.issue(query)
        starting_inflight = self.inflight == 0
        if starting_inflight and self._on_inflight_started is not None:
            self.inflight_started_ns = time.monotonic_ns()
        self.inflight += 1
        # The callback arms the liveness guard, which reads this counter, so it
        # must observe the incremented value rather than the prior idle state.
        if starting_inflight and self._on_inflight_started is not None:
            self._on_inflight_started()
        self.issued_count += 1
        return query_id

    def register_skipped(
        self,
        sample_index: int,
        conversation_id: str = "",
        turn: int | None = None,
    ) -> str | None:
        if self._stop_check():
            return None
        query_id = uuid.uuid4().hex
        self.uuid_to_index[query_id] = sample_index
        self.uuid_to_conv_info[query_id] = (conversation_id, turn)
        self.completed_uuids.add(query_id)
        self._publisher.publish(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=time.monotonic_ns(),
                sample_uuid=query_id,
                conversation_id=conversation_id,
                turn=turn,
                data=PromptData(),
            )
        )
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
        endpoint_response_idle_timeout_s: float | None = None,
    ):
        self._issuer = issuer
        self._publisher = event_publisher
        self._loop = loop
        self._on_sample_complete = on_sample_complete
        self.session_id = session_id or uuid.uuid4().hex

        # Mutable state
        self._stop_requested = False
        self._current_phase_stopped = False
        self._done = False
        self._current_phase_issuer: PhaseIssuer | None = None
        self._current_phase_type: PhaseType | None = None
        self._current_strategy: LoadStrategy | None = None
        self._recv_task: asyncio.Task | None = None
        self._strategy_task: asyncio.Task | None = None
        self._drain_event = asyncio.Event()
        self._fatal_error: RuntimeError | None = None
        self._last_response_progress_ns: int | None = None
        # Liveness guard: a single re-arming timer, not a task. It exists only
        # while work is in flight, so the disabled and idle paths cost nothing.
        self._endpoint_response_idle_timeout_ns = (
            int(endpoint_response_idle_timeout_s * 1_000_000_000)
            if endpoint_response_idle_timeout_s is not None
            else None
        )
        self._progress_timer: asyncio.TimerHandle | None = None

    def stop(self) -> None:
        """Signal early termination. Safe to call from signal handler.

        Cancels the running strategy task to unblock strategies that may be
        waiting on semaphores or other async primitives. Also sets the drain
        event to unblock _drain_inflight if it's waiting for responses.
        """
        self._stop_requested = True
        self._drain_event.set()
        self._disarm_endpoint_response_idle_timeout()
        if self._strategy_task and not self._strategy_task.done():
            self._strategy_task.cancel()

    @property
    def stop_requested(self) -> bool:
        """True once stop() ran — Ctrl-C, transport closure, or watchdog.

        Distinguishes a session whose run() returned after an abort from one
        that completed normally; the per-phase cap (stop_current_phase) does
        NOT set it, since reaching max_issue_duration_ms is a normal phase end.
        """
        return self._stop_requested

    def stop_current_phase(self) -> None:
        """Stop issuing in the current phase without aborting the session.

        Cancels a strategy blocked on an async primitive, but deliberately
        leaves the drain event untouched: already-issued requests must still
        complete or exhaust the phase drain timeout before the next phase.
        """
        if self._strategy_task is None or self._strategy_task.done():
            return
        self._current_phase_stopped = True
        self._strategy_task.cancel()

    async def run(
        self,
        phases: list[PhaseConfig],
        on_phase_start: Callable[[PhaseConfig], None] | None = None,
    ) -> SessionResult:
        """Run all benchmark phases sequentially.

        Returns SessionResult with per-phase results.
        """
        session_start = time.monotonic_ns()
        self._publish_session_event(SessionEventType.STARTED)

        self._recv_task = asyncio.create_task(self._receive_responses())
        phase_results: list[PhaseResult] = []

        try:
            for phase in phases:
                if self._fatal_error is not None:
                    raise self._fatal_error
                if self._stop_requested:
                    break
                if on_phase_start is not None:
                    on_phase_start(phase)
                result = await self._run_phase(phase)
                if result is not None:
                    phase_results.append(result)
                if self._fatal_error is not None:
                    raise self._fatal_error
        finally:
            self._done = True
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
                try:
                    await self._recv_task
                except asyncio.CancelledError:
                    pass
            self._disarm_endpoint_response_idle_timeout()
            if self._stop_requested:
                # Aborted run (Ctrl-C, transport closure, run watchdog): mark
                # it BEFORE the terminal ENDED so the aggregator's ENDED-driven
                # finalize — which still drains buffered samples first — writes
                # state=interrupted rather than a normal COMPLETE snapshot.
                self._publish_session_event(SessionEventType.INTERRUPTED)
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
        # Per-phase stop flag is scoped to this phase; clear any cap left set by
        # a previous phase so it can't short-circuit this one.
        self._current_phase_stopped = False
        # A new phase must not inherit a response timestamp from the previous
        # phase: its first in-flight request gets a full liveness interval.
        if self._endpoint_response_idle_timeout_ns is not None:
            self._last_response_progress_ns = None
            self._disarm_endpoint_response_idle_timeout()

        # Create per-phase state
        if phase.strategy is not None:
            strategy = phase.strategy
        else:
            sample_order = create_sample_order(
                phase.runtime_settings,
                sequential=(phase.phase_type == PhaseType.ACCURACY),
            )
            strategy = create_load_strategy(
                phase.runtime_settings, self._loop, sample_order
            )
        phase_issuer = PhaseIssuer(
            dataset=phase.dataset,
            issuer=self._issuer,
            publisher=self._publisher,
            stop_check=self._make_stop_check(phase.runtime_settings, phase_start),
            on_inflight_started=(
                self._on_inflight_started
                if self._endpoint_response_idle_timeout_ns is not None
                else None
            ),
            on_inflight_drained=self._on_inflight_drained,
            routing_headers=phase.routing_headers,
        )

        self._current_phase_issuer = phase_issuer
        self._current_phase_type = phase.phase_type
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

        if phase.drain_after:
            await self._drain_inflight(phase_issuer, phase.drain_timeout)

        if phase.phase_type == PhaseType.PERFORMANCE:
            phase_issuer.stop_performance_tracking()

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

    async def _drain_inflight(
        self, phase_issuer: PhaseIssuer, timeout: float | None = None
    ) -> None:
        """Wait for all in-flight responses from this phase to complete.

        Bounded by ``timeout`` seconds; on expiry the whole session is aborted
        because advancing with unresolved requests would mix phase accounting.
        ``timeout=None`` intentionally permits long accuracy and offline drains.
        A dropped transport still unblocks via the receive-loop close path.
        """
        if phase_issuer.inflight <= 0 or self._stop_requested:
            return
        logger.info("Draining %d in-flight responses...", phase_issuer.inflight)
        self._drain_event.clear()
        # Re-check after clear: a completion may have set the event between the
        # initial inflight check and clear().
        if phase_issuer.inflight <= 0 or self._stop_requested:
            return
        if timeout is None:
            await self._drain_event.wait()
            return
        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
        except TimeoutError:
            logger.error(
                "Drain timed out after %.0f s with %d responses still in flight; "
                "aborting the session.",
                timeout,
                phase_issuer.inflight,
            )
            self.stop()

    async def _receive_responses(self) -> None:
        """Receive responses from the issuer. Runs as a concurrent task."""
        try:
            while not self._done:
                resp = await self._issuer.recv()
                if resp is None:
                    # Transport closed unexpectedly — trigger stop so strategy
                    # and drain don't hang waiting for responses that will never arrive.
                    logger.warning("Issuer recv() returned None — transport closed")
                    self._stop_requested = True
                    self._drain_event.set()  # Unblock _drain_inflight
                    # Cancel a strategy blocked awaiting a semaphore that will
                    # never be released.
                    if self._strategy_task and not self._strategy_task.done():
                        self._strategy_task.cancel()
                    break
                # One clock read per response, shared by the event records and
                # the liveness stamp.
                now_ns = time.monotonic_ns()
                self._handle_response(resp, now_ns)
                if self._endpoint_response_idle_timeout_ns is not None:
                    self._last_response_progress_ns = now_ns
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = RuntimeError(f"Endpoint response receiver failed: {exc}")
            error.__cause__ = exc
            # First error wins: a stalled endpoint often drops its connection
            # too, and the transport error that follows is a symptom of the
            # diagnosis already recorded, not a better one.
            if self._fatal_error is None:
                self._fatal_error = error
            logger.exception("%s", error)
            self.stop()

    def _on_inflight_started(self) -> None:
        """Arm the liveness guard for a 0-to-1 in-flight transition."""
        if (
            self._endpoint_response_idle_timeout_ns is None
            or self._progress_timer is not None
        ):
            return
        self._progress_timer = self._loop.call_later(
            self._endpoint_response_idle_timeout_ns / 1e9,
            self._fire_endpoint_response_idle_timeout,
        )

    def _on_inflight_drained(self) -> None:
        """Unblock drain waits and retire the liveness guard."""
        self._drain_event.set()
        self._disarm_endpoint_response_idle_timeout()

    def _disarm_endpoint_response_idle_timeout(self) -> None:
        if self._progress_timer is not None:
            self._progress_timer.cancel()
            self._progress_timer = None

    def _fire_endpoint_response_idle_timeout(self) -> None:
        """Fail the run when outstanding work makes no response progress.

        This is transport- and engine-agnostic: a response chunk or final result
        is progress. It deliberately does not try to classify TensorRT-LLM or
        vLLM errors, so it also catches the silent worker hang where no HTTP
        response is ever produced.
        """
        self._progress_timer = None
        timeout_ns = self._endpoint_response_idle_timeout_ns
        assert timeout_ns is not None
        if self._done or self._stop_requested:
            return
        phase_issuer = self._current_phase_issuer
        if (
            phase_issuer is None
            or phase_issuer.inflight <= 0
            or phase_issuer.inflight_started_ns is None
        ):
            return
        last_progress_ns = max(
            timestamp
            for timestamp in (
                self._last_response_progress_ns,
                phase_issuer.inflight_started_ns,
            )
            if timestamp is not None
        )
        idle_ns = time.monotonic_ns() - last_progress_ns
        if idle_ns < timeout_ns:
            # Progress raced the deadline: re-arm for the remainder rather than
            # reporting a stall that did not happen.
            self._progress_timer = self._loop.call_later(
                (timeout_ns - idle_ns) / 1e9, self._fire_endpoint_response_idle_timeout
            )
            return
        error = EndpointResponseIdleTimeoutError(
            "Endpoint response idle timeout after "
            f"{timeout_ns / 1e9:.1f}s with {phase_issuer.inflight} request(s) in flight"
        )
        logger.error("%s", error)
        if self._fatal_error is None:
            self._fatal_error = error
        self.stop()

    def _handle_response(self, resp: QueryResult | StreamChunk, now_ns: int) -> None:
        """Route a response to the appropriate handler.

        Transport contract for streaming: the worker sends intermediate
        StreamChunk messages for timing events, then a final QueryResult
        with accumulated output for completion.

        `now_ns` is the caller's arrival timestamp, used for every event record
        that would otherwise read the clock again.
        """
        phase_issuer = self._current_phase_issuer

        if isinstance(resp, QueryResult):
            query_id = resp.id
            # Drop late responses for queries already synthetically terminated
            # (e.g. by AgenticInferenceStrategy._handle_timeout). Without this gate,
            # a real response arriving after timeout double-publishes ERROR/COMPLETE
            # and double-decrements inflight (no per-request HTTP timeout
            # exists in endpoint_client; late arrivals are possible).
            if phase_issuer is not None and query_id in phase_issuer.completed_uuids:
                return

            conv_id_str, turn_num = ("", None)
            if phase_issuer is not None:
                conv_id_str, turn_num = phase_issuer.uuid_to_conv_info.pop(
                    query_id, ("", None)
                )

            # Emit ERROR before COMPLETE for failed queries so downstream
            # consumers (notably the metrics aggregator) see the ERROR
            # while the in-flight tracked row still exists. COMPLETE
            # removes the row, so any state lookup at ERROR time after
            # COMPLETE would silently miss tracked failures.
            #
            # Invariant: the EventPublisher MUST preserve publish-call
            # order on the wire (ZMQ PUB→SUB delivers in order to a
            # single SUB, and ZmqMessagePublisher batches without
            # reordering). Any future transport refactor that breaks
            # this property breaks tracked-failure counting — and
            # silently, since neither side has an assertion.
            if resp.error is not None:
                self._publisher.publish(
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=now_ns,
                        sample_uuid=query_id,
                        conversation_id=conv_id_str,
                        turn=turn_num,
                        data=resp.error,
                    )
                )
            if self._current_phase_type != PhaseType.WARMUP:
                finish_reason = resp.metadata.get("finish_reason")
                worker_id = resp.metadata.get("worker_id")
                self._publisher.publish(
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        timestamp_ns=resp.completed_at
                        if isinstance(resp.completed_at, int)
                        else now_ns,
                        sample_uuid=query_id,
                        conversation_id=conv_id_str,
                        turn=turn_num,
                        data=resp.response_output,
                        finish_reason=(
                            finish_reason
                            if isinstance(finish_reason, str)
                            else msgspec.UNSET
                        ),
                        worker_id=(
                            worker_id if isinstance(worker_id, int) else msgspec.UNSET
                        ),
                    )
                )

            if phase_issuer is not None and query_id in phase_issuer.uuid_to_index:
                phase_issuer.mark_inflight_complete()
                if self._current_strategy:
                    self._current_strategy.on_query_complete(query_id)
                if (
                    self._on_sample_complete
                    and self._current_phase_type != PhaseType.WARMUP
                ):
                    self._on_sample_complete(resp)

        elif isinstance(resp, StreamChunk):
            is_first = resp.metadata.get("first_chunk", False)
            event_type = (
                SampleEventType.RECV_FIRST
                if is_first
                else SampleEventType.RECV_NON_FIRST
            )
            conv_id_str, turn_num = ("", None)
            if phase_issuer is not None:
                conv_id_str, turn_num = phase_issuer.uuid_to_conv_info.get(
                    resp.id, ("", None)
                )
            self._publisher.publish(
                EventRecord(
                    event_type=event_type,
                    timestamp_ns=now_ns,
                    sample_uuid=resp.id,
                    conversation_id=conv_id_str,
                    turn=turn_num,
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
            settings.max_issue_duration_ms * 1_000_000
            if settings.max_issue_duration_ms is not None
            else 0
        )
        total_samples = settings.total_samples_to_issue()
        stop_on_sample_count = not (
            settings.load_pattern is not None
            and settings.load_pattern.type == LoadPatternType.AGENTIC_INFERENCE
        )

        def check() -> bool:
            if self._stop_requested or self._current_phase_stopped:
                return True
            if (
                stop_on_sample_count
                and self._current_phase_issuer
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
