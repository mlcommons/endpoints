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

"""Tests for the async BenchmarkSession."""

from __future__ import annotations

import asyncio
import random

import pytest
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import LoadPattern, LoadPatternType
from inference_endpoint.core.record import (
    ErrorEventType,
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import ErrorData, Query, QueryResult, StreamChunk
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseIssuer,
    PhaseResult,
    PhaseType,
    SessionResult,
    _extract_prompt_text,
)
from inference_endpoint.metrics.metric import Throughput

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeDataset(Dataset):
    """In-memory dataset for tests."""

    def __init__(self, n_samples: int = 10):
        self._n = n_samples

    def load_sample(self, index: int) -> dict:
        return {"prompt": f"sample_{index}", "model": "test"}

    def num_samples(self) -> int:
        return self._n


class FakeIssuer:
    """Fake SampleIssuer that queues responses for controlled delivery."""

    def __init__(self, response_delay: float = 0.001):
        self._issued: list[Query] = []
        self._response_queue: asyncio.Queue[QueryResult | StreamChunk | None] = (
            asyncio.Queue()
        )
        self._response_delay = response_delay
        self._auto_respond = True
        self._loop: asyncio.AbstractEventLoop | None = None

    def issue(self, query: Query) -> None:
        self._issued.append(query)
        if self._auto_respond and self._loop:

            def _enqueue_response(q: Query = query) -> None:
                self._response_queue.put_nowait(
                    QueryResult(id=q.id, response_output=None)
                )

            self._loop.call_later(self._response_delay, _enqueue_response)

    async def recv(self) -> QueryResult | StreamChunk | None:
        return await self._response_queue.get()

    def shutdown(self) -> None:
        self._response_queue.put_nowait(None)

    def inject_response(self, resp: QueryResult | StreamChunk) -> None:
        self._response_queue.put_nowait(resp)

    @property
    def issued_queries(self) -> list[Query]:
        return self._issued


class FakePublisher:
    """Captures published EventRecords."""

    def __init__(self):
        self.events: list[EventRecord] = []

    def publish(self, event_record: EventRecord) -> None:
        self.events.append(event_record)

    def flush(self) -> None:
        pass

    def events_of_type(self, event_type) -> list[EventRecord]:
        return [e for e in self.events if e.event_type == event_type]


def _make_settings(
    load_pattern: LoadPattern | None = None,
    n_samples: int = 10,
    max_duration_ms: int | None = None,
) -> RuntimeSettings:
    return RuntimeSettings(
        metric_target=Throughput(100),
        reported_metrics=[],
        min_duration_ms=0,
        max_duration_ms=max_duration_ms,
        n_samples_from_dataset=n_samples,
        n_samples_to_issue=n_samples,
        min_sample_count=n_samples,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=load_pattern or LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )


# ---------------------------------------------------------------------------
# PhaseIssuer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseIssuer:
    def test_issue_builds_query_and_publishes(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        issuer._auto_respond = False
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: False)

        result = phase_issuer.issue(3)
        assert result is not None
        assert phase_issuer.issued_count == 1
        assert phase_issuer.inflight == 1
        assert len(issuer.issued_queries) == 1
        assert issuer.issued_queries[0].id == result
        assert 3 in phase_issuer.uuid_to_index.values()
        # Single-turn callers omit conv_id/turn — defaults flow through.
        assert phase_issuer.uuid_to_conv_info[result] == ("", None)

        # Should have published ISSUED event
        issued_events = publisher.events_of_type(SampleEventType.ISSUED)
        assert len(issued_events) == 1
        assert issued_events[0].sample_uuid == result
        assert issued_events[0].conversation_id == ""
        assert issued_events[0].turn is None

    def test_issue_returns_none_when_stopped(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        issuer._auto_respond = False
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: True)

        result = phase_issuer.issue(0)
        assert result is None
        assert phase_issuer.issued_count == 0

    def test_uuid_is_unique_per_issue(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        issuer._auto_respond = False
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: False)

        ids = [phase_issuer.issue(i % 5) for i in range(10)]
        assert len(set(ids)) == 10

    def test_issue_stamps_conversation_id_and_turn_on_issued_event(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        issuer._auto_respond = False
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: False)

        query_id = phase_issuer.issue(2, conversation_id="conv-1", turn=3)
        assert query_id is not None
        assert issuer.issued_queries[0].headers == {"X-Session-ID": "conv-1"}
        assert phase_issuer.uuid_to_conv_info[query_id] == ("conv-1", 3)

        issued = publisher.events_of_type(SampleEventType.ISSUED)
        assert len(issued) == 1
        assert issued[0].sample_uuid == query_id
        assert issued[0].conversation_id == "conv-1"
        assert issued[0].turn == 3

    def test_register_skipped_populates_state_without_issuing_http(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        issuer._auto_respond = False
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: False)

        qid = phase_issuer.register_skipped(2, conversation_id="c1", turn=4)

        assert qid is not None
        assert phase_issuer.uuid_to_index[qid] == 2
        assert phase_issuer.uuid_to_conv_info[qid] == ("c1", 4)
        assert qid in phase_issuer.completed_uuids
        assert phase_issuer.issued_count == 1
        assert phase_issuer.inflight == 0
        assert issuer.issued_queries == []

        issued = publisher.events_of_type(SampleEventType.ISSUED)
        assert len(issued) == 1
        assert issued[0].sample_uuid == qid
        assert issued[0].conversation_id == "c1"
        assert issued[0].turn == 4

    def test_mark_inflight_complete_invokes_callback_when_drained(self):
        dataset = FakeDataset(5)
        issuer = FakeIssuer()
        publisher = FakePublisher()
        drained: list[bool] = []
        phase_issuer = PhaseIssuer(
            dataset,
            issuer,
            publisher,
            lambda: False,
            on_inflight_drained=lambda: drained.append(True),
        )

        phase_issuer.inflight = 1
        phase_issuer.mark_inflight_complete()

        assert phase_issuer.inflight == 0
        assert drained == [True]

    def test_register_skipped_returns_none_when_stopped(self):
        phase_issuer = PhaseIssuer(
            FakeDataset(5), FakeIssuer(), FakePublisher(), lambda: True
        )
        assert phase_issuer.register_skipped(0) is None
        assert phase_issuer.issued_count == 0


# ---------------------------------------------------------------------------
# BenchmarkSession tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkSession:
    @pytest.mark.asyncio
    async def test_single_perf_phase(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig("perf", _make_settings(n_samples=5), FakeDataset(5)),
        ]
        result = await session.run(phases)

        assert len(result.phase_results) == 1
        assert result.perf_results[0].name == "perf"
        assert result.perf_results[0].issued_count == 5
        assert len(result.perf_results[0].uuid_to_index) == 5

        # Check session events
        started = publisher.events_of_type(SessionEventType.STARTED)
        ended = publisher.events_of_type(SessionEventType.ENDED)
        start_track = publisher.events_of_type(
            SessionEventType.START_PERFORMANCE_TRACKING
        )
        stop_track = publisher.events_of_type(
            SessionEventType.STOP_PERFORMANCE_TRACKING
        )
        assert len(started) == 1
        assert len(ended) == 1
        assert len(start_track) == 1
        assert len(stop_track) == 1

    @pytest.mark.asyncio
    async def test_accuracy_phase(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig(
                "acc", _make_settings(n_samples=3), FakeDataset(3), PhaseType.ACCURACY
            ),
        ]
        result = await session.run(phases)

        assert len(result.accuracy_results) == 1
        assert result.accuracy_results[0].issued_count == 3
        # No tracking events for accuracy
        assert (
            len(publisher.events_of_type(SessionEventType.START_PERFORMANCE_TRACKING))
            == 0
        )

    @pytest.mark.asyncio
    async def test_warmup_produces_no_result(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig(
                "warmup",
                _make_settings(n_samples=3),
                FakeDataset(3),
                PhaseType.WARMUP,
            ),
        ]
        result = await session.run(phases)
        assert len(result.phase_results) == 0

    @pytest.mark.asyncio
    async def test_multi_phase(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig(
                "warmup",
                _make_settings(n_samples=2),
                FakeDataset(2),
                PhaseType.WARMUP,
            ),
            PhaseConfig(
                "perf",
                _make_settings(n_samples=5),
                FakeDataset(5),
                PhaseType.PERFORMANCE,
            ),
            PhaseConfig(
                "acc", _make_settings(n_samples=3), FakeDataset(3), PhaseType.ACCURACY
            ),
        ]
        result = await session.run(phases)

        assert len(result.perf_results) == 1
        assert result.perf_results[0].issued_count == 5
        assert len(result.accuracy_results) == 1
        assert result.accuracy_results[0].issued_count == 3

    @pytest.mark.asyncio
    async def test_stop_terminates_early(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)

        # Stop after a short delay
        loop.call_later(0.05, session.stop)

        phases = [
            PhaseConfig(
                "perf",
                _make_settings(n_samples=100_000, max_duration_ms=10_000),
                FakeDataset(100),
            ),
        ]
        result = await session.run(phases)
        # Should have stopped early, not issued all 100k
        assert result.perf_results[0].issued_count < 100_000

    @pytest.mark.asyncio
    async def test_stop_current_phase_advances_to_accuracy(self):
        """A perf-phase timeout must end only that phase, not skip accuracy.

        Mirrors _PerfPhaseTimeout firing mid-perf: stop_current_phase cancels
        the perf strategy without setting the session-wide stop flag, so the
        following accuracy phase still runs to completion.
        """
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        loop.call_later(0.05, session.stop_current_phase)

        phases = [
            PhaseConfig(
                "perf",
                _make_settings(n_samples=100_000, max_duration_ms=10_000),
                FakeDataset(100),
                PhaseType.PERFORMANCE,
            ),
            PhaseConfig(
                "acc", _make_settings(n_samples=3), FakeDataset(3), PhaseType.ACCURACY
            ),
        ]
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)

        # Perf was cut short by the phase-scoped stop...
        assert result.perf_results[0].issued_count < 100_000
        # ...but the accuracy phase still ran to completion.
        assert len(result.accuracy_results) == 1
        assert result.accuracy_results[0].issued_count == 3
        # The session-wide stop flag was never set.
        assert session._stop_requested is False

    @pytest.mark.asyncio
    async def test_stop_current_phase_unblocks_unbounded_drain(self):
        """The per-phase cap must break an in-progress unbounded drain wait.

        If the cap fires while the phase is already inside ``_drain_inflight``
        (strategy task finished, so cancelling it is a no-op) with a stuck
        in-flight response and ``timeout=None``, the drain would hang forever
        unless ``stop_current_phase`` also sets the drain event.
        """
        loop = asyncio.get_running_loop()
        session = BenchmarkSession(FakeIssuer(), FakePublisher(), loop)

        class _StuckIssuer:
            inflight = 3  # never drains

        loop.call_later(0.02, session.stop_current_phase)
        await asyncio.wait_for(
            session._drain_inflight(_StuckIssuer(), timeout=None), timeout=2.0
        )
        assert session._current_phase_stopped is True

    @pytest.mark.asyncio
    async def test_on_sample_complete_callback(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        completed: list[str] = []

        def on_complete(result: QueryResult) -> None:
            completed.append(result.id)

        session = BenchmarkSession(
            issuer, publisher, loop, on_sample_complete=on_complete
        )
        phases = [
            PhaseConfig("perf", _make_settings(n_samples=5), FakeDataset(5)),
        ]
        await session.run(phases)
        assert len(completed) == 5

    @pytest.mark.asyncio
    async def test_stale_completions_ignored_by_strategy(self):
        """Responses from warmup phase should not affect perf phase strategy."""
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        # Issuer that delays responses significantly so they arrive in next phase
        issuer = FakeIssuer(response_delay=0.1)
        issuer._loop = loop

        session = BenchmarkSession(issuer, publisher, loop)

        concurrency_settings = _make_settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.CONCURRENCY, target_concurrency=2
            ),
            n_samples=3,
        )
        phases = [
            PhaseConfig(
                "sat", _make_settings(n_samples=2), FakeDataset(2), PhaseType.WARMUP
            ),
            PhaseConfig(
                "perf", concurrency_settings, FakeDataset(3), PhaseType.PERFORMANCE
            ),
        ]
        result = await session.run(phases)

        # Perf phase should complete with its own samples, not be confused by stale ones
        assert len(result.perf_results) == 1
        assert result.perf_results[0].issued_count == 3

    @pytest.mark.asyncio
    async def test_recv_none_triggers_stop(self):
        """If issuer.recv() returns None mid-phase, drain should abort quickly."""
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig("perf", _make_settings(n_samples=5), FakeDataset(5)),
        ]

        # Schedule transport close after a short delay — recv returns None
        loop.call_later(0.05, issuer.shutdown)

        # Session should complete quickly — recv None sets stop_requested,
        # which aborts drain. wait_for prevents CI hang if this regresses.
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_streaming_query_completes_via_queryresult(self):
        """Streaming: StreamChunks publish timing events, QueryResult handles completion.

        The worker sends StreamChunk(first) → StreamChunk(delta) → QueryResult.
        Only the QueryResult decrements inflight and releases the concurrency
        semaphore. StreamChunks only publish timing events.
        """
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        session = BenchmarkSession(issuer, publisher, loop)

        settings = _make_settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.CONCURRENCY, target_concurrency=1
            ),
            n_samples=2,
        )
        phases = [PhaseConfig("perf", settings, FakeDataset(2))]

        async def inject_streaming_responses():
            """Simulate worker: StreamChunk(first) → StreamChunk(delta) → QueryResult."""
            while not issuer._issued:
                await asyncio.sleep(0.005)
            q1 = issuer._issued[0]
            issuer.inject_response(
                StreamChunk(id=q1.id, metadata={"first_chunk": True})
            )
            issuer.inject_response(StreamChunk(id=q1.id, response_chunk="more"))
            issuer.inject_response(QueryResult(id=q1.id, response_output="out1"))
            while len(issuer._issued) < 2:
                await asyncio.sleep(0.005)
            q2 = issuer._issued[1]
            issuer.inject_response(
                StreamChunk(id=q2.id, metadata={"first_chunk": True})
            )
            issuer.inject_response(StreamChunk(id=q2.id, response_chunk="more"))
            issuer.inject_response(QueryResult(id=q2.id, response_output="out2"))

        injector = asyncio.create_task(inject_streaming_responses())
        result = await asyncio.wait_for(session.run(phases), timeout=5.0)
        await injector

        assert result.perf_results[0].issued_count == 2
        recv_first = publisher.events_of_type(SampleEventType.RECV_FIRST)
        assert len(recv_first) == 2

    @pytest.mark.asyncio
    async def test_concurrency_strategy_transport_close_no_deadlock(self):
        """ConcurrencyStrategy must not deadlock when transport closes mid-phase."""
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer(response_delay=999)  # Responses never arrive in time
        issuer._loop = loop
        issuer._auto_respond = False

        session = BenchmarkSession(issuer, publisher, loop)
        settings = _make_settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.CONCURRENCY, target_concurrency=2
            ),
            n_samples=100,
        )
        phases = [PhaseConfig("perf", settings, FakeDataset(10))]

        # Close transport after strategy issues initial batch and blocks on semaphore
        loop.call_later(0.1, issuer.shutdown)

        # Must complete without deadlock — wait_for prevents CI hang
        result = await asyncio.wait_for(session.run(phases), timeout=5.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_on_sample_complete_called_for_streaming_query(self):
        """on_sample_complete fires exactly once per streaming query (on QueryResult).

        StreamChunks only publish timing events — callback fires only for QueryResult.
        """
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        completed: list[QueryResult | StreamChunk] = []

        def on_complete(result: QueryResult | StreamChunk) -> None:
            completed.append(result)

        session = BenchmarkSession(
            issuer, publisher, loop, on_sample_complete=on_complete
        )
        settings = _make_settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.CONCURRENCY, target_concurrency=1
            ),
            n_samples=1,
        )
        phases = [PhaseConfig("perf", settings, FakeDataset(1))]

        async def inject():
            while not issuer._issued:
                await asyncio.sleep(0.005)
            q = issuer._issued[0]
            issuer.inject_response(StreamChunk(id=q.id, metadata={"first_chunk": True}))
            issuer.inject_response(StreamChunk(id=q.id, response_chunk="more"))
            issuer.inject_response(QueryResult(id=q.id, response_output="done"))

        asyncio.create_task(inject())
        await asyncio.wait_for(session.run(phases), timeout=5.0)

        assert len(completed) == 1
        assert isinstance(completed[0], QueryResult)

    @pytest.mark.asyncio
    async def test_failed_query_published_as_error_event(self):
        """Bug #5: QueryResult with error should publish ErrorEventType, not just COMPLETE."""
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        session = BenchmarkSession(issuer, publisher, loop)
        settings = _make_settings(n_samples=1)
        phases = [PhaseConfig("perf", settings, FakeDataset(1))]

        async def inject_error():
            while not issuer._issued:
                await asyncio.sleep(0.005)
            q = issuer._issued[0]
            issuer.inject_response(
                QueryResult(
                    id=q.id,
                    error=ErrorData(error_type="timeout", error_message="timed out"),
                )
            )

        asyncio.create_task(inject_error())
        await asyncio.wait_for(session.run(phases), timeout=5.0)

        # Should have published both COMPLETE and an error event
        complete_events = publisher.events_of_type(SampleEventType.COMPLETE)
        error_events = [
            e for e in publisher.events if isinstance(e.event_type, ErrorEventType)
        ]
        assert len(complete_events) == 1
        # Bug #5: error event should also be published
        assert len(error_events) == 1

        # ERROR must be emitted BEFORE COMPLETE so the metrics aggregator can
        # observe the in-flight tracked row before set_field(...COMPLETE...)
        # removes it. Reverting this order would silently zero
        # tracked_samples_failed.
        error_idx = publisher.events.index(error_events[0])
        complete_idx = publisher.events.index(complete_events[0])
        assert error_idx < complete_idx, (
            f"ERROR event must be emitted before COMPLETE for metrics "
            f"aggregator correctness; got error at idx {error_idx}, "
            f"complete at idx {complete_idx}"
        )

    @pytest.mark.asyncio
    async def test_handle_response_stamps_conversation_id_and_turn(self):
        """All event types inherit (conv_id, turn) seeded at issue time; streaming
        events use .get() so the entry survives for the terminal QueryResult pop."""
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        publisher = FakePublisher()
        session = BenchmarkSession(issuer, publisher, loop)

        phase_issuer = PhaseIssuer(FakeDataset(3), issuer, publisher, lambda: False)
        session._current_phase_issuer = phase_issuer

        # Streaming path: entry stays available for the terminal COMPLETE pop.
        phase_issuer.uuid_to_conv_info["q-stream"] = ("conv-s", 7)
        session._handle_response(
            StreamChunk(id="q-stream", metadata={"first_chunk": True})
        )
        session._handle_response(StreamChunk(id="q-stream", response_chunk="delta"))
        assert (
            publisher.events_of_type(SampleEventType.RECV_FIRST)[0].conversation_id,
            publisher.events_of_type(SampleEventType.RECV_FIRST)[0].turn,
        ) == ("conv-s", 7)
        assert (
            publisher.events_of_type(SampleEventType.RECV_NON_FIRST)[0].conversation_id,
            publisher.events_of_type(SampleEventType.RECV_NON_FIRST)[0].turn,
        ) == ("conv-s", 7)
        assert "q-stream" in phase_issuer.uuid_to_conv_info

        # Success path: COMPLETE inherits conv info, entry is popped.
        phase_issuer.uuid_to_index["q-ok"] = 0
        phase_issuer.uuid_to_conv_info["q-ok"] = ("conv-9", 5)
        phase_issuer.inflight = 1
        session._handle_response(
            QueryResult(
                id="q-ok",
                response_output="ok",
                metadata={"finish_reason": "stop"},
                completed_at=12345,
            )
        )
        complete = publisher.events_of_type(SampleEventType.COMPLETE)
        assert [(e.conversation_id, e.turn) for e in complete] == [("conv-9", 5)]
        assert complete[0].finish_reason == "stop"
        assert "q-ok" not in phase_issuer.uuid_to_conv_info
        assert "q-ok" not in phase_issuer.completed_uuids

        # Error path: ERROR (emitted before COMPLETE) also carries conv info.
        phase_issuer.uuid_to_index["q-err"] = 1
        phase_issuer.uuid_to_conv_info["q-err"] = ("conv-err", 2)
        phase_issuer.inflight = 1
        session._handle_response(
            QueryResult(
                id="q-err",
                error=ErrorData(error_type="boom", error_message="x"),
            )
        )
        error_events = [
            e for e in publisher.events if isinstance(e.event_type, ErrorEventType)
        ]
        assert [(e.conversation_id, e.turn) for e in error_events] == [("conv-err", 2)]
        complete = publisher.events_of_type(SampleEventType.COMPLETE)
        assert (complete[-1].conversation_id, complete[-1].turn) == ("conv-err", 2)
        assert "q-err" not in phase_issuer.completed_uuids


@pytest.mark.unit
class TestBenchmarkSessionPoissonIntegration:
    """Poisson strategy (TimedIssueStrategy) integration with session."""

    @pytest.mark.asyncio
    async def test_poisson_issues_all_samples(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        poisson_settings = _make_settings(
            load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=5000.0),
            n_samples=8,
        )
        phases = [
            PhaseConfig("perf", poisson_settings, FakeDataset(8)),
        ]
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)

        assert len(result.perf_results) == 1
        assert result.perf_results[0].issued_count == 8

    @pytest.mark.asyncio
    async def test_poisson_respects_stop(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        poisson_settings = _make_settings(
            load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=100.0),
            n_samples=100_000,
            max_duration_ms=60_000,
        )
        phases = [
            PhaseConfig("perf", poisson_settings, FakeDataset(100)),
        ]
        loop.call_later(0.05, session.stop)
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)
        assert result.perf_results[0].issued_count < 100_000


@pytest.mark.unit
class TestBenchmarkSessionMaxDuration:
    """max_duration_ms timeout: phase stops after duration even with samples remaining."""

    @pytest.mark.asyncio
    async def test_max_duration_stops_phase(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        # Very short max_duration with many samples to issue
        settings = _make_settings(
            load_pattern=LoadPattern(type=LoadPatternType.POISSON, target_qps=10.0),
            n_samples=100_000,
            max_duration_ms=50,
        )
        phases = [PhaseConfig("perf", settings, FakeDataset(100))]
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)

        # Should have stopped well before issuing all samples
        assert result.perf_results[0].issued_count < 100_000

    @pytest.mark.asyncio
    async def test_max_duration_with_burst(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        settings = _make_settings(n_samples=1_000_000, max_duration_ms=20)
        phases = [PhaseConfig("perf", settings, FakeDataset(100))]
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)

        # Burst fires fast, but stop_check should cut it short
        assert result.perf_results[0].issued_count < 1_000_000


@pytest.mark.unit
class TestBenchmarkSessionAccuracyErrorHandling:
    """Error handling in accuracy phase: query fails, verify it doesn't corrupt scoring."""

    @pytest.mark.asyncio
    async def test_failed_query_in_accuracy_phase_preserves_uuid_map(self):
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()
        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        completed_results: list[QueryResult | StreamChunk] = []

        def on_complete(result: QueryResult | StreamChunk) -> None:
            completed_results.append(result)

        session = BenchmarkSession(
            issuer, publisher, loop, on_sample_complete=on_complete
        )
        settings = _make_settings(n_samples=3)
        phases = [
            PhaseConfig("acc", settings, FakeDataset(3), PhaseType.ACCURACY),
        ]

        async def inject_mixed_responses():
            while len(issuer._issued) < 3:
                await asyncio.sleep(0.005)
            # First query: success
            issuer.inject_response(
                QueryResult(id=issuer._issued[0].id, response_output="answer1")
            )
            # Second query: error
            issuer.inject_response(
                QueryResult(
                    id=issuer._issued[1].id,
                    error=ErrorData(error_type="timeout", error_message="timed out"),
                )
            )
            # Third query: success
            issuer.inject_response(
                QueryResult(id=issuer._issued[2].id, response_output="answer3")
            )

        asyncio.create_task(inject_mixed_responses())
        result = await asyncio.wait_for(session.run(phases), timeout=5.0)

        assert len(result.accuracy_results) == 1
        acc = result.accuracy_results[0]
        # All 3 samples should be in uuid_to_index, including the failed one
        assert acc.issued_count == 3
        assert len(acc.uuid_to_index) == 3
        # on_sample_complete should have fired for all 3
        assert len(completed_results) == 3

    @pytest.mark.asyncio
    async def test_error_event_published_in_accuracy_phase(self):
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()
        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        session = BenchmarkSession(issuer, publisher, loop)
        settings = _make_settings(n_samples=1)
        phases = [
            PhaseConfig("acc", settings, FakeDataset(1), PhaseType.ACCURACY),
        ]

        async def inject_error():
            while not issuer._issued:
                await asyncio.sleep(0.005)
            issuer.inject_response(
                QueryResult(
                    id=issuer._issued[0].id,
                    error=ErrorData(error_type="server_error", error_message="500"),
                )
            )

        asyncio.create_task(inject_error())
        await asyncio.wait_for(session.run(phases), timeout=5.0)

        error_events = [
            e for e in publisher.events if isinstance(e.event_type, ErrorEventType)
        ]
        assert len(error_events) == 1


@pytest.mark.unit
class TestBenchmarkSessionMultiPhaseSatPerfSequence:
    """Multi-perf + warmup sequence (sat -> perf -> sat -> perf)."""

    @pytest.mark.asyncio
    async def test_sat_perf_sat_perf(self):
        loop = asyncio.get_running_loop()
        issuer = FakeIssuer()
        issuer._loop = loop
        publisher = FakePublisher()

        session = BenchmarkSession(issuer, publisher, loop)
        phases = [
            PhaseConfig(
                "warmup1",
                _make_settings(n_samples=2),
                FakeDataset(2),
                PhaseType.WARMUP,
            ),
            PhaseConfig(
                "perf1",
                _make_settings(n_samples=4),
                FakeDataset(4),
                PhaseType.PERFORMANCE,
            ),
            PhaseConfig(
                "warmup2",
                _make_settings(n_samples=3),
                FakeDataset(3),
                PhaseType.WARMUP,
            ),
            PhaseConfig(
                "perf2",
                _make_settings(n_samples=6),
                FakeDataset(6),
                PhaseType.PERFORMANCE,
            ),
        ]
        result = await asyncio.wait_for(session.run(phases), timeout=10.0)

        # Both perf phases should produce results
        assert len(result.perf_results) == 2
        assert result.perf_results[0].name == "perf1"
        assert result.perf_results[0].issued_count == 4
        assert result.perf_results[1].name == "perf2"
        assert result.perf_results[1].issued_count == 6

        # Saturation phases produce no results
        assert len(result.phase_results) == 2

        # Should have start/stop tracking for each perf phase
        start_track = publisher.events_of_type(
            SessionEventType.START_PERFORMANCE_TRACKING
        )
        stop_track = publisher.events_of_type(
            SessionEventType.STOP_PERFORMANCE_TRACKING
        )
        assert len(start_track) == 2
        assert len(stop_track) == 2


@pytest.mark.unit
class TestBenchmarkSessionStaleStreamChunk:
    """Stale StreamChunk from previous phase is ignored."""

    @pytest.mark.asyncio
    async def test_stale_stream_chunk_ignored(self):
        """StreamChunk from warmup phase should not affect perf phase counts."""
        loop = asyncio.get_running_loop()
        publisher = FakePublisher()

        issuer = FakeIssuer()
        issuer._loop = loop
        issuer._auto_respond = False

        completed: list[str] = []

        def on_complete(result: QueryResult | StreamChunk) -> None:
            completed.append(result.id)

        session = BenchmarkSession(
            issuer, publisher, loop, on_sample_complete=on_complete
        )

        # Saturation with slow responses, perf with concurrency
        sat_settings = _make_settings(n_samples=2)
        perf_settings = _make_settings(
            load_pattern=LoadPattern(
                type=LoadPatternType.CONCURRENCY, target_concurrency=1
            ),
            n_samples=2,
        )

        phases = [
            PhaseConfig(
                "sat", sat_settings, FakeDataset(2), PhaseType.WARMUP, drain_after=False
            ),
            PhaseConfig("perf", perf_settings, FakeDataset(2), PhaseType.PERFORMANCE),
        ]

        async def inject_responses():
            # Wait for warmup queries
            while len(issuer._issued) < 2:
                await asyncio.sleep(0.005)
            sat_ids = [q.id for q in issuer._issued[:2]]

            # Wait for perf phase queries to start
            while len(issuer._issued) < 3:
                await asyncio.sleep(0.005)

            # Inject stale StreamChunks from warmup phase into perf phase
            issuer.inject_response(StreamChunk(id=sat_ids[0], response_chunk="stale"))
            issuer.inject_response(StreamChunk(id=sat_ids[1], response_chunk="stale"))

            # Now complete the perf queries
            perf_queries = issuer._issued[2:]
            for q in perf_queries:
                issuer.inject_response(QueryResult(id=q.id, response_output="ok"))
            # Wait for second perf query if not yet issued
            while len(issuer._issued) < 4:
                await asyncio.sleep(0.005)
                for q in issuer._issued[2:]:
                    if q.id not in list(completed):
                        issuer.inject_response(
                            QueryResult(id=q.id, response_output="ok")
                        )

        asyncio.create_task(inject_responses())
        result = await asyncio.wait_for(session.run(phases), timeout=5.0)

        # Perf phase should have exactly 2 issued samples
        assert len(result.perf_results) == 1
        assert result.perf_results[0].issued_count == 2
        # on_sample_complete should only be called for perf-phase queries
        # (stale sat queries are not in perf's uuid_to_index)
        for cid in completed:
            assert cid in result.perf_results[0].uuid_to_index


@pytest.mark.unit
class TestSessionResult:
    def test_perf_results_filter(self):
        results = [
            PhaseResult("sat", PhaseType.WARMUP, {}, 0, 0, 0),
            PhaseResult("perf1", PhaseType.PERFORMANCE, {"a": 1}, 10, 0, 100),
            PhaseResult("perf2", PhaseType.PERFORMANCE, {"b": 2}, 20, 100, 200),
            PhaseResult("acc", PhaseType.ACCURACY, {"c": 3}, 5, 200, 300),
        ]
        sr = SessionResult("test", results, 0, 300)
        assert len(sr.perf_results) == 2
        assert len(sr.accuracy_results) == 1
        assert sr.perf_results[0].name == "perf1"


@pytest.mark.unit
class TestExtractPromptText:
    def test_string_content_extracted(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert _extract_prompt_text(messages) == "Hello\nHi"

    def test_multimodal_list_content_text_parts_extracted(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url"},
                ],
            }
        ]
        assert _extract_prompt_text(messages) == "Describe this image"

    def test_mixed_string_and_list_content(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url"},
                ],
            },
        ]
        assert _extract_prompt_text(messages) == "You are helpful\nWhat is this?"

    def test_none_content_skipped(self):
        messages = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Hello"},
        ]
        assert _extract_prompt_text(messages) == "Hello"

    def test_list_content_with_no_text_parts_returns_none(self):
        messages = [{"role": "user", "content": [{"type": "image_url"}]}]
        assert _extract_prompt_text(messages) is None

    def test_non_dict_messages_skipped(self):
        messages = ["not a dict", {"role": "user", "content": "Valid"}]
        assert _extract_prompt_text(messages) == "Valid"

    def test_tool_calls_included(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = _extract_prompt_text(messages)
        assert result is not None
        assert "What's the weather?" in result
        assert "get_weather" in result


@pytest.mark.unit
class TestBenchmarkSessionHandleResponse:
    """Direct invocation of BenchmarkSession._handle_response (no session.run)."""

    @pytest.mark.asyncio
    async def test_drops_late_response_after_timeout(self):
        """A late QueryResult for a query already in completed_uuids must be a no-op:
        no duplicate ERROR/COMPLETE publish and no second inflight decrement."""
        loop = asyncio.get_running_loop()
        dataset = FakeDataset(1)
        issuer = FakeIssuer()
        publisher = FakePublisher()
        phase_issuer = PhaseIssuer(dataset, issuer, publisher, lambda: False)

        phase_issuer.uuid_to_index["q-late"] = 0
        phase_issuer.completed_uuids.add("q-late")
        phase_issuer.inflight = 1

        session = BenchmarkSession(issuer, publisher, loop)
        session._current_phase_issuer = phase_issuer

        late_resp = QueryResult(
            id="q-late",
            error=ErrorData(error_type="late", error_message="late arrival"),
        )
        session._handle_response(late_resp)

        assert publisher.events == []
        assert phase_issuer.inflight == 1
