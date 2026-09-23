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

"""Live steady-state collection inside the metrics aggregator.

The load-bearing test is producer equivalence: one set of ``EventRecord``s is
driven through the aggregator AND written out through the event logger's own
encoder, and the live series must equal ``build_super_pass_series`` over that
log. Everything else here pins an ordering rule that equivalence depends on.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from inference_endpoint.async_utils.services.event_logger.file_writer import JSONLWriter
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import PhaseData, TextModelOutput
from inference_endpoint.metrics.steady_state_diagnostics import (
    build_super_pass_series,
    compute_steady_state_metrics,
)

from .conftest import (
    MockBatchTokenizer,
    count_text_tokens,
    make_aggregator,
    session_event,
)

SUPERPASS = 2


def phase_start(num_turns: int = SUPERPASS, phase_type: str = "performance"):
    return EventRecord(
        event_type=SessionEventType.PHASE_START,
        timestamp_ns=0,
        data=PhaseData(phase_type=phase_type, drain_after=True, num_turns=num_turns),
    )


def _sample(uuid: str, issue: int, ttft: int, decode: int, chunks, turn=None):
    """ISSUED -> RECV_FIRST -> COMPLETE for one streamed sample."""
    return [
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=issue,
            sample_uuid=uuid,
            turn=turn,
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST,
            timestamp_ns=issue + ttft,
            sample_uuid=uuid,
            turn=turn,
            data=TextModelOutput(output=(chunks[0],)),
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=issue + ttft + decode,
            sample_uuid=uuid,
            turn=turn,
            data=TextModelOutput(output=tuple(chunks)),
        ),
    ]


def _event_stream() -> list[EventRecord]:
    """A run exercising every collector rule that equivalence can break on."""
    records = [
        session_event(SessionEventType.STARTED, ts=0),
        # A warmup phase announces first; its samples are untracked, and its
        # announcement must not outlive the performance one.
        phase_start(num_turns=99, phase_type="warmup"),
        *_sample("warm", 10, 5, 50, ["w ", "x y"]),
        phase_start(),
        session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=100),
    ]
    issue = 200
    for i in range(6):
        records += _sample(
            f"s{i}",
            issue,
            10 + i,
            1000 + 10 * i,
            [f"c{i} ", "a b c"],
            turn=(i % 2) + 1,
        )
        issue += 500
    # A retry: re-ISSUED inside the window, so only its issue timestamp moves.
    records += _sample("s3", issue, 12, 900, ["r ", "z z"])
    # Non-streaming output: no post-first-chunk text, so no TPOT contribution.
    records += [
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=issue + 100,
            sample_uuid="ns",
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=issue + 900,
            sample_uuid="ns",
            data=TextModelOutput(output="whole answer"),
        ),
    ]
    # Still in flight when the window closes.
    records.append(
        EventRecord(
            event_type=SampleEventType.ISSUED,
            timestamp_ns=issue + 200,
            sample_uuid="inflight",
        )
    )
    records.append(
        session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=issue + 2000)
    )
    # Re-issued after the window closed. `MetricsTable.set_field` returns early
    # without dropping the in-flight row, so the row still exists; the collector
    # must skip it anyway, or its last_issue_ns -- the throughput denominator --
    # moves and the two producers disagree on system TPS.
    records += _sample("inflight", issue + 900_000, 10, 100, ["late ", "q q"])
    return records


def _write_log(tmp_path, records):
    writer = JSONLWriter(tmp_path / "events")
    for record in records:
        writer.write(record)
    writer.close()
    return str(writer.file_path)


async def _collect(tmp_path, records, socket_name):
    """Drive ``records`` through a steady-state-enabled aggregator."""
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            socket_name,
            tokenizer=MockBatchTokenizer(),
            steady_state_profile="concurrency",
            streaming=True,
        )
        try:
            await agg.process(records)
            if agg.token_queue is not None:
                await agg.token_queue.flush_remaining(None)
            return agg._collector
        finally:
            agg.close()


STEADY_SUPERPASS = 20


def _steady_run_records(n_super_passes: int = 14) -> list[EventRecord]:
    """A run long enough for the detector to certify a steady window.

    Three settling super-passes then a flat plateau, spanning well past the
    600s wall-time floor, so the two producers have to agree on a verdict that
    has an actual window -- TPS, TTFT, TPOT, OSL, latency, CoV and window
    bounds all populated -- rather than agreeing on "not found", which any two
    empty series would also do.
    """
    records = [
        session_event(SessionEventType.STARTED, ts=0),
        phase_start(num_turns=STEADY_SUPERPASS),
        session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=1),
    ]
    issue = 1_000_000_000
    for i in range(n_super_passes * STEADY_SUPERPASS):
        settling = max(0, 3 - i // STEADY_SUPERPASS) * 40_000_000
        wobble = (i % 5) * 1_000_000
        words = 8 + i % 3
        records += _sample(
            f"p{i}",
            issue,
            200_000_000 + settling + wobble,
            words * (20_000_000 + settling + wobble),
            ["head ", "w " * words],
        )
        issue += 6_000_000_000  # 120s per super-pass
    records.append(session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=issue))
    return records


@pytest.mark.unit
class TestProducerEquivalence:
    @pytest.mark.asyncio
    async def test_live_series_equals_the_event_log_parse(self, tmp_path):
        records = _event_stream()
        collector = await _collect(tmp_path, records, "ss_equiv")
        replayed = build_super_pass_series(
            _write_log(tmp_path, records), SUPERPASS, count_text_tokens
        )
        live = collector.series()
        assert live, "the collector produced no super-passes"
        assert [dataclasses.asdict(sp) for sp in live] == [
            dataclasses.asdict(sp) for sp in replayed
        ]

    @pytest.mark.asyncio
    async def test_both_producers_reach_the_same_verdict(self, tmp_path):
        records = _event_stream()
        collector = await _collect(tmp_path, records, "ss_equiv_verdict")
        replayed = build_super_pass_series(
            _write_log(tmp_path, records), SUPERPASS, count_text_tokens
        )
        kwargs = {"superpass_size": SUPERPASS, "warmup": 0}
        assert compute_steady_state_metrics(
            collector.series(), **kwargs
        ) == compute_steady_state_metrics(replayed, **kwargs)

    @pytest.mark.asyncio
    async def test_a_certified_window_is_identical_on_both_producers(self, tmp_path):
        """Equivalence where it counts: a run the detector actually certifies."""
        records = _steady_run_records()
        collector = await _collect(tmp_path, records, "ss_equiv_window")
        replayed = build_super_pass_series(
            _write_log(tmp_path, records), STEADY_SUPERPASS, count_text_tokens
        )
        live_verdict = compute_steady_state_metrics(
            collector.series(), superpass_size=STEADY_SUPERPASS
        )
        assert live_verdict["found"] is True, live_verdict["reason"]
        assert live_verdict["window"]["n_samples"] > 0
        assert live_verdict["tps"]["system"] > 0
        assert live_verdict == compute_steady_state_metrics(
            replayed, superpass_size=STEADY_SUPERPASS
        )


@pytest.mark.unit
class TestOrderingInvariants:
    @pytest.mark.asyncio
    async def test_tpot_lands_in_the_bucket_the_sample_was_issued_into(self, tmp_path):
        """Pins both ordering rules in the aggregator's TPOT path.

        ``on_complete`` must run AFTER ``set_field`` (``TpotTrigger`` fires from
        inside it and reads the sample's super-pass), and ``TpotTrigger`` must
        read that super-pass at fire time, not inside the recorder closure --
        the token count only arrives at the drain flush, long after the row is
        released. Break either and the count has nowhere to go.
        """
        records = [
            session_event(SessionEventType.STARTED, ts=0),
            phase_start(),
            session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=10),
            *_sample("a", 100, 10, 900, ["c ", "x y z"]),
            *_sample("b", 200, 10, 900, ["c ", "x y z"]),
            *_sample("c", 900, 10, 900, ["c ", "x y z"]),
        ]
        collector = await _collect(tmp_path, records, "ss_order")
        series = collector.series()
        assert [sp.n_issued for sp in series] == [2, 1]
        # 3 words after the first chunk -> tpot = 900 / 3.
        assert [list(sp.tpot_ns) for sp in series] == [[300.0, 300.0], [300.0]]
        assert [sp.out_tokens for sp in series] == [6, 3]

    @pytest.mark.asyncio
    async def test_the_row_is_gone_by_the_time_the_token_count_arrives(self, tmp_path):
        """Why the super-pass is read at fire time and carried in the closure.

        The count only arrives at the drain flush. By then ``set_field`` has
        dropped the sample from the table, so nothing at flush time could look
        its super-pass up again -- and until the flush runs, nothing is
        attributed at all.
        """
        records = [
            session_event(SessionEventType.STARTED, ts=0),
            phase_start(),
            session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=10),
            *_sample("a", 100, 10, 900, ["c ", "x y z"]),
        ]
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                "ss_noflush",
                tokenizer=MockBatchTokenizer(),
                steady_state_profile="concurrency",
                streaming=True,
            )
            try:
                await agg.process(records)
                assert agg.table.get_row("a") is None
                assert list(agg._collector.series()[0].tpot_ns) == []
                await agg.token_queue.flush_remaining(None)
                assert list(agg._collector.series()[0].tpot_ns) == [300.0]
            finally:
                agg.close()


@pytest.mark.unit
class TestCollectionGate:
    @pytest.mark.asyncio
    async def test_disabled_by_default(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(ctx, loop, "ss_off")
            try:
                await agg.process(_event_stream())
                assert agg._collector is None
            finally:
                agg.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("profile", ["agentic", "offline"])
    async def test_an_unsupported_profile_collects_nothing(self, tmp_path, profile):
        """The aggregator refuses a profile its collector does not describe.

        Agentic's steady-state metric is per-trajectory NATL, not a super-pass
        window; offline's min-duration gate collapses when every query is issued
        at t=0. Refused here as well as by the parent's gate.
        """
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                f"ss_unsupported_{profile}",
                tokenizer=MockBatchTokenizer(),
                steady_state_profile=profile,
            )
            try:
                await agg.process(_event_stream())
                assert agg._collector is None
                assert agg._steady_state_verdict(n_pending=0) is None
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_a_warmup_announcement_never_sizes_the_series(self, tmp_path):
        """Warmup announces first, carrying its own dataset size.

        The series-is-empty guard alone would let that size through until the
        performance phase overwrote it; only the phase-type check keeps the
        warmup dataset out of the bucket size entirely.
        """
        records = [
            session_event(SessionEventType.STARTED, ts=0),
            phase_start(num_turns=99, phase_type="warmup"),
        ]
        collector = await _collect(tmp_path, records, "ss_warmup_size")
        assert collector.superpass_size == 0

    @pytest.mark.asyncio
    async def test_bucket_size_comes_from_the_performance_phase(self, tmp_path):
        """A later phase cannot re-size a series already being bucketed."""
        records = [
            session_event(SessionEventType.STARTED, ts=0),
            phase_start(num_turns=3),
            session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=10),
            *_sample("a", 100, 10, 900, ["c ", "x"]),
            phase_start(num_turns=77, phase_type="accuracy"),
        ]
        collector = await _collect(tmp_path, records, "ss_size")
        assert collector.superpass_size == 3


@pytest.mark.unit
class TestVerdictOnTheFinalSnapshot:
    """The verdict rides the terminal snapshot, and only when it is deserved."""

    @staticmethod
    async def _finalize(tmp_path, socket_name, *, extra=()):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, publisher = make_aggregator(
                ctx,
                loop,
                socket_name,
                tokenizer=MockBatchTokenizer(),
                steady_state_profile="concurrency",
            )
            try:
                await agg.process(
                    [
                        *_event_stream(),
                        *extra,
                        session_event(SessionEventType.ENDED, ts=10**9),
                    ]
                )
            finally:
                agg.close()
            return publisher.publish_final.call_args.kwargs

    @pytest.mark.asyncio
    async def test_verdict_reaches_publish_final(self, tmp_path):
        kwargs = await self._finalize(tmp_path, "ss_final")
        verdict = kwargs["steady_state"]
        assert verdict is not None
        assert verdict["superpass_size"] == SUPERPASS
        assert verdict["n_super_passes"] > 0

    @pytest.mark.asyncio
    async def test_no_verdict_for_an_interrupted_run(self, tmp_path):
        kwargs = await self._finalize(
            tmp_path,
            "ss_final_int",
            extra=[session_event(SessionEventType.INTERRUPTED, ts=10**9 - 1)],
        )
        assert kwargs["steady_state"] is None

    @pytest.mark.asyncio
    async def test_no_verdict_when_tokenizations_did_not_drain(self, tmp_path):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                "ss_final_pending",
                tokenizer=MockBatchTokenizer(),
                steady_state_profile="concurrency",
            )
            try:
                await agg.process(_event_stream())
                assert agg._steady_state_verdict(n_pending=3) is None
                assert agg._steady_state_verdict(n_pending=0) is not None
            finally:
                agg.close()

    @pytest.mark.asyncio
    async def test_a_detector_failure_never_costs_the_run_its_snapshot(
        self, tmp_path, monkeypatch
    ):
        loop = asyncio.get_event_loop()
        with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
            agg, _, _ = make_aggregator(
                ctx,
                loop,
                "ss_final_boom",
                tokenizer=MockBatchTokenizer(),
                steady_state_profile="concurrency",
            )
            try:
                await agg.process(_event_stream())
                monkeypatch.setattr(
                    "inference_endpoint.async_utils.services.metrics_aggregator"
                    ".aggregator.compute_steady_state_metrics",
                    lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
                )
                assert agg._steady_state_verdict(n_pending=0) is None
            finally:
                agg.close()
