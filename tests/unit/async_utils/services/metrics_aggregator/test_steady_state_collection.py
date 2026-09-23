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

"""Steady-state collection inside the aggregator, against the standalone parse.

The aggregator rolls super-passes up from the events it already routes, reusing
the token counts its TPOT trigger already computed. The standalone detector
reconstructs the same rollups from the event log after the fact. These tests
pin the two producers together at the service level, using one set of
``EventRecord``s for both: the aggregator consumes them directly, and the same
records are written out through the event logger's encoder for the parse.
"""

from __future__ import annotations

import asyncio
import logging

import msgspec
import pytest
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput
from inference_endpoint.metrics import steady_state_diagnostics as ssd

from .conftest import MockBatchTokenizer, make_aggregator

SUPERPASS_SIZE = 4


def _records() -> list[EventRecord]:
    """A tracked run carrying the cases the two producers must agree on."""
    out: list[EventRecord] = [
        EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=500),
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING,
            timestamp_ns=1000,
        ),
    ]
    ts = 1000
    for i in range(12):
        uuid = f"s{i}"
        ts += 100
        out.append(
            EventRecord(
                event_type=SampleEventType.ISSUED,
                timestamp_ns=ts,
                sample_uuid=uuid,
            )
        )
        out.append(
            EventRecord(
                event_type=SampleEventType.RECV_FIRST,
                timestamp_ns=ts + 200,
                sample_uuid=uuid,
                # Turn 1 is the cold first turn of an agentic trajectory and is
                # excluded from warm TTFT by both producers.
                turn=1 if i % 4 == 0 else 2,
            )
        )
        if i % 5 == 0:
            data = TextModelOutput(
                output=["a ", "b b "], reasoning=["think ", "think "]
            )
        else:
            data = TextModelOutput(output=["one ", "two three ", "four"])
        out.append(
            EventRecord(
                event_type=SampleEventType.COMPLETE,
                timestamp_ns=ts + 900,
                sample_uuid=uuid,
                data=data,
            )
        )
    out.append(
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING,
            timestamp_ns=ts + 2000,
        )
    )
    return out


def _write_log(tmp_path, records: list[EventRecord]) -> str:
    """Serialize records exactly as the event logger's JSONL writer does."""
    encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)
    path = tmp_path / "events.jsonl"
    with path.open("w") as fh:
        for record in records:
            fh.write(encoder.encode(record).decode("utf-8") + "\n")
    return str(path)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_series_matches_the_event_log_parse(tmp_path):
    """The invariant the redesign rests on, at the service level: routing a run
    through the aggregator and parsing the log of that same run must produce
    the same super-pass series."""
    records = _records()
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            "agg_steady_state_series",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
        )
        try:
            await agg.process(records)
            await agg.token_queue.drain_all()
            live = agg.steady_state_series()
        finally:
            agg.close()

    from_log = ssd.build_super_pass_series(
        _write_log(tmp_path, records),
        SUPERPASS_SIZE,
        MockBatchTokenizer().count_sync_batch,
    )

    assert live == from_log


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aggregator_collects_nothing_unless_asked(tmp_path):
    """Steady-state collection is opt-in: without a super-pass size the
    aggregator must not accumulate rollups at all."""
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx, loop, "agg_no_steady_state", tokenizer=MockBatchTokenizer()
        )
        try:
            await agg.process(_records())
            assert agg.steady_state_series() is None
        finally:
            agg.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_untracked_samples_are_not_collected(tmp_path):
    """Samples issued outside a performance-tracking window get no bucket --
    the aggregator sees every sample, tracked or not."""
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            "agg_untracked_steady_state",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
        )
        try:
            await agg.process(
                [
                    EventRecord(
                        event_type=SampleEventType.ISSUED,
                        timestamp_ns=100,
                        sample_uuid="before",
                    ),
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        timestamp_ns=900,
                        sample_uuid="before",
                        data=TextModelOutput(output=["a ", "b"]),
                    ),
                ]
            )
            assert agg.steady_state_series() == []
        finally:
            agg.close()


async def _run_to_drain(tmp_path, socket_name, *, out_path=None, superpass_size=None):
    """Route a full tracked run, including ENDED, through the aggregator."""
    records = _records()
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, publisher = make_aggregator(
            ctx,
            loop,
            socket_name,
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=superpass_size,
            steady_state_out=out_path,
        )
        try:
            await agg.process(records)
            await agg.process(
                [EventRecord(event_type=SessionEventType.ENDED, timestamp_ns=99_999)]
            )
        finally:
            agg.close()
    return records, publisher


@pytest.mark.unit
@pytest.mark.asyncio
async def test_verdict_written_at_drain_matches_the_event_log_verdict(tmp_path):
    """The verdict the aggregator writes is the one the standalone detector
    reaches from the same run's event log -- equivalence carried through the
    algorithm and out to the artifact the report reads."""
    out = tmp_path / "steady_state.json"

    records, _ = await _run_to_drain(
        tmp_path, "agg_ss_drain", out_path=out, superpass_size=SUPERPASS_SIZE
    )

    written = msgspec.json.decode(out.read_bytes())
    from_log = ssd.analyse(
        ssd.build_super_pass_series(
            _write_log(tmp_path, records),
            SUPERPASS_SIZE,
            MockBatchTokenizer().count_sync_batch,
        ),
        superpass_size=SUPERPASS_SIZE,
    )

    assert written == msgspec.json.decode(msgspec.json.encode(from_log))
    assert (tmp_path / "steady_state.txt").read_text().startswith("super-passes:")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_verdict_written_when_collection_is_off(tmp_path, caplog):
    """A run that never opted in skips detection outright. It must not reach
    the detector and fall into the best-effort handler -- that would bury a
    real failure under a skip that looks the same from the outside."""
    out = tmp_path / "steady_state.json"

    with caplog.at_level(logging.ERROR):
        await _run_to_drain(tmp_path, "agg_ss_off", out_path=out)

    assert not out.exists()
    assert "steady-state detection failed" not in caplog.text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_verdict_write_still_publishes_the_final_snapshot(tmp_path):
    """Steady state is best-effort; the final snapshot is the report's primary
    source. A detector or IO failure must not cost the run its snapshot."""
    unwritable = tmp_path / "nonexistent-dir" / "steady_state.json"

    _, publisher = await _run_to_drain(
        tmp_path, "agg_ss_io_fail", out_path=unwritable, superpass_size=SUPERPASS_SIZE
    )

    assert not unwritable.exists()
    publisher.publish_final.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_retry_after_tracking_stops_is_ignored_by_both_producers(tmp_path):
    """A sample re-issued after STOP_PERFORMANCE_TRACKING is outside the
    measured window, and the log parse drops it. The aggregator must too:
    `MetricsTable.set_field` returns early for an untracked ISSUED without
    dropping the in-flight row, so the row is still there to be found. The
    disagreement lands on last_issue_ns -- the throughput denominator -- so
    the two producers would report different system TPS for one run."""
    records = [
        EventRecord(event_type=SessionEventType.STARTED, timestamp_ns=500),
        EventRecord(
            event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=1000
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=1100, sample_uuid="A"
        ),
        EventRecord(
            event_type=SampleEventType.RECV_FIRST, timestamp_ns=1200, sample_uuid="A"
        ),
        EventRecord(
            event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=5000
        ),
        EventRecord(
            event_type=SampleEventType.ISSUED, timestamp_ns=9000, sample_uuid="A"
        ),
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=9500,
            sample_uuid="A",
            data=TextModelOutput(output=["a ", "b b"]),
        ),
    ]
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            "agg_retry_after_stop",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
        )
        try:
            await agg.process(records)
            await agg.token_queue.drain_all()
            live = agg.steady_state_series()
        finally:
            agg.close()

    from_log = ssd.build_super_pass_series(
        _write_log(tmp_path, records),
        SUPERPASS_SIZE,
        MockBatchTokenizer().count_sync_batch,
    )

    assert live == from_log


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_verdict_is_written_for_an_interrupted_run(tmp_path):
    """The session publishes INTERRUPTED just before ENDED on an aborted run,
    so the aggregator already knows the series is truncated. Writing a verdict
    anyway leaves one on disk that only the parent's withdrawal removes -- and
    on a SIGKILL or an expired teardown grace the parent never gets there, so
    the artifact outlives the run it misdescribes."""
    out = tmp_path / "steady_state.json"
    records = _records()
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            "agg_ss_interrupted",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
            steady_state_out=out,
        )
        try:
            await agg.process(records)
            await agg.process(
                [
                    EventRecord(
                        event_type=SessionEventType.INTERRUPTED, timestamp_ns=99_000
                    ),
                    EventRecord(event_type=SessionEventType.ENDED, timestamp_ns=99_999),
                ]
            )
        finally:
            agg.close()

    assert not out.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_final_snapshot_lands_before_the_verdict(tmp_path):
    """final_snapshot.json is the Report's primary source; the verdict is
    best-effort. Analysis cost grows with the run, so doing it first puts the
    snapshot behind an unbounded delay -- and an abort during it costs the run
    its snapshot. The parent reads the verdict only after this process exits,
    so writing it afterwards is free."""
    out = tmp_path / "steady_state.json"
    seen: list[bool] = []
    records = _records()
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, publisher = make_aggregator(
            ctx,
            loop,
            "agg_ss_order",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
            steady_state_out=out,
        )

        async def record_order(*_args, **_kwargs):
            seen.append(out.exists())

        publisher.publish_final.side_effect = record_order
        try:
            await agg.process(records)
            await agg.process(
                [EventRecord(event_type=SessionEventType.ENDED, timestamp_ns=99_999)]
            )
        finally:
            agg.close()

    assert seen == [False], "verdict was written before the final snapshot"
    assert out.exists(), "verdict never written"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_verdict_is_judged_on_the_workload_profile(tmp_path):
    """The standalone CLI resolves CoV bounds and the warmup driver from the
    run's workload profile. The aggregator must use the same ones, or the
    verdict a run publishes and the verdict a hand re-run reaches are judged
    against different thresholds -- silently, since the two supported profiles
    happen to carry the module defaults today."""
    out = tmp_path / "steady_state.json"
    loop = asyncio.get_event_loop()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, _, _ = make_aggregator(
            ctx,
            loop,
            "agg_ss_profile",
            tokenizer=MockBatchTokenizer(),
            steady_state_superpass_size=SUPERPASS_SIZE,
            steady_state_out=out,
            steady_state_profile="agentic",
        )
        try:
            await agg.process(_records())
            await agg.process(
                [EventRecord(event_type=SessionEventType.ENDED, timestamp_ns=99_999)]
            )
        finally:
            agg.close()

    written = msgspec.json.decode(out.read_bytes())
    bounds = {
        bound
        for window in written["cov"].values()
        for metric in window.values()
        for bound in metric["passes"]
    }
    agentic = ssd.PROFILES["agentic"]
    assert bounds == {str(b) for b in agentic.cov_bounds}
