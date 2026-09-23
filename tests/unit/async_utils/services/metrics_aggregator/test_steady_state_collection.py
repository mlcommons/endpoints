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
