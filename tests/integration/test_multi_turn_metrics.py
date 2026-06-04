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

"""Integration tests: ISL precompute and ISL/OSL/TPOT aggregator metrics for multi-turn runs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pandas as pd
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.aggregator import (
    MetricsAggregatorService,
)
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricSeriesKey,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    SeriesStat,
    SessionState,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.commands.benchmark.execute import _precompute_isl_for_multi_turn
from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import PromptData, TextModelOutput
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset


class _MockTokenizePool:
    async def token_count_async(
        self, text: str, _loop: asyncio.AbstractEventLoop
    ) -> int:
        return len(text.split())

    async def token_count_message_async(
        self,
        content: str,
        reasoning: str | None,
        tool_calls,
        _loop: asyncio.AbstractEventLoop,
    ) -> int:
        tool_calls_str = (
            msgspec.json.encode(list(tool_calls)).decode() if tool_calls else ""
        )
        combined = (content or "") + " " + (reasoning or "") + " " + tool_calls_str
        return len(combined.split())

    def close(self) -> None:
        pass


def _make_aggregator_with_mock_publisher(
    zmq_ctx: ManagedZMQContext,
    loop: asyncio.AbstractEventLoop,
    socket_name: str,
    shutdown_event: asyncio.Event,
    tokenize_pool=None,
) -> tuple[MetricsAggregatorService, MetricsRegistry]:
    """Build an aggregator with a mocked publisher (no ZMQ / disk I/O)."""
    registry = MetricsRegistry()
    publisher = MagicMock()
    publisher.publish_final = AsyncMock()
    publisher.aclose = AsyncMock()
    agg = MetricsAggregatorService(
        socket_name,
        zmq_ctx,
        loop,
        registry=registry,
        publisher=publisher,
        publish_interval_s=1.0,
        sig_figs=3,
        n_histogram_buckets=10,
        tokenize_pool=tokenize_pool,
        streaming=True,
        shutdown_event=shutdown_event,
    )
    return agg, registry


def _session_event(ev_type: SessionEventType, ts: int = 0) -> EventRecord:
    return EventRecord(event_type=ev_type, timestamp_ns=ts)


def _sample_event(
    ev_type: SampleEventType,
    uuid: str,
    ts: int = 0,
    data=None,
    conversation_id: str = "",
    turn: int | None = None,
) -> EventRecord:
    return EventRecord(
        event_type=ev_type,
        timestamp_ns=ts,
        sample_uuid=uuid,
        data=data,
        conversation_id=conversation_id,
        turn=turn,
    )


def _snapshot_series_count(registry: MetricsRegistry, name: str) -> int:
    snap = registry.build_snapshot(state=SessionState.LIVE, n_pending_tasks=0)
    for m in snap.metrics:
        if isinstance(m, SeriesStat) and m.name == name:
            return m.count
    return 0


@pytest.mark.integration
def test_multi_turn_isl_uses_precomputed_token_count():
    """_precompute_isl_for_multi_turn stores input_tokens on each sample dict."""
    transformers = pytest.importorskip("transformers")

    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hello there"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hi"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "How are you?"},
    ]
    ds = MultiTurnDataset(pd.DataFrame(rows))
    ds.load()

    tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        pytest.skip(f"Tokenizer '{tokenizer_name}' not available in this environment")

    _precompute_isl_for_multi_turn(ds, tokenizer_name)

    for i, sample in enumerate(ds.data or []):
        assert (
            "input_tokens" in sample
        ), f"Sample {i} missing input_tokens after precompute"
        assert isinstance(
            sample["input_tokens"], list
        ), f"Sample {i} input_tokens not a list"
        assert len(sample["input_tokens"]) > 0, f"Sample {i} input_tokens is empty"


_TOOL_CALLS = (
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q":"hello"}'},
    },
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_chunk", "later_chunk", "complete_data"),
    [
        (
            TextModelOutput(output=("chunk1",)),
            TextModelOutput(output=("chunk2",)),
            TextModelOutput(output=("chunk1", "chunk2", "chunk3")),
        ),
        (
            TextModelOutput(output=(), tool_calls=_TOOL_CALLS),
            TextModelOutput(output=(), tool_calls=_TOOL_CALLS),
            TextModelOutput(output=(), tool_calls=_TOOL_CALLS),
        ),
    ],
    ids=["text", "tool_calls_only"],
)
async def test_multi_turn_aggregator_records_metrics_streaming(
    tmp_path: Path,
    first_chunk: TextModelOutput,
    later_chunk: TextModelOutput,
    complete_data: TextModelOutput,
):
    """TTFT/ISL/OSL/TPOT fire for streamed multi-turn turns (text and tool-call payloads)."""
    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()
    with ManagedZMQContext.scoped(socket_dir=str(tmp_path)) as ctx:
        agg, registry = _make_aggregator_with_mock_publisher(
            ctx,
            loop,
            "test_mt_streaming_metrics",
            shutdown_event,
            tokenize_pool=_MockTokenizePool(),
        )
        try:
            t = 0

            def ts() -> int:
                nonlocal t
                t += 1_000_000
                return t

            uuid = "mt-turn-1"
            events = [
                _session_event(SessionEventType.STARTED, ts=ts()),
                _session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=ts()),
                _sample_event(
                    SampleEventType.ISSUED,
                    uuid,
                    ts=ts(),
                    data=PromptData(token_ids=(1, 2, 3, 4, 5)),
                    conversation_id="c1",
                    turn=1,
                ),
                _sample_event(
                    SampleEventType.RECV_FIRST,
                    uuid,
                    ts=ts(),
                    data=first_chunk,
                    conversation_id="c1",
                    turn=1,
                ),
                _sample_event(
                    SampleEventType.RECV_NON_FIRST,
                    uuid,
                    ts=ts(),
                    data=later_chunk,
                    conversation_id="c1",
                    turn=1,
                ),
                _sample_event(
                    SampleEventType.COMPLETE,
                    uuid,
                    ts=ts(),
                    data=complete_data,
                    conversation_id="c1",
                    turn=1,
                ),
                _session_event(SessionEventType.STOP_PERFORMANCE_TRACKING, ts=ts()),
                _session_event(SessionEventType.ENDED, ts=ts()),
            ]
            await agg.process(events)

            # OSL/TPOT fire via async tokenization tasks; poll until they land.
            for _ in range(30):
                if _snapshot_series_count(registry, MetricSeriesKey.OSL.value) > 0:
                    break
                await asyncio.sleep(0.05)

            for key in (
                MetricSeriesKey.ISL,
                MetricSeriesKey.TTFT_NS,
                MetricSeriesKey.OSL,
                MetricSeriesKey.TPOT_NS,
            ):
                assert (
                    _snapshot_series_count(registry, key.value) > 0
                ), f"{key.value} must be recorded"
        finally:
            agg.close()
