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

"""Real tokenization of accumulator output across the first response boundary."""

import asyncio
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    OslTrigger,
    SampleField,
    SampleRow,
    TpotTrigger,
)
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
    TokenBatchQueue,
)
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
)
from inference_endpoint.core.record import (
    EventRecord,
    EventRecordCodec,
    SampleEventType,
)
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.types import SSEChoice, SSEDelta

from .conftest import snapshot_series_count, snapshot_series_total

_TOKENIZER = Path(__file__).resolve().parents[4] / "assets/tokenizers/char_chat"
_TOOL = {
    "index": 0,
    "id": "call-0",
    "type": "function",
    "function": {"name": "f", "arguments": "{}"},
}
_FULL_TOOL = {key: value for key, value in _TOOL.items() if key != "index"}


@pytest.fixture(scope="module")
def response_tokenizer():
    with BatchTokenizer(str(_TOKENIZER), n_workers=0, live_workers=1) as tokenizer:
        yield tokenizer


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_all_chunks", [False, True])
@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize(
    "deltas, full_parts, tail_parts",
    [
        pytest.param(
            [("first", "think", None), ("answer", "more", None)],
            ("firstanswer", "thinkmore", None),
            ("answer", "more", None),
            id="mixed-first",
        ),
        pytest.param(
            [("", "", [_TOOL]), ("answer", "think", None)],
            ("answer", "think", (_FULL_TOOL,)),
            ("answer", "think", None),
            id="tools-before-text",
        ),
        pytest.param(
            [("first", "", None), ("answer", "think", [_TOOL])],
            ("firstanswer", "think", (_FULL_TOOL,)),
            ("answer", "think", (_FULL_TOOL,)),
            id="content-before-reasoning",
        ),
        pytest.param(
            [("answer", "think", [_TOOL])],
            ("answer", "think", (_FULL_TOOL,)),
            ("", None, None),
            id="single-mixed-delta",
        ),
    ],
)
async def test_response_metrics_use_full_output_and_actual_tail(
    response_tokenizer,
    deltas,
    full_parts,
    tail_parts,
    reasoning_field,
    stream_all_chunks,
):
    accumulator = OpenAISSEAccumulator("qid", stream_all_chunks)
    for content, reasoning, tools in deltas:
        accumulator.add_chunk(
            SSEChoice(
                delta=SSEDelta(
                    content=content, tool_calls=tools, **{reasoning_field: reasoning}
                )
            )
        )
    event = EventRecord(
        event_type=SampleEventType.COMPLETE,
        timestamp_ns=11000,
        sample_uuid="qid",
        data=accumulator.get_final_output().response_output,
    )
    # Exercise the same boundary transfer used by the worker/metrics processes.
    codec = EventRecordCodec()
    _, payload = codec.encode(event)
    event = codec.decode(payload)
    loop = asyncio.get_running_loop()
    full_count, tail_count = await response_tokenizer.count_batch_async(
        [MessageInput(*full_parts), MessageInput(*tail_parts)], loop
    )
    assert full_count > 0
    registry = MetricsRegistry()
    registry.register_series("osl", hdr_low=1, hdr_high=100000, dtype=int)
    registry.register_series("tpot_ns", hdr_low=1, hdr_high=100000000, dtype=float)
    queue = TokenBatchQueue(response_tokenizer, loop)
    row = SampleRow(sample_uuid="qid")
    pre_change = {SampleField.RECV_FIRST_NS: 1000}
    OslTrigger(registry, queue).fire(event, row, pre_change)
    TpotTrigger(registry, queue).fire(event, row, pre_change)
    await queue.drain_all()
    assert snapshot_series_total(registry, "osl") == full_count
    if any(tail_parts):
        assert tail_count > 0
        assert snapshot_series_count(registry, "tpot_ns") == 1
        assert snapshot_series_total(registry, "tpot_ns") == pytest.approx(
            10000 / tail_count
        )
    else:
        assert tail_count == 0
        assert snapshot_series_count(registry, "tpot_ns") == 0
