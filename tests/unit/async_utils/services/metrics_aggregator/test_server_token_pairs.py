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

"""Server token counts are invariant across inline and structured responses.

The same completion arrives either with reasoning and tool calls inline in
``content`` (``<think>...</think>``, ``<tool_call>...``) or split into
structured ``reasoning_content`` / ``tool_calls`` fields. The server reports
one usage count for both, so the server-basis ISL, OSL and TPOT must match
across each pair while the tokenizer-basis OSL follows the representation it
was given. Both hold whichever basis is primary.
"""

import asyncio
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.registry import (
    MetricsRegistry,
)
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    BatchTokenizer,
)
from inference_endpoint.async_utils.services.metrics_aggregator.tokenization import (
    MessageInput,
    TextInput,
    extract_tokenization_input,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.core.record import (
    EventRecord,
    EventRecordCodec,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput, TokenCountSource, basis_key
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter
from inference_endpoint.openai.types import SSEChoice, SSEDelta

from .conftest import (
    make_aggregator,
    sample_event,
    session_event,
    snapshot_counters,
    snapshot_series_count,
    snapshot_series_total,
)

_TOKENIZERS = Path(__file__).resolve().parents[4] / "assets/tokenizers"
_PROMPT, _COMPLETION = 5, 6  # identical server usage for both members of a pair
_TOOL = {
    "index": 0,
    "id": "call-0",
    "type": "function",
    "function": {"name": "grep", "arguments": '{"q": "x"}'},
}
_SPLIT = [{"reasoning_content": "c"}, {"reasoning_content": "d"}, {"content": "ab"}]
# inline deltas, structured deltas, client OSL (inline, structured),
# client TPOT tail tokens (inline, structured)
_PAIRS = [
    pytest.param(
        [{"content": "ab"}, {"content": "cd"}],
        [{"content": "ab"}, {"content": "cd"}],
        (4, 4),
        (2, 2),
        id="content",
    ),
    pytest.param(
        [{"content": "<think>cd"}, {"content": "ef</think>"}],
        [{"reasoning_content": "cd"}, {"reasoning_content": "ef"}],
        (19, 4),
        (10, 2),
        id="reasoning",
    ),
    pytest.param(
        [{"content": "<think>cd"}, {"content": "</think>ab"}],
        [{"reasoning_content": "cd"}, {"content": "ab"}],
        (19, 4),
        (10, 2),
        id="both",
    ),
    pytest.param(
        [
            {
                "content": '<tool_call>{"name": "grep", "arguments": {"q": "x"}}</tool_call>'
            }
        ],
        [{"tool_calls": [_TOOL]}],
        (64, 39),
        (0, 0),
        id="tool",
    ),
    pytest.param(
        [
            {"content": "<think>"},
            {"content": "c"},
            {"content": "d</th"},
            {"content": "ink>ab"},
        ],
        _SPLIT,
        (19, 4),
        (12, 3),
        id="split",
    ),
    pytest.param(
        [{"content": "<th"}, {"content": "ink>cd</think>ab"}],
        _SPLIT,
        (19, 4),
        (16, 3),
        id="first",
    ),
]


@pytest.fixture(scope="module")
def bare_tokenizer() -> Iterator[BatchTokenizer]:
    with BatchTokenizer(
        str(_TOKENIZERS / "char_chat_bare_reasoning"), n_workers=0, live_workers=1
    ) as tok:
        yield tok


@pytest.fixture
def socket_dir() -> Iterator[str]:
    # macOS tmp_path breaks the 103-char IPC socket path limit.
    with tempfile.TemporaryDirectory(prefix="pair") as d:
        yield d


def _output(deltas: list[dict[str, Any]]) -> TextModelOutput:
    acc = OpenAISSEAccumulator("s1", stream_all_chunks=False)
    for delta in deltas:
        acc.add_chunk(SSEChoice(delta=SSEDelta(**delta)))
    out = acc.get_final_output().response_output
    assert isinstance(out, TextModelOutput)
    return out


async def _aggregate(
    tokenizer: BatchTokenizer,
    socket_dir: str,
    name: str,
    output: TextModelOutput,
    *,
    primary: TokenCountSource,
    prompt_tokens: int = _PROMPT,
    completion_tokens: int = _COMPLETION,
) -> MetricsRegistry:
    """Run one tracked sample (ISSUED 500, first chunk 1000, COMPLETE 11000)."""
    with ManagedZMQContext.scoped(socket_dir=socket_dir) as ctx:
        agg, registry, _ = make_aggregator(
            ctx,
            asyncio.get_running_loop(),
            name,
            tokenizer=tokenizer,
            token_count_source=primary,
        )
        try:
            await agg.process(
                [
                    session_event(SessionEventType.START_PERFORMANCE_TRACKING, ts=0),
                    sample_event(SampleEventType.ISSUED, "s1", ts=500),
                    sample_event(SampleEventType.RECV_FIRST, "s1", ts=1_000),
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        timestamp_ns=11_000,
                        sample_uuid="s1",
                        data=output,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                ]
            )
            assert agg.token_queue is not None  # a tokenizer was supplied
            await agg.token_queue.drain_all()
        finally:
            agg.close()
    return registry


@pytest.mark.unit
@pytest.mark.parametrize(
    ("asset", "structured"), [("char_chat", 19), ("char_chat_bare_reasoning", 4)]
)
def test_asset_frames_reasoning_without_inline_markers(asset, structured):
    with BatchTokenizer(str(_TOKENIZERS / asset), n_workers=0, live_workers=1) as tok:
        assert tok.count_sync(TextInput("<think>cd</think>ab")) == 19
        assert tok.count_sync(MessageInput("ab", "cd", None)) == structured


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("inline", "structured", "osl_client", "tail"), _PAIRS)
async def test_server_series_match_across_representations(
    bare_tokenizer, socket_dir, primary, inline, structured, osl_client, tail
):
    server = {
        base: basis_key(base, TokenCountSource.SERVER, primary=primary)
        for base in ("isl", "osl", "tpot_ns")
    }
    client = {
        base: basis_key(base, TokenCountSource.CLIENT, primary=primary)
        for base in ("osl", "tpot_ns")
    }
    for i, (deltas, expected, tail_tokens) in enumerate(
        zip((inline, structured), osl_client, tail, strict=True)
    ):
        output = _output(deltas)
        item = extract_tokenization_input(output)
        assert item is not None and bare_tokenizer.count_sync(item) == expected
        registry = await _aggregate(
            bare_tokenizer, socket_dir, f"pair{i}", output, primary=primary
        )
        assert snapshot_series_total(registry, server["osl"]) == _COMPLETION
        assert snapshot_series_total(registry, server["isl"]) == _PROMPT
        assert snapshot_series_total(registry, server["tpot_ns"]) == pytest.approx(
            10_000 / (_COMPLETION - 1)
        )
        assert snapshot_series_total(registry, client["osl"]) == expected
        if tail_tokens:
            assert snapshot_series_total(registry, client["tpot_ns"]) == pytest.approx(
                10_000 / tail_tokens
            )
        else:
            assert snapshot_series_count(registry, client["tpot_ns"]) == 0
        c = snapshot_counters(registry)
        assert (
            c["osl_paired_samples"],
            c["osl_paired_server_tokens"],
            c["osl_paired_client_tokens"],
        ) == (1, _COMPLETION, expected)
        assert c["tracked_samples_missing_usage"] == 0


_FRAMES = [
    {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "say <think> literally"},
            }
        ]
    },
    {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-0",
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": '{"pattern": "</think>"}',
                            },
                        }
                    ]
                },
            }
        ]
    },
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
    },
]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_literal_markers_survive_capture_verbatim(
    bare_tokenizer, socket_dir, primary
):
    buffer = (
        b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in _FRAMES)
        + b"data: [DONE]\n\n"
    )
    acc = OpenAISSEAccumulator("s1", stream_all_chunks=False)
    for chunk in OpenAIMsgspecAdapter.parse_sse_chunk(buffer, len(buffer)):
        acc.add_chunk(chunk)
    result = acc.get_final_output()
    codec = EventRecordCodec()
    _, payload = codec.encode(
        EventRecord(
            event_type=SampleEventType.COMPLETE,
            timestamp_ns=1,
            sample_uuid="s1",
            data=result.response_output,
        )
    )
    output = codec.decode(payload).data
    assert isinstance(output, TextModelOutput)
    content, reasoning, tool_calls = output.as_message_parts()
    assert (content, reasoning) == ("say <think> literally", None)
    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == '{"pattern": "</think>"}'
    md = result.metadata
    registry = await _aggregate(
        bare_tokenizer,
        socket_dir,
        "verbatim",
        output,
        primary=primary,
        prompt_tokens=md["prompt_tokens"],
        completion_tokens=md["completion_tokens"],
    )
    server_osl = basis_key("osl", TokenCountSource.SERVER, primary=primary)
    client_osl = basis_key("osl", TokenCountSource.CLIENT, primary=primary)
    assert snapshot_series_total(registry, server_osl) == 9
    # Markers are counted as ordinary characters.
    assert snapshot_series_total(registry, client_osl) == 73
