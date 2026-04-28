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

"""Integration tests for multi-turn benchmarking end-to-end.

Validates that MultiTurnDataset + MultiTurnStrategy + BenchmarkSession work
correctly together against a real HTTP echo server.

Tests cover:
  1. Dataset-history mode (use_dataset_history=True): pre-built messages are
     issued as-is; each turn is issued sequentially per conversation.
  2. Live-history mode (use_dataset_history=False): messages are built at
     runtime from ConversationManager.message_history; the injected messages
     grow with each turn.
  3. Multiple concurrent conversations complete successfully.
  4. Turn ordering: turn N+1 is never issued before turn N completes.
"""

import asyncio
import random
import time
from urllib.parse import urljoin

import pandas as pd
import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    LoadPattern,
    LoadPatternType,
    MultiTurnConfig,
)
from inference_endpoint.core.record import EventRecord
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.multi_turn_strategy import MultiTurnStrategy
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    PhaseConfig,
    PhaseType,
)
from inference_endpoint.testing.echo_server import EchoServer


class _NoOpPublisher:
    def publish(self, event_record: EventRecord) -> None:
        pass

    def flush(self) -> None:
        pass


def _make_dataset(rows: list[dict]) -> MultiTurnDataset:
    """Build a loaded MultiTurnDataset from a list of row dicts."""
    df = pd.DataFrame(rows)
    ds = MultiTurnDataset(dataframe=df)
    ds.load()
    return ds


def _make_strategy(
    ds: MultiTurnDataset,
    use_dataset_history: bool = True,
) -> MultiTurnStrategy:
    mt_cfg = MultiTurnConfig(
        turn_timeout_s=10.0,
        use_dataset_history=use_dataset_history,
    )
    return MultiTurnStrategy(
        conversation_manager=ConversationManager(),
        dataset_metadata=ds.conversation_metadata,
        multi_turn_config=mt_cfg,
    )


async def _run_session(
    server_url: str,
    ds: MultiTurnDataset,
    strategy: MultiTurnStrategy,
    responses_out: dict,
) -> int:
    """Wire up HTTPEndpointClient + BenchmarkSession and run one phase.

    Populates responses_out[query_id] = response_text for every completed turn.
    Returns issued_count.
    """
    loop = asyncio.get_running_loop()

    def on_complete(result: QueryResult) -> None:
        strategy.on_sample_complete(result)
        responses_out[result.id] = result.get_response_output_string()

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(server_url, "/v1/chat/completions")],
        warmup_connections=0,
        num_workers=2,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)

    try:
        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=_NoOpPublisher(),
            loop=loop,
            on_sample_complete=on_complete,
        )
        rt = RuntimeSettings(
            metrics.Throughput(1000),
            [metrics.Throughput(1000)],
            min_duration_ms=0,
            max_duration_ms=30_000,
            n_samples_from_dataset=ds.num_samples(),
            n_samples_to_issue=ds.num_samples(),
            min_sample_count=1,
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        phase = PhaseConfig(
            "perf",
            rt,
            ds,
            PhaseType.PERFORMANCE,
            strategy=strategy,
        )
        result = await asyncio.wait_for(session.run([phase]), timeout=30.0)
        return result.perf_results[0].issued_count
    finally:
        await http_client.shutdown_async()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_server():
    server = EchoServer(port=0)
    server.start()
    try:
        yield server
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_single_conversation_all_turns_issued(echo_server):
    """All turns of a single conversation are issued and completed."""
    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hello"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hi"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "Bye"},
    ]
    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)
    responses: dict = {}

    count = await _run_session(echo_server.url, ds, strategy, responses)

    # Two user turns (turns 1 and 3); turn 2 is assistant so not a client turn
    assert count == 2
    assert len(responses) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_conversations_all_issued(echo_server):
    """Multiple conversations complete independently and concurrently."""
    rows = []
    for conv_idx in range(3):
        conv_id = f"conv_{conv_idx}"
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 1,
                "role": "user",
                "content": f"Q1 {conv_idx}",
            }
        )
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 2,
                "role": "assistant",
                "content": f"A1 {conv_idx}",
            }
        )
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 3,
                "role": "user",
                "content": f"Q2 {conv_idx}",
            }
        )
    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)
    responses: dict = {}

    count = await _run_session(echo_server.url, ds, strategy, responses)

    # 3 conversations × 2 user turns each = 6
    assert count == 6
    assert len(responses) == 6


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dataset_history_messages_present(echo_server):
    """Dataset-history mode: each request contains the messages array from the dataset."""
    received_payloads: list[dict] = []

    # Override get_response to capture the incoming request body.
    # EchoServer._handle_echo_chat_completions_request parses it into
    # CreateChatCompletionRequest — we capture the raw JSON at the HTTP layer
    # by subclassing and overriding get_response (called with first user content).
    # Instead, use a custom echo server that logs the full payload.
    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception:
                pass
            return await super()._handle_echo_chat_completions_request(request)

    server = CapturingEchoServer(port=0)
    server.start()
    try:
        rows = [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "First question",
            },
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": "First answer",
            },
            {
                "conversation_id": "c1",
                "turn": 3,
                "role": "user",
                "content": "Second question",
            },
        ]
        ds = _make_dataset(rows)
        strategy = _make_strategy(ds, use_dataset_history=True)
        responses: dict = {}

        count = await _run_session(server.url, ds, strategy, responses)
        assert count == 2

        # Both requests must include a "messages" array
        assert len(received_payloads) == 2
        for payload in received_payloads:
            assert "messages" in payload
            assert len(payload["messages"]) >= 1

        # Turn 1 should have 1 user message; turn 3 should have 3 messages
        # (system? no system here — user, assistant, user)
        msg_counts = sorted(len(p["messages"]) for p in received_payloads)
        assert msg_counts == [1, 3]
    finally:
        server.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_history_messages_grow_each_turn(echo_server):
    """Live-history mode: messages array grows with each completed turn."""
    received_payloads: list[dict] = []

    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception:
                pass
            return await super()._handle_echo_chat_completions_request(request)

    server = CapturingEchoServer(port=0)
    server.start()
    try:
        rows = [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Turn one"},
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": "Answer one",
            },
            {"conversation_id": "c1", "turn": 3, "role": "user", "content": "Turn two"},
        ]
        ds = _make_dataset(rows)
        strategy = _make_strategy(ds, use_dataset_history=False)
        responses: dict = {}

        count = await _run_session(server.url, ds, strategy, responses)
        assert count == 2

        assert len(received_payloads) == 2
        msg_counts = sorted(len(p["messages"]) for p in received_payloads)
        # Turn 1: [user msg] = 1; Turn 3: [user, assistant, user] = 3
        assert msg_counts == [1, 3]
    finally:
        server.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_ordering_enforced_end_to_end(echo_server):
    """Turn N+1 is issued after Turn N's response arrives, verified by timestamps."""
    complete_times: dict[str, float] = {}

    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "First"},
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "content": "Response",
        },
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "Second"},
    ]
    ds = _make_dataset(rows)
    mt_cfg = MultiTurnConfig(turn_timeout_s=10.0, use_dataset_history=True)
    conv_manager = ConversationManager()
    strategy = MultiTurnStrategy(
        conversation_manager=conv_manager,
        dataset_metadata=ds.conversation_metadata,
        multi_turn_config=mt_cfg,
    )

    # Wrap on_sample_complete to record completion timestamps
    orig_on_sample_complete = strategy.on_sample_complete

    def tracked_on_sample_complete(result: QueryResult) -> None:
        # Map query_id → sample_index via uuid_to_index (set after session runs)
        complete_times[result.id] = time.monotonic()
        orig_on_sample_complete(result)

    strategy.on_sample_complete = tracked_on_sample_complete

    loop = asyncio.get_running_loop()
    responses: dict[str, str] = {}

    http_config = HTTPClientConfig(
        endpoint_urls=[urljoin(echo_server.url, "/v1/chat/completions")],
        warmup_connections=0,
        num_workers=1,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)

    rt = RuntimeSettings(
        metrics.Throughput(1000),
        [metrics.Throughput(1000)],
        min_duration_ms=0,
        max_duration_ms=30_000,
        n_samples_from_dataset=ds.num_samples(),
        n_samples_to_issue=ds.num_samples(),
        min_sample_count=1,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
    )

    try:

        def on_complete(result: QueryResult) -> None:
            tracked_on_sample_complete(result)
            responses[result.id] = result.get_response_output_string()

        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=_NoOpPublisher(),
            loop=loop,
            on_sample_complete=on_complete,
        )
        phase = PhaseConfig("perf", rt, ds, PhaseType.PERFORMANCE, strategy=strategy)
        result = await asyncio.wait_for(session.run([phase]), timeout=30.0)
    finally:
        await http_client.shutdown_async()

    assert result.perf_results[0].issued_count == 2

    # Build query_id → sample_index from session result
    uuid_to_index = result.perf_results[0].uuid_to_index
    index_to_query = {v: k for k, v in uuid_to_index.items()}

    # Sample 0 = turn 1, sample 1 = turn 3
    q_turn1 = index_to_query[0]
    q_turn3 = index_to_query[1]

    # Turn 3 must complete after turn 1 completes
    assert complete_times[q_turn3] >= complete_times[q_turn1]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_use_conversation_all_turns_issued(echo_server):
    """Tool-use conversation: all client turns (user + tool) are issued and completed."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "test"}'},
        }
    ]
    tool_results = [{"tool_call_id": "call_1", "content": "search result"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            },
        }
    ]

    rows = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "Find something",
            "tools": tools,
        },
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "tool",
            "tool_results": tool_results,
            "tools": tools,
        },
        {
            "conversation_id": "c1",
            "turn": 4,
            "role": "assistant",
            "content": "Here is the result",
        },
        {"conversation_id": "c1", "turn": 5, "role": "user", "content": "Thanks"},
    ]
    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)
    responses: dict = {}

    count = await _run_session(echo_server.url, ds, strategy, responses)

    # Client turns: turn 1 (user) + turn 3 (tool) + turn 5 (user) = 3
    assert count == 3
    assert len(responses) == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_ending_with_tool_row(echo_server):
    """Conversation ending with a tool row completes normally (matches agentic_coding dataset pattern)."""
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "write_file", "arguments": '{"path": "out.py"}'},
        }
    ]
    tool_results = [{"tool_call_id": "call_1", "content": "file written"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    rows = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "Write a file",
            "tools": tools,
        },
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        },
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "tool",
            "tool_results": tool_results,
            "tools": tools,
        },
    ]
    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)
    responses: dict = {}

    count = await _run_session(echo_server.url, ds, strategy, responses)

    # Client turns: turn 1 (user) + turn 3 (tool) = 2
    assert count == 2
    assert len(responses) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_field_forwarded_to_endpoint(echo_server):
    """The 'tools' array from the dataset reaches the endpoint in every request payload.

    TODO: Add a tool-call-aware server that returns dynamic tool_call_ids to
    validate live-history mode with real tool_call_id round-tripping.
    """
    received_payloads: list[dict] = []

    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception:
                pass
            return await super()._handle_echo_chat_completions_request(request)

    server = CapturingEchoServer(port=0)
    server.start()
    try:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "hello"}'},
            }
        ]
        tool_results = [{"tool_call_id": "call_1", "content": "result"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]

        rows = [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "Search for hello",
                "tools": tools,
            },
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            {
                "conversation_id": "c1",
                "turn": 3,
                "role": "tool",
                "tool_results": tool_results,
                "tools": tools,
            },
        ]
        ds = _make_dataset(rows)
        strategy = _make_strategy(ds, use_dataset_history=True)
        responses: dict = {}

        count = await _run_session(server.url, ds, strategy, responses)
        assert count == 2

        assert len(received_payloads) == 2
        for payload in received_payloads:
            assert "tools" in payload
            assert len(payload["tools"]) == 1
            assert payload["tools"][0]["function"]["name"] == "search"
    finally:
        server.stop()
