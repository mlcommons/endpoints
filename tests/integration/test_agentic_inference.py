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

"""Integration tests for agentic inference benchmarking end-to-end.

Validates that AgenticInferenceDataset + AgenticInferenceStrategy + BenchmarkSession work
correctly together against a real HTTP echo server.

Tests cover:
  1. Pre-built messages are issued as-is; each turn is issued sequentially per
     conversation.
  2. Multiple concurrent conversations complete successfully.
  3. Turn ordering: turn N+1 is never issued before turn N completes.
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
    AgenticInferenceConfig,
    LoadPattern,
    LoadPatternType,
)
from inference_endpoint.core.record import EventRecord, SampleEventType
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    AgenticInferenceDataset,
)
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator.agentic_inference_strategy import (
    AgenticInferenceStrategy,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.session import (
    BenchmarkSession,
    EventPublisher,
    PhaseConfig,
    PhaseType,
)
from inference_endpoint.testing.echo_server import EchoServer


class _NoOpPublisher:
    def publish(self, event_record: EventRecord) -> None:
        pass

    def flush(self) -> None:
        pass


class _RecordingPublisher:
    def __init__(self, records: list[EventRecord]):
        self.records = records

    def publish(self, event_record: EventRecord) -> None:
        self.records.append(event_record)

    def flush(self) -> None:
        pass


def _make_dataset(rows: list[dict]) -> AgenticInferenceDataset:
    """Build a loaded AgenticInferenceDataset from a list of row dicts."""
    df = pd.DataFrame(rows)
    ds = AgenticInferenceDataset(dataframe=df)
    ds.load()
    return ds


def _make_strategy(
    ds: AgenticInferenceDataset,
    target_concurrency: int | None = None,
    inject_tool_delay: bool = False,
) -> AgenticInferenceStrategy:
    agentic_cfg = AgenticInferenceConfig(
        enable_salt=False,
        turn_timeout_s=10.0,
        inject_tool_delay=inject_tool_delay,
    )
    assert ds.conversation_metadata is not None
    return AgenticInferenceStrategy(
        conversation_manager=ConversationManager(),
        dataset_metadata=ds.conversation_metadata,
        agentic_inference_config=agentic_cfg,
        target_concurrency=target_concurrency,
    )


async def _run_session(
    server_url: str,
    ds: AgenticInferenceDataset,
    strategy: AgenticInferenceStrategy,
    responses_out: dict,
    event_records_out: list[EventRecord] | None = None,
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
        endpoint_urls=[urljoin(server_url.rstrip("/") + "/", "v1/chat/completions")],
        warmup_connections=0,
        num_workers=1,
        max_connections=4,
        min_required_connections=0,
        worker_initialization_timeout=120.0,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)
    publisher: EventPublisher
    if event_records_out is not None:
        publisher = _RecordingPublisher(event_records_out)
    else:
        publisher = _NoOpPublisher()

    try:
        session = BenchmarkSession(
            issuer=issuer,
            event_publisher=publisher,
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
                pass  # request body may not be JSON
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
        strategy = _make_strategy(ds)
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
async def test_pre_built_messages_grow_each_turn(echo_server):
    """Pre-built messages array grows with each dataset turn."""
    received_payloads: list[dict] = []

    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception:
                pass  # request body may not be JSON
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
        strategy = _make_strategy(ds)
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
    agentic_cfg = AgenticInferenceConfig(enable_salt=False, turn_timeout_s=10.0)
    conv_manager = ConversationManager()
    strategy = AgenticInferenceStrategy(
        conversation_manager=conv_manager,
        dataset_metadata=ds.conversation_metadata,
        agentic_inference_config=agentic_cfg,
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
        endpoint_urls=[
            urljoin(echo_server.url.rstrip("/") + "/", "v1/chat/completions")
        ],
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
async def test_concurrent_conversations_stress(echo_server):
    """12 conversations × 3 turns each complete with correct counts."""
    num_convs = 12
    turns_per_conv = 3  # 2 user turns + 1 assistant turn each
    rows = []
    for i in range(num_convs):
        conv_id = f"stress_conv_{i}"
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 1,
                "role": "user",
                "content": f"Q1-{i}",
            }
        )
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 2,
                "role": "assistant",
                "content": f"A1-{i}",
            }
        )
        rows.append(
            {
                "conversation_id": conv_id,
                "turn": 3,
                "role": "user",
                "content": f"Q2-{i}",
            }
        )

    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)
    responses: dict = {}

    count = await _run_session(echo_server.url, ds, strategy, responses)

    # 12 conversations × 2 client turns each = 24
    expected_client_turns = num_convs * (turns_per_conv - 1)  # 24
    assert count == expected_client_turns
    assert len(responses) == expected_client_turns


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agentic_inference_active_conversations_respects_target_concurrency(
    echo_server,
):
    num_convs = 20
    rows = []
    for i in range(num_convs):
        conv_id = f"cap_conv_{i}"
        rows += [
            {
                "conversation_id": conv_id,
                "turn": 1,
                "role": "user",
                "content": f"Q1-{i}",
            },
            {
                "conversation_id": conv_id,
                "turn": 2,
                "role": "assistant",
                "content": f"A1-{i}",
            },
            {
                "conversation_id": conv_id,
                "turn": 3,
                "role": "user",
                "content": f"Q2-{i}",
            },
        ]

    ds = _make_dataset(rows)
    strategy = _make_strategy(ds, target_concurrency=4)
    responses: dict = {}

    observed_max: list[int] = []
    orig_on_sample_complete = strategy.on_sample_complete

    def tracked_on_sample_complete(result) -> None:
        observed_max.append(len(strategy._active_iters))
        orig_on_sample_complete(result)

    strategy.on_sample_complete = tracked_on_sample_complete

    await _run_session(echo_server.url, ds, strategy, responses)

    assert len(responses) == num_convs * 2  # 2 client turns per conversation
    assert max(observed_max, default=0) <= 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agentic_inference_pipeline_exception_propagates(echo_server):
    rows = [
        {"conversation_id": "err_c1", "turn": 1, "role": "user", "content": "Q1"},
        {"conversation_id": "err_c1", "turn": 2, "role": "assistant", "content": "A1"},
        {"conversation_id": "err_c1", "turn": 3, "role": "user", "content": "Q2"},
    ]
    ds = _make_dataset(rows)
    strategy = _make_strategy(ds)

    call_count = 0
    orig_issue_next_turn = strategy._issue_next_turn

    def failing_issue_next_turn(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("injected pipeline error")
        return orig_issue_next_turn(*args, **kwargs)

    strategy._issue_next_turn = failing_issue_next_turn

    loop = asyncio.get_running_loop()
    http_config = HTTPClientConfig(
        endpoint_urls=[
            urljoin(echo_server.url.rstrip("/") + "/", "v1/chat/completions")
        ],
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
            on_sample_complete=strategy.on_sample_complete,
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
        phase = PhaseConfig("perf", rt, ds, PhaseType.PERFORMANCE, strategy=strategy)

        with pytest.raises(RuntimeError, match="injected pipeline error"):
            await asyncio.wait_for(session.run([phase]), timeout=30.0)

        assert strategy._inflight == {}
    finally:
        await http_client.shutdown_async()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tools_field_forwarded_to_endpoint(echo_server):
    """The 'tools' array from the dataset reaches the endpoint in every request payload."""
    received_payloads: list[dict] = []

    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception:
                pass  # request body may not be JSON
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
        strategy = _make_strategy(ds)
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


def _tool_use_rows_with_delays(
    tool_row_delays: dict[int, float | None],
) -> list[dict]:
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
    rows: list[dict] = [
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
    for turn_idx, delay in tool_row_delays.items():
        if delay is None:
            continue
        for r in rows:
            if r["turn"] == turn_idx and r["role"] == "tool":
                r["delay_seconds"] = delay
    return rows


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delay_seconds_end_to_end(echo_server):
    baseline_rows = _tool_use_rows_with_delays({3: 1.5})

    ds_off = _make_dataset(baseline_rows)
    strat_off = _make_strategy(ds_off, inject_tool_delay=False)
    events_off: list[EventRecord] = []
    count_off = await _run_session(echo_server.url, ds_off, strat_off, {}, events_off)

    ds_on = _make_dataset(baseline_rows)
    strat_on = _make_strategy(ds_on, inject_tool_delay=True)
    events_on: list[EventRecord] = []
    count_on = await _run_session(echo_server.url, ds_on, strat_on, {}, events_on)

    assert count_off == count_on == 3

    def issue_delta_s(events: list[EventRecord]) -> float:
        issued_by_turn = {
            event.turn: event.timestamp_ns
            for event in events
            if event.event_type is SampleEventType.ISSUED
        }
        return (issued_by_turn[3] - issued_by_turn[1]) / 1e9

    delta = issue_delta_s(events_on) - issue_delta_s(events_off)
    assert (
        1.0 <= delta <= 3.0
    ), f"delay-on minus delay-off should be ~1.5s, got delta={delta:.3f}s"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delay_does_not_leak_to_endpoint_payload():
    received_payloads: list[dict] = []
    payload_capture_errors: list[str] = []

    class CapturingEchoServer(EchoServer):
        async def _handle_echo_chat_completions_request(self, request):
            try:
                payload = await request.json()
                received_payloads.append(payload)
            except Exception as exc:
                payload_capture_errors.append(repr(exc))
            return await super()._handle_echo_chat_completions_request(request)

    server = CapturingEchoServer(port=0)
    server.start()
    try:
        rows = _tool_use_rows_with_delays({3: 0.05})
        ds = _make_dataset(rows)
        strategy = _make_strategy(ds, inject_tool_delay=True)
        responses: dict = {}

        await _run_session(server.url, ds, strategy, responses)
    finally:
        server.stop()

    assert not payload_capture_errors
    assert received_payloads, "echo server captured no payloads"
    for payload in received_payloads:
        assert (
            "delay_seconds" not in payload
        ), f"delay_seconds leaked into request payload: keys={list(payload.keys())}"
