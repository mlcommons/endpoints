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
correctly together against a real HTTP echo server (echo tests) and a live
model endpoint (live tests at port 8868).

Tests cover:
  1. Dataset-history mode (use_dataset_history=True): pre-built messages are
     issued as-is; each turn is issued sequentially per conversation.
  2. Live-history mode (use_dataset_history=False): messages are built at
     runtime from ConversationManager.message_history; the injected messages
     grow with each turn.
  3. Multiple concurrent conversations complete successfully.
  4. Turn ordering: turn N+1 is never issued before turn N completes.
  5. Live concurrency: parametrized target_concurrency levels against a real
     model endpoint verify all turns complete regardless of throttle setting.
"""

import asyncio
import json
import random
import time
from urllib.parse import urljoin
from urllib.request import urlopen

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


# ---------------------------------------------------------------------------
# Live endpoint fixtures and helpers
# ---------------------------------------------------------------------------

_LIVE_ENDPOINT = "http://localhost:8868"


def _query_model_name(endpoint: str) -> str:
    """Return the first model name from the endpoint, or skip if unreachable."""
    try:
        with urlopen(f"{endpoint}/v1/models", timeout=5.0) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception as e:
        pytest.skip(f"Live endpoint {endpoint} not reachable: {e}")
        return ""


def _make_live_rows(
    model: str, n_conversations: int = 20, n_user_turns: int = 3
) -> list[dict]:
    """Build a multi-conversation dataset rows list.

    Each conversation has n_user_turns user turns interleaved with scripted
    assistant placeholders (needed to satisfy the turn-structure validator but
    never sent to the endpoint). The resulting dataset produces
    n_conversations × n_user_turns client-turn samples.
    """
    rows = []
    _user_prompts = [
        "Reply with exactly one word: the number {n} in English.",
        "Add one to the previous number. Reply with only that word.",
        "Add one more. Reply with only that word.",
    ]
    for i in range(n_conversations):
        conv_id = f"live_conv_{i:03d}"
        turn = 1
        for j in range(n_user_turns):
            prompt = _user_prompts[j % len(_user_prompts)].format(n=i + 1)
            rows.append(
                {
                    "conversation_id": conv_id,
                    "turn": turn,
                    "role": "user",
                    "content": prompt,
                    "model": model,
                    "max_completion_tokens": 10,
                }
            )
            turn += 1
            if j < n_user_turns - 1:
                rows.append(
                    {
                        "conversation_id": conv_id,
                        "turn": turn,
                        "role": "assistant",
                        "content": "placeholder",
                    }
                )
                turn += 1
    return rows


async def _run_live_session(
    model: str,
    n_conversations: int,
    n_user_turns: int,
    target_concurrency: int | None,
    timeout_s: float = 300.0,
) -> tuple[int, dict[str, str]]:
    """Run a live multi-turn session against the endpoint at _LIVE_ENDPOINT.

    Returns (issued_count, {query_id: response_text}).
    """
    rows = _make_live_rows(model, n_conversations, n_user_turns)
    ds = MultiTurnDataset(dataframe=pd.DataFrame(rows))
    ds.load()

    mt_cfg = MultiTurnConfig(
        turn_timeout_s=60.0,
        use_dataset_history=True,
    )
    strategy = MultiTurnStrategy(
        conversation_manager=ConversationManager(),
        dataset_metadata=ds.conversation_metadata,
        multi_turn_config=mt_cfg,
        target_concurrency=target_concurrency,
    )

    loop = asyncio.get_running_loop()
    responses: dict[str, str] = {}

    def on_complete(result: QueryResult) -> None:
        strategy.on_sample_complete(result)
        responses[result.id] = result.get_response_output_string()

    http_config = HTTPClientConfig(
        endpoint_urls=[f"{_LIVE_ENDPOINT}/v1/chat/completions"],
        warmup_connections=0,
        num_workers=4,
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
            max_duration_ms=int(timeout_s * 1000),
            n_samples_from_dataset=ds.num_samples(),
            n_samples_to_issue=ds.num_samples(),
            min_sample_count=1,
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        phase = PhaseConfig("perf", rt, ds, PhaseType.PERFORMANCE, strategy=strategy)
        result = await asyncio.wait_for(session.run([phase]), timeout=timeout_s)
        return result.perf_results[0].issued_count, responses
    finally:
        await http_client.shutdown_async()


# ---------------------------------------------------------------------------
# Live concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_concurrency",
    [
        pytest.param(1, id="concurrency_1"),
        pytest.param(4, id="concurrency_4"),
        pytest.param(None, id="concurrency_unlimited"),
    ],
)
async def test_live_concurrency(target_concurrency):
    """All turns of 20 concurrent conversations complete for each concurrency level.

    Uses the live model endpoint at port 8868. Each conversation has 3 user
    turns (60 total requests). Verifies that every turn receives a non-empty
    response regardless of the concurrency throttle applied by target_concurrency.
    """
    model = _query_model_name(_LIVE_ENDPOINT)
    n_conversations = 20
    n_user_turns = 3
    expected_turns = n_conversations * n_user_turns  # 60 total requests

    issued, responses = await _run_live_session(
        model=model,
        n_conversations=n_conversations,
        n_user_turns=n_user_turns,
        target_concurrency=target_concurrency,
        timeout_s=300.0,
    )

    assert issued == expected_turns, f"Expected {expected_turns} issued, got {issued}"
    assert (
        len(responses) == expected_turns
    ), f"Expected {expected_turns} responses, got {len(responses)}"
    for qid, text in responses.items():
        assert text.strip(), f"Query {qid} returned empty response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_turn_ordering_multi_conversation():
    """Turn N+1 of each conversation is always issued after turn N completes.

    Runs 10 conversations with 3 turns each concurrently (30 total requests).
    Records per-query completion timestamps and asserts that within every
    conversation each successive turn completes no earlier than the previous.
    """
    model = _query_model_name(_LIVE_ENDPOINT)
    n_conversations = 10
    n_user_turns = 3
    rows = _make_live_rows(model, n_conversations, n_user_turns)

    ds = MultiTurnDataset(dataframe=pd.DataFrame(rows))
    ds.load()

    conv_manager = ConversationManager()
    mt_cfg = MultiTurnConfig(turn_timeout_s=60.0, use_dataset_history=True)
    strategy = MultiTurnStrategy(
        conversation_manager=conv_manager,
        dataset_metadata=ds.conversation_metadata,
        multi_turn_config=mt_cfg,
    )

    complete_times: dict[str, float] = {}
    orig_on_sample_complete = strategy.on_sample_complete

    def tracked_complete(result: QueryResult) -> None:
        complete_times[result.id] = time.monotonic()
        orig_on_sample_complete(result)

    strategy.on_sample_complete = tracked_complete

    loop = asyncio.get_running_loop()
    responses: dict[str, str] = {}

    http_config = HTTPClientConfig(
        endpoint_urls=[f"{_LIVE_ENDPOINT}/v1/chat/completions"],
        warmup_connections=0,
        num_workers=4,
    )
    http_client = await HTTPEndpointClient.create(http_config, loop)
    issuer = HttpClientSampleIssuer(http_client)

    try:

        def on_complete(result: QueryResult) -> None:
            tracked_complete(result)
            responses[result.id] = result.get_response_output_string()

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
            max_duration_ms=300_000,
            n_samples_from_dataset=ds.num_samples(),
            n_samples_to_issue=ds.num_samples(),
            min_sample_count=1,
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(type=LoadPatternType.MAX_THROUGHPUT),
        )
        phase = PhaseConfig("perf", rt, ds, PhaseType.PERFORMANCE, strategy=strategy)
        result = await asyncio.wait_for(session.run([phase]), timeout=300.0)
    finally:
        await http_client.shutdown_async()

    expected_total = n_conversations * n_user_turns
    assert result.perf_results[0].issued_count == expected_total

    # Build index → query_id map and verify per-conversation ordering.
    # Samples are grouped by conversation, turns sorted ascending within each:
    #   conv_0_t1, conv_0_t2, conv_0_t3, conv_1_t1, ...
    uuid_to_index = result.perf_results[0].uuid_to_index
    index_to_query = {v: k for k, v in uuid_to_index.items()}

    for conv_i in range(n_conversations):
        base = conv_i * n_user_turns
        for turn_j in range(n_user_turns - 1):
            q_cur = index_to_query[base + turn_j]
            q_next = index_to_query[base + turn_j + 1]
            assert complete_times[q_cur] <= complete_times[q_next], (
                f"conv {conv_i}: turn {turn_j + 2} completed before turn {turn_j + 1} "
                f"(t{turn_j + 1}={complete_times[q_cur]:.4f}, "
                f"t{turn_j + 2}={complete_times[q_next]:.4f})"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_large_concurrency():
    """All turns complete correctly under a large concurrency limit (>=512).

    Uses 200 conversations × 3 turns = 600 total requests with
    target_concurrency=512. The semaphore allows up to 512 simultaneous
    in-flight requests, so the first wave of 200 first-turns is issued
    without throttling, and subsequent turns queue naturally. Verifies
    that all 600 turns complete and return non-empty responses, confirming
    the semaphore implementation handles large values without deadlock or
    starvation.
    """
    model = _query_model_name(_LIVE_ENDPOINT)
    n_conversations = 200
    n_user_turns = 3
    expected_turns = n_conversations * n_user_turns  # 600 total requests

    issued, responses = await _run_live_session(
        model=model,
        n_conversations=n_conversations,
        n_user_turns=n_user_turns,
        target_concurrency=512,
        timeout_s=300.0,
    )

    assert issued == expected_turns, f"Expected {expected_turns} issued, got {issued}"
    assert (
        len(responses) == expected_turns
    ), f"Expected {expected_turns} responses, got {len(responses)}"
    for qid, text in responses.items():
        assert text.strip(), f"Query {qid} returned empty response"
