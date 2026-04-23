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

"""Unit tests for MultiTurnStrategy."""

import asyncio

import pytest
from inference_endpoint.core.types import QueryResult, TextModelOutput
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.multi_turn_strategy import MultiTurnStrategy


class FakePhaseIssuer:
    """Minimal PhaseIssuerProtocol stub."""

    def __init__(self, stop_after: int | None = None):
        self._count = 0
        self._stop_after = stop_after
        self.issued: list[int] = []
        self.issued_count = 0

    def issue(self, sample_index: int, data_override: dict | None = None) -> str | None:
        if self._stop_after is not None and self._count >= self._stop_after:
            return None
        self._count += 1
        self.issued_count += 1
        query_id = f"q{sample_index:04d}"
        self.issued.append(sample_index)
        return query_id


def _make_dataset_metadata(conversations: dict[str, list[int]]) -> dict:
    """Build dataset_metadata dict from {conv_id: [turn_numbers]} mapping."""
    samples = []
    sample_index = 0
    for conv_id, turns in conversations.items():
        for turn in turns:
            samples.append(
                {
                    "conversation_id": conv_id,
                    "turn": turn,
                    "sample_index": sample_index,
                }
            )
            sample_index += 1
    return {"samples": samples}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_conversation_single_turn():
    """Single conversation, single turn — should issue exactly one sample."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    # Simulate response completion (turn 1 is issued, then completes)
    async def complete_turns():
        # Wait a tick for the strategy to issue the first turn
        await asyncio.sleep(0.01)
        # Mark turn 1 complete
        state = conv_manager.get_state("conv1")
        if state:
            await conv_manager.mark_turn_complete("conv1", "response 1")

    asyncio.create_task(complete_turns())
    count = await strategy.execute(issuer)

    assert count == 1
    assert issuer.issued == [0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_conversation_multi_turn():
    """Single conversation, 3 turns — turns must be issued sequentially."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1, 3, 5]})
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    issued_order: list[str] = []
    original_issue = issuer.issue

    def tracked_issue(idx, data_override=None):
        q = original_issue(idx, data_override=data_override)
        if q:
            issued_order.append(q)
        return q

    issuer.issue = tracked_issue

    async def simulate_responses():
        await asyncio.sleep(0.01)
        for turn_q, resp in [("q0000", "r1"), ("q0001", "r2"), ("q0002", "r3")]:
            # Signal turn complete via on_sample_complete
            result = QueryResult(
                id=turn_q, response_output=TextModelOutput(output=resp)
            )
            strategy.on_sample_complete(result)
            await asyncio.sleep(0.01)

    asyncio.create_task(simulate_responses())
    count = await strategy.execute(issuer)

    assert count == 3
    assert issuer.issued == [0, 1, 2]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_conversations_concurrent():
    """Two conversations run concurrently, each with 2 turns."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1, 3], "conv2": [1, 3]})
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    async def simulate_responses():
        await asyncio.sleep(0.02)
        # Complete all turns for both conversations
        for q_prefix in range(4):
            q = f"q{q_prefix:04d}"
            result = QueryResult(id=q, response_output=TextModelOutput(output="resp"))
            strategy.on_sample_complete(result)
            await asyncio.sleep(0.01)

    asyncio.create_task(simulate_responses())
    count = await strategy.execute(issuer)

    assert count == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_turn_ordering_enforced():
    """Turn 2 must not be issued before Turn 1 completes."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1, 3]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    issue_timestamps: dict[int, float] = {}
    complete_timestamps: dict[int, float] = {}

    class TimedIssuer:
        issued_count = 0
        issued: list[int] = []

        def issue(self, idx: int, data_override: dict | None = None) -> str | None:
            import time

            issue_timestamps[idx] = time.monotonic()
            self.issued.append(idx)
            self.issued_count += 1
            return f"q{idx:04d}"

    issuer = TimedIssuer()

    async def simulate_responses():
        import time

        await asyncio.sleep(0.02)
        # Complete turn 1 (sample 0) after a delay
        complete_timestamps[0] = time.monotonic()
        result = QueryResult(id="q0000", response_output=TextModelOutput(output="r1"))
        strategy.on_sample_complete(result)
        await asyncio.sleep(0.05)
        # Complete turn 2 (sample 1)
        complete_timestamps[1] = time.monotonic()
        result = QueryResult(id="q0001", response_output=TextModelOutput(output="r2"))
        strategy.on_sample_complete(result)

    asyncio.create_task(simulate_responses())
    count = await strategy.execute(issuer)

    assert count == 2
    # Turn 2 (sample index 1) must be issued AFTER turn 1 completes
    assert issue_timestamps[1] >= complete_timestamps[0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_turn_timeout_triggers_failure():
    """A turn that never completes should timeout and abort remaining turns."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1, 3]})
    strategy = MultiTurnStrategy(conv_manager, metadata, target_concurrency=None)
    strategy._turn_timeout_s = 0.1  # Very short timeout for testing
    issuer = FakePhaseIssuer()

    # Do NOT simulate any response — turn 1 will timeout
    await strategy.execute(issuer)

    # Only turn 1 should be issued (turn 2 never gets to run)
    assert issuer.issued_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_query_complete_releases_semaphore():
    """on_query_complete releases the concurrency semaphore."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata, target_concurrency=1)
    assert strategy._sem is not None

    # Acquire the semaphore manually
    await strategy._sem.acquire()
    assert strategy._sem._value == 0  # type: ignore[attr-defined]

    strategy.on_query_complete("some-query")
    assert strategy._sem._value == 1  # type: ignore[attr-defined]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_sample_complete_routes_to_manager():
    """on_sample_complete marks the turn complete in the ConversationManager."""
    conv_manager = ConversationManager()
    await conv_manager.get_or_create("conv1", expected_client_turns=1)
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    # Simulate issuer registering conv_id in _inflight
    strategy._inflight["q0001"] = "conv1"
    # Pre-issue a turn so the state has pending_client_turn
    await conv_manager.mark_turn_issued("conv1", 1)

    result = QueryResult(id="q0001", response_output=TextModelOutput(output="hello"))
    strategy.on_sample_complete(result)

    # Allow the ensure_future coroutine to run
    await asyncio.sleep(0.01)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.completed_client_turns == 1
    assert state.is_complete()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_response_marks_turn_failed():
    """on_sample_complete marks failed when result.error is set."""
    from inference_endpoint.core.types import ErrorData

    conv_manager = ConversationManager()
    await conv_manager.get_or_create("conv1", expected_client_turns=1)
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    strategy._inflight["q0001"] = "conv1"
    await conv_manager.mark_turn_issued("conv1", 1)

    result = QueryResult(
        id="q0001",
        response_output=None,
        error=ErrorData(error_type="timeout", error_message="timed out"),
    )
    strategy.on_sample_complete(result)
    await asyncio.sleep(0.01)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.failed_client_turns == 1
