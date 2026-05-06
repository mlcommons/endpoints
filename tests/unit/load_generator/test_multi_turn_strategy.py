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

    async def complete_turns():
        await asyncio.sleep(0.01)
        result = QueryResult(
            id="q0000", response_output=TextModelOutput(output="response 1")
        )
        strategy.on_sample_complete(result)

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
        complete_timestamps[0] = time.monotonic()
        result = QueryResult(id="q0000", response_output=TextModelOutput(output="r1"))
        strategy.on_sample_complete(result)
        await asyncio.sleep(0.05)
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
async def test_on_sample_complete_routes_to_manager():
    """on_sample_complete marks the turn complete in the ConversationManager."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=1)
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    # Simulate issuer registering conv_id in _inflight
    strategy._inflight["q0001"] = "conv1"

    result = QueryResult(id="q0001", response_output=TextModelOutput(output="hello"))
    strategy.on_sample_complete(result)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.completed_turns == 1
    assert state.is_complete()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_response_marks_turn_failed():
    """on_sample_complete marks failed when result.error is set."""
    from inference_endpoint.core.types import ErrorData

    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=1)
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    strategy._inflight["q0001"] = "conv1"

    result = QueryResult(
        id="q0001",
        response_output=None,
        error=ErrorData(error_type="timeout", error_message="timed out"),
    )
    strategy.on_sample_complete(result)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.failed_turns == 1


def _make_metadata_with_system(
    conversations: dict[str, list[int]],
    system_prompts: dict[str, str | None] | None = None,
) -> dict:
    """Build metadata dict including system_prompts_by_conv."""
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
    return {
        "samples": samples,
        "system_prompts_by_conv": system_prompts or {},
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_history_initializes_system_prompt():
    """In live-history mode, ConversationManager.message_history starts with system message."""
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _make_metadata_with_system(
        {"conv1": [1]},
        system_prompts={"conv1": "Be helpful"},
    )
    mt_cfg = MultiTurnConfig(use_dataset_history=False, turn_timeout_s=10.0)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=mt_cfg)
    issuer = FakePhaseIssuer()

    async def complete_turn():
        await asyncio.sleep(0.01)
        result = QueryResult(
            id="q0000", response_output=TextModelOutput(output="response")
        )
        strategy.on_sample_complete(result)

    asyncio.create_task(complete_turn())
    await strategy.execute(issuer)

    state = conv_manager.get_state("conv1")
    assert state is not None
    # message_history[0] must be the system message
    assert len(state.message_history) >= 1
    assert state.message_history[0] == {"role": "system", "content": "Be helpful"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_history_no_system_prompt_when_none():
    """In live-history mode, no system message is prepended when system_prompt is None."""
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _make_metadata_with_system(
        {"conv1": [1]},
        system_prompts={"conv1": None},
    )
    mt_cfg = MultiTurnConfig(use_dataset_history=False, turn_timeout_s=10.0)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=mt_cfg)
    issuer = FakePhaseIssuer()

    async def complete_turn():
        await asyncio.sleep(0.01)
        result = QueryResult(
            id="q0000", response_output=TextModelOutput(output="response")
        )
        strategy.on_sample_complete(result)

    asyncio.create_task(complete_turn())
    await strategy.execute(issuer)

    state = conv_manager.get_state("conv1")
    assert state is not None
    # No system message should be in history
    system_msgs = [m for m in state.message_history if m.get("role") == "system"]
    assert len(system_msgs) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dataset_history_mode_does_not_inject_system_prompt():
    """In dataset-history mode (use_dataset_history=True), system_message is not passed."""
    conv_manager = ConversationManager()
    metadata = _make_metadata_with_system(
        {"conv1": [1]},
        system_prompts={"conv1": "Some system"},
    )
    # Default: use_dataset_history=True → _store_in_history=False
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    async def complete_turn():
        await asyncio.sleep(0.01)
        result = QueryResult(
            id="q0000", response_output=TextModelOutput(output="response")
        )
        strategy.on_sample_complete(result)

    asyncio.create_task(complete_turn())
    await strategy.execute(issuer)

    state = conv_manager.get_state("conv1")
    assert state is not None
    # message_history should be empty (dataset-history mode doesn't accumulate)
    assert len(state.message_history) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pipeline_error_propagated():
    """execute() re-raises when _issue_next_turn raises an exception."""
    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"conv1": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    class ErrorIssuer:
        issued_count = 0
        issued: list[int] = []

        def issue(self, idx: int, data_override: dict | None = None) -> str | None:
            raise RuntimeError("simulated pipeline error")

    with pytest.raises(RuntimeError, match="simulated pipeline error"):
        await strategy.execute(ErrorIssuer())


@pytest.mark.unit
def test_mark_turn_complete_preserves_tool_calls():
    """mark_turn_complete stores tool_calls in history when metadata contains them."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=1)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
        }
    ]
    conv_manager.mark_turn_complete(
        "conv1",
        response="",
        store_in_history=True,
        metadata={"tool_calls": tool_calls},
    )

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert len(state.message_history) == 1
    msg = state.message_history[0]
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert msg["tool_calls"] == tool_calls


@pytest.mark.unit
def test_mark_turn_complete_with_response_and_tool_calls():
    """mark_turn_complete stores both content and tool_calls when both are present."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=1)

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
    ]
    conv_manager.mark_turn_complete(
        "conv1",
        response="Calling search...",
        store_in_history=True,
        metadata={"tool_calls": tool_calls},
    )

    state = conv_manager.get_state("conv1")
    assert state is not None
    msg = state.message_history[0]
    assert msg["content"] == "Calling search..."
    assert msg["tool_calls"] == tool_calls


@pytest.mark.unit
def test_mark_turn_complete_no_history_when_empty():
    """mark_turn_complete does not append when response is empty and no tool_calls."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=1)

    conv_manager.mark_turn_complete("conv1", response="", store_in_history=True)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert len(state.message_history) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_on_sample_complete_passes_metadata():
    """on_sample_complete forwards result.metadata (including tool_calls) to ConversationManager."""
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata_dict = _make_metadata_with_system({"conv1": [1]})
    mt_cfg = MultiTurnConfig(use_dataset_history=False, turn_timeout_s=10.0)
    strategy = MultiTurnStrategy(conv_manager, metadata_dict, multi_turn_config=mt_cfg)

    conv_manager.get_or_create("conv1", expected_client_turns=1)
    strategy._inflight["q0001"] = "conv1"

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
    ]
    result = QueryResult(
        id="q0001",
        response_output=TextModelOutput(output=""),
        metadata={"tool_calls": tool_calls},
    )
    strategy.on_sample_complete(result)

    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.completed_turns == 1
    assert len(state.message_history) == 1
    assert state.message_history[0]["tool_calls"] == tool_calls
    assert state.message_history[0]["content"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrency_limits_active_conversations():
    """target_concurrency=2 starts at most 2 conversation pipelines simultaneously.

    Uses 2-turn conversations so each pipeline has an await point between turns.
    With 4 conversations and 2 workers, the 3rd and 4th conversations cannot start
    until a worker finishes its current conversation.
    """
    conv_manager = ConversationManager()
    # 4 two-turn conversations; pipeline awaits turn-1 response before issuing turn-2
    metadata = _make_dataset_metadata(
        {"conv1": [1, 2], "conv2": [1, 2], "conv3": [1, 2], "conv4": [1, 2]}
    )
    strategy = MultiTurnStrategy(conv_manager, metadata, target_concurrency=2)
    issuer = FakePhaseIssuer()

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                q = f"q{idx:04d}"
                strategy.on_sample_complete(
                    QueryResult(id=q, response_output=TextModelOutput(output="r"))
                )
                already_done += 1
            await asyncio.sleep(0.02)

    responder_task = asyncio.create_task(auto_respond())
    execute_task = asyncio.create_task(strategy.execute(issuer))

    # Let both seed turns get issued before auto_respond fires
    await asyncio.sleep(0.01)

    # Only 2 workers → exactly 2 turn-1 queries issued (conv3/conv4 not started yet)
    assert issuer.issued_count == 2

    await asyncio.wait_for(execute_task, timeout=5.0)
    responder_task.cancel()

    assert issuer.issued_count == 8  # 4 conversations × 2 turns


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_slot_reuse():
    """With target_concurrency=1, worker completes conv1 before starting conv2.

    The single slot must process both turns of conv1 before conv2's turn 1 is issued.
    """
    conv_manager = ConversationManager()
    # 2 two-turn conversations; sample indices: conv1→[0,1], conv2→[2,3]
    metadata = _make_dataset_metadata({"conv1": [1, 2], "conv2": [1, 2]})
    strategy = MultiTurnStrategy(conv_manager, metadata, target_concurrency=1)
    issuer = FakePhaseIssuer()

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                q = f"q{idx:04d}"
                strategy.on_sample_complete(
                    QueryResult(id=q, response_output=TextModelOutput(output="r"))
                )
                already_done += 1
            await asyncio.sleep(0.02)

    responder_task = asyncio.create_task(auto_respond())
    await strategy.execute(issuer)
    responder_task.cancel()

    # Single slot: conv1 turns (samples 0,1) must be issued before conv2 turns (2,3)
    assert issuer.issued[:2] == [0, 1], "Conv1 turns should be issued before conv2"
    assert issuer.issued[2:] == [2, 3], "Conv2 turns should follow conv1"
