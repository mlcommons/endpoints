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
from unittest.mock import MagicMock

import pytest
from inference_endpoint.core.record import ErrorEventType, SampleEventType
from inference_endpoint.core.types import ErrorData, QueryResult, TextModelOutput
from inference_endpoint.dataset_manager.multi_turn_dataset import (
    ConversationMetadata,
    ConversationSampleEntry,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.multi_turn_strategy import MultiTurnStrategy


class FakePhaseIssuer:
    """Minimal PhaseIssuerProtocol stub."""

    def __init__(self, stop_after: int | None = None):
        self._count = 0
        self._stop_after = stop_after
        self.issued: list[int] = []
        self.issued_count = 0
        self.inflight: int = 0
        self.uuid_to_index: dict[str, int] = {}
        self.uuid_to_conv_info: dict[str, tuple[str, int | None]] = {}
        self.completed_uuids: set[str] = set()
        self.drained = False

    def issue(
        self,
        sample_index: int,
        data_override: dict | None = None,
        conversation_id: str = "",
        turn: int | None = None,
    ) -> str | None:
        if self._stop_after is not None and self._count >= self._stop_after:
            return None
        self._count += 1
        self.issued_count += 1
        query_id = f"q{sample_index:04d}"
        self.issued.append(sample_index)
        self.uuid_to_conv_info[query_id] = (conversation_id, turn)
        return query_id

    def register_skipped(
        self,
        sample_index: int,
        conversation_id: str = "",
        turn: int | None = None,
    ) -> str | None:
        self.issued_count += 1
        query_id = f"q-skip-{sample_index:04d}"
        self.uuid_to_index[query_id] = sample_index
        self.uuid_to_conv_info[query_id] = (conversation_id, turn)
        self.completed_uuids.add(query_id)
        return query_id

    def mark_inflight_complete(self) -> None:
        self.inflight -= 1
        if self.inflight <= 0:
            self.drained = True


def _make_dataset_metadata(conversations: dict[str, list[int]]) -> ConversationMetadata:
    """Build ConversationMetadata from {conv_id: [turn_numbers]} mapping."""
    samples = []
    sample_index = 0
    for conv_id, turns in conversations.items():
        for turn in turns:
            samples.append(
                ConversationSampleEntry(
                    conversation_id=conv_id,
                    turn=turn,
                    sample_index=sample_index,
                )
            )
            sample_index += 1
    return ConversationMetadata(
        samples=samples,
        num_conversations=len(conversations),
        max_turns_per_conv=max((max(t) for t in conversations.values()), default=0),
        client_turns_per_conversation={c: len(t) for c, t in conversations.items()},
    )


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

    def tracked_issue(idx, data_override=None, conversation_id="", turn=None):
        q = original_issue(
            idx,
            data_override=data_override,
            conversation_id=conversation_id,
            turn=turn,
        )
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

        def issue(
            self,
            idx: int,
            data_override: dict | None = None,
            conversation_id: str = "",
            turn: int | None = None,
        ) -> str | None:
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

    # Turn 1 was issued normally; turn 2 registered as skipped (total = 2)
    assert issuer.issued_count == 2


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
    strategy._all_done = asyncio.Event()

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
) -> ConversationMetadata:
    """Build ConversationMetadata including system_prompts_by_conv."""
    samples = []
    sample_index = 0
    for conv_id, turns in conversations.items():
        for turn in turns:
            samples.append(
                ConversationSampleEntry(
                    conversation_id=conv_id,
                    turn=turn,
                    sample_index=sample_index,
                )
            )
            sample_index += 1
    return ConversationMetadata(
        samples=samples,
        num_conversations=len(conversations),
        max_turns_per_conv=max((max(t) for t in conversations.values()), default=0),
        client_turns_per_conversation={c: len(t) for c, t in conversations.items()},
        system_prompts_by_conv=system_prompts or {},
    )


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

        def issue(
            self,
            idx: int,
            data_override: dict | None = None,
            conversation_id: str = "",
            turn: int | None = None,
        ) -> str | None:
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_publishes_error_and_complete_events():
    """_handle_timeout publishes ERROR+COMPLETE for timed-out turn and each dropped turn."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv-x", expected_client_turns=3)
    metadata = _make_dataset_metadata({"conv-x": [1, 2, 3]})
    strategy = MultiTurnStrategy(conv_manager, metadata)

    publisher = MagicMock()
    on_sample_complete = MagicMock()
    strategy._session_publisher = publisher
    strategy._session_on_sample_complete = on_sample_complete

    # Seed: turn 1 in-flight, turns 2+3 still pending
    strategy._inflight["q-x"] = "conv-x"
    strategy._active_iters["conv-x"] = ([(1, 2), (2, 3)], 0)

    issuer = FakePhaseIssuer()
    issuer.uuid_to_index["q-x"] = 0
    issuer.uuid_to_conv_info["q-x"] = ("conv-x", 1)
    issuer.inflight = 1

    strategy._all_done = asyncio.Event()
    strategy._loop = asyncio.get_running_loop()
    strategy._phase_issuer = issuer

    strategy._handle_timeout("q-x", "conv-x")

    # 2 events for the timed-out turn + 2 per dropped turn (ERROR + COMPLETE each)
    assert publisher.publish.call_count == 6
    assert issuer.inflight == 0
    assert issuer.drained is True
    assert "q-x" in issuer.completed_uuids
    # Conv info cleared so a late real response can't reuse stale state.
    assert "q-x" not in issuer.uuid_to_conv_info

    published_records = [call.args[0] for call in publisher.publish.call_args_list]
    event_turn_pairs = {(r.event_type, r.turn) for r in published_records}
    assert (ErrorEventType.GENERIC, 1) in event_turn_pairs
    assert (SampleEventType.COMPLETE, 1) in event_turn_pairs
    assert (ErrorEventType.GENERIC, 2) in event_turn_pairs
    assert (SampleEventType.COMPLETE, 2) in event_turn_pairs
    assert (ErrorEventType.GENERIC, 3) in event_turn_pairs
    assert (SampleEventType.COMPLETE, 3) in event_turn_pairs

    assert issuer.issued_count == 2
    assert "q-skip-0001" in issuer.uuid_to_index
    assert "q-skip-0002" in issuer.uuid_to_index
    assert issuer.completed_uuids == {"q-x", "q-skip-0001", "q-skip-0002"}
    assert on_sample_complete.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_issue_passes_conversation_id_and_turn_to_phase_issuer():
    """MultiTurnStrategy must forward (conv_id, turn) to phase_issuer.issue()."""
    conv_manager = ConversationManager()
    conversations = {"conv-A": [1, 2], "conv-B": [1]}
    metadata = _make_dataset_metadata(conversations)
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    # Build the expected (query_id -> (conv_id, turn)) map from the same
    # sample_index ordering _make_dataset_metadata uses, so the test does not
    # encode that ordering as magic numbers.
    expected: dict[str, tuple[str, int]] = {}
    sample_index = 0
    for conv_id, turns in conversations.items():
        for turn in turns:
            expected[f"q{sample_index:04d}"] = (conv_id, turn)
            sample_index += 1

    async def respond_in_order():
        await asyncio.sleep(0.01)
        for query_id in expected:
            strategy.on_sample_complete(
                QueryResult(id=query_id, response_output=TextModelOutput(output="ok"))
            )
            await asyncio.sleep(0.005)

    asyncio.create_task(respond_in_order())
    await strategy.execute(issuer)

    assert issuer.uuid_to_conv_info == expected


def _metadata_with_delay(
    conv_id: str, turns: list[int], delay_turn: int, delay: float
) -> ConversationMetadata:
    md = _make_dataset_metadata({conv_id: turns})
    md.delay_seconds_by_key = {(conv_id, delay_turn): delay}
    return md


@pytest.mark.unit
@pytest.mark.asyncio
async def test_abort_remaining_turns_includes_pending_delayed_turn():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    conv_manager.get_or_create("c1", expected_client_turns=3)
    metadata = _metadata_with_delay("c1", [1, 2, 3], delay_turn=2, delay=60.0)
    cfg = MultiTurnConfig(turn_timeout_s=5.0, inject_tool_delay=True)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=cfg)
    issuer = FakePhaseIssuer()
    publisher = MagicMock()
    on_sample_complete = MagicMock()

    strategy._loop = asyncio.get_running_loop()
    strategy._phase_issuer = issuer
    strategy._session_publisher = publisher
    strategy._session_on_sample_complete = on_sample_complete
    strategy._active_iters["c1"] = ([(0, 1), (1, 2), (2, 3)], 1)

    strategy._issue_next_turn("c1")

    assert issuer.issued == []
    delay_handle = strategy._delay_handles["c1"]

    dropped = strategy._abort_remaining_turns("c1", reason="prior turn failed")

    assert dropped == 2
    assert delay_handle.cancelled()
    assert "c1" not in strategy._delay_handles
    assert "c1" not in strategy._active_iters
    assert "q-skip-0001" in issuer.uuid_to_index
    assert "q-skip-0002" in issuer.uuid_to_index
    assert issuer.completed_uuids == {"q-skip-0001", "q-skip-0002"}
    assert on_sample_complete.call_count == 2

    published_records = [call.args[0] for call in publisher.publish.call_args_list]
    event_turn_pairs = {(r.event_type, r.turn) for r in published_records}
    assert event_turn_pairs == {
        (ErrorEventType.GENERIC, 2),
        (SampleEventType.COMPLETE, 2),
        (ErrorEventType.GENERIC, 3),
        (SampleEventType.COMPLETE, 3),
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_waits_for_delayed_first_turns():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"c1": [1], "c2": [1]})
    metadata.delay_seconds_by_key = {("c1", 1): 0.02, ("c2", 1): 0.02}
    cfg = MultiTurnConfig(turn_timeout_s=5.0, inject_tool_delay=True)
    strategy = MultiTurnStrategy(
        conv_manager,
        metadata,
        multi_turn_config=cfg,
        target_concurrency=2,
    )
    issuer = FakePhaseIssuer()

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                strategy.on_sample_complete(
                    QueryResult(
                        id=f"q{idx:04d}", response_output=TextModelOutput(output="r")
                    )
                )
                already_done += 1
            await asyncio.sleep(0.005)

    responder = asyncio.create_task(auto_respond())
    count = await asyncio.wait_for(strategy.execute(issuer), timeout=1.0)
    responder.cancel()

    assert count == 2
    assert issuer.issued == [0, 1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_tool_delay_defers_issue_via_call_later():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _metadata_with_delay("c1", [1, 2], delay_turn=2, delay=0.05)
    cfg = MultiTurnConfig(turn_timeout_s=5.0, inject_tool_delay=True)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=cfg)
    issuer = FakePhaseIssuer()

    issue_times: dict[int, float] = {}
    original_issue = issuer.issue

    def timing_issue(idx, data_override=None, conversation_id="", turn=None):
        q = original_issue(
            idx,
            data_override=data_override,
            conversation_id=conversation_id,
            turn=turn,
        )
        if q:
            issue_times[idx] = asyncio.get_running_loop().time()
        return q

    issuer.issue = timing_issue  # type: ignore[method-assign]

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                strategy.on_sample_complete(
                    QueryResult(
                        id=f"q{idx:04d}", response_output=TextModelOutput(output="r")
                    )
                )
                already_done += 1
            await asyncio.sleep(0.005)

    responder = asyncio.create_task(auto_respond())
    await strategy.execute(issuer)
    responder.cancel()

    assert issuer.issued == [0, 1]
    gap = issue_times[1] - issue_times[0]
    assert gap >= 0.04, f"expected >=0.04s gap between issues, got {gap:.4f}s"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fill_slot_failure_does_not_hang_execute():
    """_fill_slot errors surface from execute() instead of hanging on _all_done."""
    conv_manager = ConversationManager()
    # Two 1-turn conversations; target_concurrency=1 so conv2 only starts via _fill_slot.
    metadata = _make_dataset_metadata({"conv1": [1], "conv2": [1]})
    strategy = MultiTurnStrategy(conv_manager, metadata, target_concurrency=1)

    call_count = 0

    class RaisingIssuer:
        issued_count = 0
        inflight = 0
        uuid_to_index: dict = {}
        uuid_to_conv_info: dict = {}
        completed_uuids: set = set()

        def issue(self, idx, data_override=None, conversation_id="", turn=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                self.issued_count += 1
                query_id = f"q{idx:04d}"
                self.uuid_to_conv_info[query_id] = (conversation_id, turn)
                return query_id
            # Raises on the second call, which is triggered by _fill_slot after conv1 completes.
            raise RuntimeError("simulated slot-refill failure")

    issuer = RaisingIssuer()

    async def complete_conv1():
        await asyncio.sleep(0.02)
        result = QueryResult(id="q0000", response_output=TextModelOutput(output="ok"))
        strategy.on_sample_complete(result)

    asyncio.create_task(complete_conv1())
    with pytest.raises(RuntimeError, match="simulated slot-refill failure"):
        await asyncio.wait_for(strategy.execute(issuer), timeout=2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_error_turn_aborts_remaining_turns():
    """on_sample_complete with result.error aborts and marks-failed remaining turns."""
    conv_manager = ConversationManager()
    conv_manager.get_or_create("conv1", expected_client_turns=3)
    metadata = _make_dataset_metadata({"conv1": [1, 2, 3]})
    strategy = MultiTurnStrategy(conv_manager, metadata)
    issuer = FakePhaseIssuer()

    publisher = MagicMock()
    on_sample_complete = MagicMock()
    strategy._session_publisher = publisher
    strategy._session_on_sample_complete = on_sample_complete

    strategy._all_done = asyncio.Event()
    strategy._loop = asyncio.get_running_loop()
    strategy._phase_issuer = issuer

    # Seed: conv1 is active with turns 2 and 3 still pending
    remaining_turns = ([(1, 2), (2, 3)], 0)
    strategy._active_iters["conv1"] = remaining_turns
    strategy._inflight["q0001"] = "conv1"
    strategy._conv_states["conv1"] = conv_manager.get_state("conv1")

    result = QueryResult(
        id="q0001",
        response_output=None,
        error=ErrorData(
            error_type="endpoint_error", error_message="500 Internal Server Error"
        ),
    )
    strategy.on_sample_complete(result)

    # Conversation should no longer be active
    assert "conv1" not in strategy._active_iters
    # Remaining 2 turns were marked failed
    state = conv_manager.get_state("conv1")
    assert state is not None
    assert state.failed_turns == 3  # the failing turn + 2 dropped

    assert issuer.issued_count == 2
    assert "q-skip-0001" in issuer.uuid_to_index
    assert "q-skip-0002" in issuer.uuid_to_index
    assert issuer.completed_uuids == {"q-skip-0001", "q-skip-0002"}
    assert issuer.inflight == 0

    assert on_sample_complete.call_count == 2
    for call in on_sample_complete.call_args_list:
        dropped_result = call.args[0]
        assert dropped_result.error is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_tool_delay_disabled_issues_immediately():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _metadata_with_delay("c1", [1, 2], delay_turn=2, delay=2.0)
    cfg = MultiTurnConfig(turn_timeout_s=5.0, inject_tool_delay=False)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=cfg)
    issuer = FakePhaseIssuer()

    issue_times: dict[int, float] = {}
    original_issue = issuer.issue

    def timing_issue(idx, data_override=None, conversation_id="", turn=None):
        q = original_issue(
            idx,
            data_override=data_override,
            conversation_id=conversation_id,
            turn=turn,
        )
        if q:
            issue_times[idx] = asyncio.get_running_loop().time()
        return q

    issuer.issue = timing_issue  # type: ignore[method-assign]

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                strategy.on_sample_complete(
                    QueryResult(
                        id=f"q{idx:04d}", response_output=TextModelOutput(output="r")
                    )
                )
                already_done += 1
            await asyncio.sleep(0.005)

    responder = asyncio.create_task(auto_respond())
    await asyncio.wait_for(strategy.execute(issuer), timeout=1.0)
    responder.cancel()

    assert issuer.issued == [0, 1]
    gap = issue_times[1] - issue_times[0]
    assert gap < 0.5, f"delay should be ignored, observed {gap:.3f}s gap"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_tool_delay_no_dataset_field_back_compat():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _make_dataset_metadata({"c1": [1, 2]})
    cfg = MultiTurnConfig(turn_timeout_s=5.0, inject_tool_delay=True)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=cfg)
    issuer = FakePhaseIssuer()

    async def auto_respond():
        already_done = 0
        while True:
            while already_done < len(issuer.issued):
                idx = issuer.issued[already_done]
                strategy.on_sample_complete(
                    QueryResult(
                        id=f"q{idx:04d}", response_output=TextModelOutput(output="r")
                    )
                )
                already_done += 1
            await asyncio.sleep(0.005)

    responder = asyncio.create_task(auto_respond())
    await asyncio.wait_for(strategy.execute(issuer), timeout=1.0)
    responder.cancel()

    assert issuer.issued == [0, 1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inject_tool_delay_cancels_on_timeout():
    from inference_endpoint.config.schema import MultiTurnConfig

    conv_manager = ConversationManager()
    metadata = _metadata_with_delay("c1", [1, 2, 3], delay_turn=3, delay=1.0)
    cfg = MultiTurnConfig(turn_timeout_s=0.1, inject_tool_delay=True)
    strategy = MultiTurnStrategy(conv_manager, metadata, multi_turn_config=cfg)
    issuer = FakePhaseIssuer()

    async def respond_first_only():
        while not issuer.issued:
            await asyncio.sleep(0.005)
        strategy.on_sample_complete(
            QueryResult(id="q0000", response_output=TextModelOutput(output="r"))
        )

    responder = asyncio.create_task(respond_first_only())
    await asyncio.wait_for(strategy.execute(issuer), timeout=2.0)
    responder.cancel()

    assert issuer.issued == [
        0,
        1,
    ], f"turn 3 should not have been issued after timeout; issued={issuer.issued}"
