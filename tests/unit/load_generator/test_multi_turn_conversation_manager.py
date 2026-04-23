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

import asyncio
import logging

import pytest
from inference_endpoint.load_generator.conversation_manager import (
    ConversationManager,
    ConversationState,
)


@pytest.mark.unit
def test_conversation_state_initialization():
    """Test ConversationState initializes with correct default values."""
    state = ConversationState(conversation_id="conv_001")

    assert state.conversation_id == "conv_001"
    assert state.current_turn == 0
    assert state.pending_client_turn is None


@pytest.mark.unit
def test_conversation_state_add_client_turn():
    """Test adding a client turn updates sequencing state."""
    state = ConversationState(conversation_id="conv_001")

    state.add_client_turn(1)

    assert state.pending_client_turn == 1
    assert state.issued_client_turns == 1
    assert state.current_turn == 0  # Not incremented until assistant response


@pytest.mark.unit
def test_conversation_state_add_assistant_turn():
    """Test adding assistant turn completes turn cycle."""
    state = ConversationState(conversation_id="conv_001")

    state.add_client_turn(1)
    state.add_assistant_turn()

    assert state.current_turn == 2
    assert state.pending_client_turn is None
    assert state.completed_client_turns == 1


@pytest.mark.unit
def test_conversation_state_late_response_after_complete_is_silently_ignored(caplog):
    """Late response for a conversation that already completed is silently dropped."""
    state = ConversationState(conversation_id="conv_001", expected_client_turns=1)

    state.add_client_turn(1)
    state.add_assistant_turn()
    assert state.is_complete()

    completed_before = state.completed_client_turns
    current_turn_before = state.current_turn

    with caplog.at_level(logging.WARNING):
        state.add_assistant_turn()

    assert state.completed_client_turns == completed_before
    assert state.current_turn == current_turn_before
    assert "no pending client turn" not in caplog.text


@pytest.mark.unit
def test_conversation_state_is_ready_for_turn():
    """Test turn readiness checks using completion counts."""
    state = ConversationState(conversation_id="conv_001")

    assert not state.is_ready_for_turn()

    state.add_client_turn(1)
    assert not state.is_ready_for_turn()

    state.add_assistant_turn()
    assert state.is_ready_for_turn()

    state.add_client_turn(2)
    assert not state.is_ready_for_turn()

    state.add_assistant_turn()
    assert state.is_ready_for_turn()


@pytest.mark.unit
def test_conversation_state_multi_turn_sequence():
    """Test multi-turn conversation flow updates current_turn correctly."""
    state = ConversationState(conversation_id="conv_001")

    state.add_client_turn(1)
    state.add_assistant_turn()
    assert state.current_turn == 2

    state.add_client_turn(3)
    state.add_assistant_turn()
    assert state.current_turn == 4

    state.add_client_turn(5)
    state.add_assistant_turn()
    assert state.current_turn == 6


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_get_or_create():
    """Test get_or_create returns same state for same conversation_id."""
    manager = ConversationManager()

    state1 = await manager.get_or_create("conv_001")
    state2 = await manager.get_or_create("conv_001")

    assert state1 is state2
    assert state1.conversation_id == "conv_001"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_multiple_conversations():
    """Test manager can track multiple conversations independently."""
    manager = ConversationManager()

    state1 = await manager.get_or_create("conv_001")
    state2 = await manager.get_or_create("conv_002")

    assert state1 is not state2

    await manager.mark_turn_issued("conv_001", 1)
    await manager.mark_turn_complete("conv_001", "Response to conv_001")

    assert state1.current_turn == 2
    assert state2.current_turn == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_mark_turn_issued():
    """Test mark_turn_issued updates sequencing state."""
    manager = ConversationManager()
    state = await manager.get_or_create("conv_001")

    await manager.mark_turn_issued("conv_001", 1)

    assert state.pending_client_turn == 1
    assert state.issued_client_turns == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_mark_turn_complete():
    """Test mark_turn_complete updates sequencing state."""
    manager = ConversationManager()
    state = await manager.get_or_create("conv_001")

    await manager.mark_turn_issued("conv_001", 1)
    await manager.mark_turn_complete("conv_001", "Assistant response")

    assert state.current_turn == 2
    assert state.pending_client_turn is None
    assert state.completed_client_turns == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_wait_for_turn_ready_immediate():
    """Test wait_for_turn_ready returns immediately when previous turn is complete."""
    manager = ConversationManager()
    await manager.get_or_create("conv_001")

    await manager.mark_turn_issued("conv_001", 1)
    await manager.mark_turn_complete("conv_001", "First response")

    result = await manager.wait_for_turn_ready("conv_001", 9, timeout=1.0)

    assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_wait_for_turn_ready_blocking():
    """Test wait_for_turn_ready blocks until previous turn completes."""
    manager = ConversationManager()
    await manager.get_or_create("conv_001")

    await manager.mark_turn_issued("conv_001", 1)

    ready_flag = []

    async def waiter():
        result = await manager.wait_for_turn_ready("conv_001", 3, timeout=2.0)
        if result:
            ready_flag.append(True)

    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert not ready_flag

    await manager.mark_turn_complete("conv_001", "Assistant response")
    await asyncio.sleep(0.05)
    await waiter_task

    assert ready_flag == [True]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_wait_for_turn_ready_timeout():
    """Test wait_for_turn_ready respects timeout."""
    manager = ConversationManager()
    await manager.get_or_create("conv_001")

    await manager.mark_turn_issued("conv_001", 1)

    result = await manager.wait_for_turn_ready("conv_001", 3, timeout=0.1)

    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_completion_tracking():
    """Test conversation completion detection."""
    manager = ConversationManager()

    state = await manager.get_or_create("conv_001", expected_client_turns=2)

    assert not state.is_complete()

    await manager.mark_turn_issued("conv_001", 1)
    assert not state.is_complete()

    await manager.mark_turn_complete("conv_001", "response 1")
    assert not state.is_complete()

    await manager.mark_turn_issued("conv_001", 3)
    await manager.mark_turn_complete("conv_001", "response 2")

    assert state.is_complete()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_completion_without_expected_turns():
    """Test that completion tracking works when expected_client_turns is None."""
    manager = ConversationManager()

    state = await manager.get_or_create("conv_001", expected_client_turns=None)

    assert not state.is_complete()

    await manager.mark_turn_issued("conv_001", 1)
    await manager.mark_turn_complete("conv_001", "response 1")

    assert not state.is_complete()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_completion_with_failures():
    """Test that conversations complete even when turns fail."""
    manager = ConversationManager()
    state = await manager.get_or_create("conv1", expected_client_turns=3)

    await manager.mark_turn_issued("conv1", 1)
    await manager.mark_turn_complete("conv1", "Hi there")
    assert state.completed_client_turns == 1
    assert not state.is_complete()

    await manager.mark_turn_issued("conv1", 2)
    await manager.mark_turn_failed("conv1")
    assert state.completed_client_turns == 2
    assert state.failed_client_turns == 1
    assert not state.is_complete()

    await manager.mark_turn_issued("conv1", 3)
    await manager.mark_turn_complete("conv1", "Bye!")
    assert state.completed_client_turns == 3
    assert state.is_complete()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mark_turn_failed_with_no_pending():
    """Test that marking failed turn without pending turn logs warning."""
    manager = ConversationManager()
    state = await manager.get_or_create("conv1", expected_client_turns=1)

    await manager.mark_turn_failed("conv1")

    assert state.completed_client_turns == 0
    assert state.failed_client_turns == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_all_turns_fail():
    """Test conversation completion when all turns fail."""
    manager = ConversationManager()
    state = await manager.get_or_create("conv1", expected_client_turns=2)

    await manager.mark_turn_issued("conv1", 1)
    await manager.mark_turn_failed("conv1")

    await manager.mark_turn_issued("conv1", 2)
    await manager.mark_turn_failed("conv1")

    assert state.is_complete()
    assert state.completed_client_turns == 2
    assert state.failed_client_turns == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_concurrent_access():
    """Test async concurrent access to multiple conversations."""
    manager = ConversationManager()
    num_conversations = 10
    user_turns_per_conv = 5

    for i in range(num_conversations):
        await manager.get_or_create(f"conv_{i:03d}")

    errors = []

    async def process_conversation(conv_id: str):
        try:
            for user_turn_idx in range(user_turns_per_conv):
                turn = user_turn_idx * 2 + 1

                if user_turn_idx > 0:
                    ready = await manager.wait_for_turn_ready(
                        conv_id, turn, timeout=5.0
                    )
                    if not ready:
                        errors.append(f"{conv_id} turn {turn} timeout")
                        return

                await manager.mark_turn_issued(conv_id, turn)
                await asyncio.sleep(0.001)
                await manager.mark_turn_complete(conv_id, f"Response {turn}")
        except Exception as e:
            errors.append(f"{conv_id} error: {e}")

    tasks = [
        asyncio.create_task(process_conversation(f"conv_{i:03d}"))
        for i in range(num_conversations)
    ]
    await asyncio.gather(*tasks)

    assert not errors, f"Errors occurred: {errors}"

    for i in range(num_conversations):
        conv_id = f"conv_{i:03d}"
        state = manager._conversations[conv_id]
        assert state.current_turn == user_turns_per_conv * 2
        assert state.completed_client_turns == user_turns_per_conv


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_wait_for_turn_ready_reliably_wakes_on_completion():
    """Test completion wakeups do not depend on timing windows."""

    async def run_one_iteration():
        mgr = ConversationManager()
        await mgr.get_or_create("conv_001")
        await mgr.mark_turn_issued("conv_001", 1)

        ready: list[bool] = []

        async def waiter(m: ConversationManager, r: list) -> None:
            r.append(await m.wait_for_turn_ready("conv_001", 3, timeout=0.5))

        waiter_task = asyncio.create_task(waiter(mgr, ready))
        await asyncio.sleep(0.005)
        await mgr.mark_turn_complete("conv_001", "Assistant response")
        await asyncio.wait_for(waiter_task, timeout=0.5)
        assert ready == [True]

    for _ in range(10):
        await run_one_iteration()
