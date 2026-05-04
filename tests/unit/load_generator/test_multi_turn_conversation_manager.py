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
import inspect

import pytest
from inference_endpoint.load_generator.conversation_manager import (
    ConversationManager,
    ConversationState,
)


@pytest.mark.unit
def test_conversation_state_initialization():
    """ConversationState initializes with correct defaults."""
    state = ConversationState(conversation_id="conv_001")

    assert state.conversation_id == "conv_001"
    assert state.message_history == []
    assert state.completed_turns == 0
    assert state.failed_turns == 0
    assert state.expected_client_turns is None


@pytest.mark.unit
def test_conversation_state_is_complete_without_expected():
    """is_complete() returns False when expected_client_turns is None."""
    state = ConversationState(conversation_id="conv_001")
    assert not state.is_complete()
    state.completed_turns = 5
    assert not state.is_complete()


@pytest.mark.unit
def test_conversation_state_is_complete_with_expected():
    """is_complete() returns True once completed_turns >= expected."""
    state = ConversationState(conversation_id="conv_001", expected_client_turns=2)
    assert not state.is_complete()
    state.completed_turns = 1
    assert not state.is_complete()
    state.completed_turns = 2
    assert state.is_complete()


@pytest.mark.unit
def test_create_is_synchronous():
    """get_or_create() must be a plain function, not a coroutine."""
    manager = ConversationManager()
    result = manager.get_or_create("conv_001")
    assert not inspect.iscoroutine(result), "get_or_create returned a coroutine"
    assert isinstance(result, ConversationState)


@pytest.mark.unit
def test_conversation_manager_get_or_create():
    """get_or_create returns the same state for the same conversation_id."""
    manager = ConversationManager()

    state1 = manager.get_or_create("conv_001")
    state2 = manager.get_or_create("conv_001")

    assert state1 is state2
    assert state1.conversation_id == "conv_001"


@pytest.mark.unit
def test_conversation_manager_multiple_conversations():
    """Manager tracks multiple conversations independently."""
    manager = ConversationManager()

    state1 = manager.get_or_create("conv_001")
    state2 = manager.get_or_create("conv_002")

    assert state1 is not state2

    manager.mark_turn_complete("conv_001", "Response to conv_001")

    assert state1.completed_turns == 1
    assert state2.completed_turns == 0


@pytest.mark.unit
def test_conversation_manager_mark_turn_complete():
    """mark_turn_complete increments counter and appends history."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001")

    manager.mark_turn_complete("conv_001", "Assistant response")

    assert state.completed_turns == 1
    assert state.failed_turns == 0
    assert state.message_history == []  # store_in_history=False by default


@pytest.mark.unit
def test_conversation_manager_mark_turn_complete_stores_history():
    """mark_turn_complete appends to history when store_in_history=True."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001")

    manager.mark_turn_complete("conv_001", "Hello", store_in_history=True)

    assert state.message_history == [{"role": "assistant", "content": "Hello"}]


@pytest.mark.unit
def test_conversation_manager_mark_turn_failed():
    """mark_turn_failed increments both counters."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", expected_client_turns=2)

    manager.mark_turn_failed("conv_001")

    assert state.completed_turns == 1
    assert state.failed_turns == 1


@pytest.mark.unit
def test_conversation_completion_tracking():
    """is_complete() returns True after all expected turns receive responses."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", expected_client_turns=2)

    assert not state.is_complete()
    manager.mark_turn_complete("conv_001", "r1")
    assert not state.is_complete()
    manager.mark_turn_complete("conv_001", "r2")
    assert state.is_complete()


@pytest.mark.unit
def test_conversation_completion_without_expected_turns():
    """Completion is never True when expected_client_turns is None."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", expected_client_turns=None)

    manager.mark_turn_complete("conv_001", "r1")

    assert not state.is_complete()


@pytest.mark.unit
def test_conversation_completion_with_failures():
    """Conversations complete even when some turns fail."""
    manager = ConversationManager()
    state = manager.get_or_create("conv1", expected_client_turns=3)

    manager.mark_turn_complete("conv1", "Hi")
    assert not state.is_complete()

    manager.mark_turn_failed("conv1")
    assert not state.is_complete()

    manager.mark_turn_complete("conv1", "Bye")
    assert state.is_complete()
    assert state.failed_turns == 1
    assert state.completed_turns == 3


@pytest.mark.unit
def test_all_turns_fail():
    """Conversation completes when all turns fail."""
    manager = ConversationManager()
    state = manager.get_or_create("conv1", expected_client_turns=2)

    manager.mark_turn_failed("conv1")
    manager.mark_turn_failed("conv1")

    assert state.is_complete()
    assert state.completed_turns == 2
    assert state.failed_turns == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_conversation_manager_concurrent_access():
    """Concurrent pipeline tasks on independent conversations complete without errors."""
    manager = ConversationManager()
    num_conversations = 10
    turns_per_conv = 5

    for i in range(num_conversations):
        manager.get_or_create(f"conv_{i:03d}", expected_client_turns=turns_per_conv)

    errors = []

    async def process_conversation(conv_id: str):
        try:
            state = manager.get_state(conv_id)
            assert state is not None
            for _ in range(turns_per_conv):
                manager.mark_turn_complete(conv_id, "response")
                await asyncio.sleep(0.001)
        except Exception as e:
            errors.append(f"{conv_id} error: {e}")

    tasks = [
        asyncio.create_task(process_conversation(f"conv_{i:03d}"))
        for i in range(num_conversations)
    ]
    await asyncio.gather(*tasks)

    assert not errors
    for i in range(num_conversations):
        state = manager._conversations[f"conv_{i:03d}"]
        assert state.completed_turns == turns_per_conv
        assert state.is_complete()
