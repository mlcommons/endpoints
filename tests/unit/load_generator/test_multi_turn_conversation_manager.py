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

import threading
import time

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
    assert state.message_history == []
    assert state.pending_user_turn is None
    assert state.system_prompt is None
    assert isinstance(state.turn_complete_event, threading.Event)
    assert not state.turn_complete_event.is_set()


@pytest.mark.unit
def test_conversation_state_add_user_turn():
    """Test adding user turn updates state correctly."""
    state = ConversationState(conversation_id="conv_001")

    state.add_user_turn(1, "Hello, how are you?")

    assert state.pending_user_turn == 1
    assert len(state.message_history) == 1
    assert state.message_history[0] == {
        "role": "user",
        "content": "Hello, how are you?",
    }
    assert state.current_turn == 0  # Not incremented until assistant response


@pytest.mark.unit
def test_conversation_state_add_assistant_turn():
    """Test adding assistant turn completes turn cycle.

    Turn numbering follows absolute conversation position:
    - Turn 1: user message
    - Turn 2: assistant response (completes after turn 1)
    """
    state = ConversationState(conversation_id="conv_001")

    state.add_user_turn(1, "Hello")
    state.add_assistant_turn("Hi there!")

    # After turn 1 (user) + turn 2 (assistant), current_turn is 2
    assert state.current_turn == 2
    assert state.pending_user_turn is None
    assert len(state.message_history) == 2
    assert state.message_history[1] == {"role": "assistant", "content": "Hi there!"}
    assert state.turn_complete_event.is_set()


@pytest.mark.unit
def test_conversation_state_is_ready_for_turn():
    """Test turn readiness checks.

    Turn numbering follows absolute conversation position:
    - Turn 1: user (issued first)
    - Turn 2: assistant (response to turn 1, not issued by us)
    - Turn 3: user (issued after turn 2 completes)
    """
    state = ConversationState(conversation_id="conv_001")

    # Turn 1 should be ready (current_turn=0, no pending)
    assert state.is_ready_for_turn(1)
    assert not state.is_ready_for_turn(2)

    # Issue turn 1 (user)
    state.add_user_turn(1, "Hello")
    assert not state.is_ready_for_turn(1)  # Turn 1 now pending
    assert not state.is_ready_for_turn(3)  # Turn 3 not ready yet (need turn 2 first)

    # Complete turn 1 with assistant response (turn 2)
    state.add_assistant_turn("Hi")
    assert not state.is_ready_for_turn(1)  # Turn 1 already complete
    assert state.is_ready_for_turn(3)  # Turn 3 now ready (current_turn=2, so 3==2+1)
    assert not state.is_ready_for_turn(5)  # Turn 5 not ready yet


@pytest.mark.unit
def test_conversation_state_multi_turn_sequence():
    """Test multi-turn conversation flow.

    Turn numbering follows absolute conversation position:
    - Turn 1, 3, 5: user messages
    - Turn 2, 4, 6: assistant responses
    """
    state = ConversationState(conversation_id="conv_001")

    # Turn 1 (user) + Turn 2 (assistant)
    state.add_user_turn(1, "User message 1")
    state.add_assistant_turn("Assistant response 1")
    assert state.current_turn == 2

    # Turn 3 (user) + Turn 4 (assistant)
    state.add_user_turn(3, "User message 2")
    state.add_assistant_turn("Assistant response 2")
    assert state.current_turn == 4

    # Turn 5 (user) + Turn 6 (assistant)
    state.add_user_turn(5, "User message 3")
    state.add_assistant_turn("Assistant response 3")
    assert state.current_turn == 6

    # Verify full history
    assert len(state.message_history) == 6
    assert state.message_history[0]["role"] == "user"
    assert state.message_history[1]["role"] == "assistant"
    assert state.message_history[2]["role"] == "user"
    assert state.message_history[3]["role"] == "assistant"


@pytest.mark.unit
def test_conversation_manager_get_or_create():
    """Test get_or_create returns same state for same conversation_id."""
    manager = ConversationManager()

    state1 = manager.get_or_create("conv_001", None)
    state2 = manager.get_or_create("conv_001", None)

    assert state1 is state2
    assert state1.conversation_id == "conv_001"


@pytest.mark.unit
def test_conversation_manager_get_or_create_with_system_prompt():
    """Test get_or_create initializes system prompt."""
    manager = ConversationManager()

    state = manager.get_or_create("conv_001", "You are a helpful assistant")

    assert state.system_prompt == "You are a helpful assistant"
    assert len(state.message_history) == 1
    assert state.message_history[0] == {
        "role": "system",
        "content": "You are a helpful assistant",
    }


@pytest.mark.unit
def test_conversation_manager_multiple_conversations():
    """Test manager can track multiple conversations independently."""
    manager = ConversationManager()

    state1 = manager.get_or_create("conv_001", None)
    state2 = manager.get_or_create("conv_002", None)

    assert state1 is not state2
    assert state1.conversation_id == "conv_001"
    assert state2.conversation_id == "conv_002"

    # Modify one conversation
    manager.mark_turn_issued("conv_001", 1, "Hello from conv_001")
    manager.mark_turn_complete("conv_001", "Response to conv_001")

    # Other conversation should be unaffected
    # After turn 1 (user) completes with assistant response, current_turn is 2
    assert state1.current_turn == 2
    assert state2.current_turn == 0


@pytest.mark.unit
def test_conversation_manager_mark_turn_issued():
    """Test mark_turn_issued updates conversation state."""
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", None)

    manager.mark_turn_issued("conv_001", 1, "User message")

    assert state.pending_user_turn == 1
    assert len(state.message_history) == 1
    assert state.message_history[0]["content"] == "User message"


@pytest.mark.unit
def test_conversation_manager_mark_turn_complete():
    """Test mark_turn_complete updates conversation state.

    After issuing turn 1 (user) and receiving assistant response (turn 2),
    current_turn should be 2.
    """
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", None)

    manager.mark_turn_issued("conv_001", 1, "User message")
    manager.mark_turn_complete("conv_001", "Assistant response")

    # After turn 1 (user) + turn 2 (assistant), current_turn is 2
    assert state.current_turn == 2
    assert state.pending_user_turn is None
    assert len(state.message_history) == 2
    assert state.message_history[1]["content"] == "Assistant response"


@pytest.mark.unit
def test_conversation_manager_wait_for_turn_ready_immediate():
    """Test wait_for_turn_ready returns immediately when ready."""
    manager = ConversationManager()
    manager.get_or_create("conv_001", None)

    # Turn 1 should be ready immediately
    start = time.time()
    result = manager.wait_for_turn_ready("conv_001", 1, timeout=1.0)
    elapsed = time.time() - start

    assert result is True
    assert elapsed < 0.1  # Should return almost instantly


@pytest.mark.unit
def test_conversation_manager_wait_for_turn_ready_blocking():
    """Test wait_for_turn_ready blocks until previous turn completes.

    With absolute turn numbering:
    - Turn 1: user (issued)
    - Turn 2: assistant (response, completes turn 1)
    - Turn 3: next user turn (ready after turn 2)
    """
    manager = ConversationManager()
    state = manager.get_or_create("conv_001", None)

    # Issue turn 1 (makes it pending, blocks turn 3)
    manager.mark_turn_issued("conv_001", 1, "User message")

    # Track when turn 3 becomes ready
    ready_event = threading.Event()

    def wait_for_turn_3():
        result = manager.wait_for_turn_ready("conv_001", 3, timeout=2.0)
        if result:
            ready_event.set()

    wait_thread = threading.Thread(target=wait_for_turn_3, daemon=True)
    wait_thread.start()

    # Give thread time to start waiting
    time.sleep(0.1)
    assert not ready_event.is_set()  # Should be blocked

    # Complete turn 1 with assistant response (turn 2)
    manager.mark_turn_complete("conv_001", "Assistant response")

    # Turn 3 should now be ready
    assert ready_event.wait(timeout=1.0)
    wait_thread.join(timeout=1.0)


@pytest.mark.unit
def test_conversation_manager_wait_for_turn_ready_timeout():
    """Test wait_for_turn_ready respects timeout.

    With absolute turn numbering, after issuing turn 1 (user),
    turn 3 (next user turn) should wait for turn 2 (assistant response).
    """
    manager = ConversationManager()
    manager.get_or_create("conv_001", None)

    # Issue turn 1 without completing it
    manager.mark_turn_issued("conv_001", 1, "User message")

    # Try to wait for turn 3 with short timeout (should timeout since turn 1 not complete)
    start = time.time()
    result = manager.wait_for_turn_ready("conv_001", 3, timeout=0.2)
    elapsed = time.time() - start

    assert result is False
    assert 0.15 < elapsed < 0.35  # Should timeout around 0.2s


@pytest.mark.unit
def test_conversation_manager_concurrent_access():
    """Test thread-safe concurrent access to multiple conversations.

    With absolute turn numbering, user turns are odd (1, 3, 5, 7, 9)
    and assistant responses are even (2, 4, 6, 8, 10).
    """
    manager = ConversationManager()
    num_conversations = 10
    user_turns_per_conv = 5  # 5 user turns (1, 3, 5, 7, 9)

    # Initialize conversations
    for i in range(num_conversations):
        manager.get_or_create(f"conv_{i:03d}", None)

    errors = []

    def process_conversation(conv_id: str):
        """Issue and complete turns for one conversation."""
        try:
            for user_turn_idx in range(user_turns_per_conv):
                # User turns are odd: 1, 3, 5, 7, 9, ...
                turn = user_turn_idx * 2 + 1

                # Wait for turn to be ready
                ready = manager.wait_for_turn_ready(conv_id, turn, timeout=5.0)
                if not ready:
                    errors.append(f"{conv_id} turn {turn} timeout")
                    return

                # Issue turn
                manager.mark_turn_issued(conv_id, turn, f"Message {turn}")

                # Simulate processing time
                time.sleep(0.01)

                # Complete turn (adds assistant response as turn+1)
                manager.mark_turn_complete(conv_id, f"Response {turn}")
        except Exception as e:
            errors.append(f"{conv_id} error: {e}")

    # Process all conversations concurrently
    threads = []
    for i in range(num_conversations):
        thread = threading.Thread(
            target=process_conversation, args=(f"conv_{i:03d}",), daemon=True
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join(timeout=10.0)

    # Verify no errors
    assert not errors, f"Errors occurred: {errors}"

    # Verify all conversations completed all turns
    for i in range(num_conversations):
        conv_id = f"conv_{i:03d}"
        state = manager._conversations[conv_id]
        # After 5 user turns (1,3,5,7,9) with assistant responses (2,4,6,8,10), current_turn is 10
        assert state.current_turn == user_turns_per_conv * 2
        assert len(state.message_history) == user_turns_per_conv * 2


@pytest.mark.unit
def test_conversation_manager_wait_for_turn_ready_multiple_waiters():
    """Test multiple threads can wait for same turn.

    With absolute turn numbering, after turn 1 (user) completes with
    assistant response (turn 2), turn 3 (next user turn) becomes ready.
    """
    manager = ConversationManager()
    manager.get_or_create("conv_001", None)

    # Issue turn 1
    manager.mark_turn_issued("conv_001", 1, "User message")

    # Multiple threads wait for turn 3 (next user turn)
    ready_count = 0
    ready_lock = threading.Lock()

    def wait_for_turn_3():
        nonlocal ready_count
        result = manager.wait_for_turn_ready("conv_001", 3, timeout=2.0)
        if result:
            with ready_lock:
                ready_count += 1

    threads = []
    for _ in range(5):
        thread = threading.Thread(target=wait_for_turn_3, daemon=True)
        threads.append(thread)
        thread.start()

    # Give threads time to start waiting
    time.sleep(0.1)

    # Complete turn 1 with assistant response (turn 2)
    manager.mark_turn_complete("conv_001", "Assistant response")

    # All threads should wake up
    for thread in threads:
        thread.join(timeout=1.0)

    assert ready_count == 5


@pytest.mark.unit
def test_conversation_manager_wait_for_turn_ready_reliably_wakes_on_completion():
    """Test completion wakeups do not depend on timing windows between clear/set."""
    manager = ConversationManager()
    manager.get_or_create("conv_001", None)

    for _ in range(25):
        manager.mark_turn_issued("conv_001", 1, "User message")

        start = time.time()
        ready = []

        def waiter():
            ready.append(manager.wait_for_turn_ready("conv_001", 3, timeout=0.5))

        wait_thread = threading.Thread(target=waiter, daemon=True)
        wait_thread.start()
        time.sleep(0.002)
        manager.mark_turn_complete("conv_001", "Assistant response")
        wait_thread.join(timeout=0.2)

        assert ready == [True]
        assert time.time() - start < 0.2

        manager = ConversationManager()
        manager.get_or_create("conv_001", None)


@pytest.mark.unit
def test_conversation_state_turn_complete_event_clears():
    """Test turn_complete_event is cleared after being checked."""
    state = ConversationState(conversation_id="conv_001")

    # Complete turn 1
    state.add_user_turn(1, "User message")
    state.add_assistant_turn("Assistant response")
    assert state.turn_complete_event.is_set()

    # Event should be cleared in wait_for_turn_ready loop
    # Simulate the clear that happens in the manager
    state.turn_complete_event.clear()
    assert not state.turn_complete_event.is_set()

    # Complete turn 2
    state.add_user_turn(2, "User message 2")
    state.add_assistant_turn("Assistant response 2")
    assert state.turn_complete_event.is_set()
