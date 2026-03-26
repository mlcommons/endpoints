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

"""Conversation state management for multi-turn benchmarking."""

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Minimum timeout to prevent busy-waiting
MIN_TIMEOUT_SECONDS = 0.001


@dataclass
class ConversationState:
    """Tracks conversation progress and history for multi-turn benchmarking.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        current_turn: Last completed turn number (0 = not started).
        message_history: Full OpenAI-style messages array for conversation history.
        pending_user_turn: Turn number of in-flight user message (None if idle).
        system_prompt: Optional system prompt for conversation.
        turn_complete_event: Threading event to signal turn completion.
    """

    conversation_id: str
    current_turn: int = 0
    message_history: list[dict[str, str]] = field(default_factory=list)
    pending_user_turn: int | None = None
    system_prompt: str | None = None
    turn_complete_event: threading.Event = field(default_factory=threading.Event)

    def add_user_turn(self, turn: int, content: str):
        """Add user message and mark as pending.

        Args:
            turn: Turn number for this user message.
            content: User message content.
        """
        self.message_history.append({"role": "user", "content": content})
        self.pending_user_turn = turn

    def add_assistant_turn(self, content: str):
        """Add assistant response and mark turn complete.

        Args:
            content: Assistant response content.
        """
        self.message_history.append({"role": "assistant", "content": content})
        # After assistant responds to turn N, conversation is at turn N+1
        # (e.g., after user turn 1 + assistant turn 2, we're ready for turn 3)
        self.current_turn = self.pending_user_turn + 1
        self.pending_user_turn = None
        self.turn_complete_event.set()

    def is_ready_for_turn(self, turn: int) -> bool:
        """Check if ready to issue this turn (previous turn must be complete).

        Args:
            turn: Turn number to check.

        Returns:
            True if ready to issue this turn, False otherwise.
        """
        return turn == self.current_turn + 1 and self.pending_user_turn is None


class ConversationManager:
    """Manages conversation state and turn sequencing for multi-turn benchmarking.

    Thread-safe manager that tracks multiple conversations and enforces turn ordering.
    Conversations are identified by unique IDs and maintain message history across turns.

    The manager ensures that:
    - Turn N+1 cannot be issued until turn N completes
    - Message history is maintained across turns
    - Concurrent access to conversation state is thread-safe
    """

    def __init__(self):
        """Initialize conversation manager with empty state."""
        self._conversations: dict[str, ConversationState] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self, conversation_id: str, system_prompt: str | None
    ) -> ConversationState:
        """Get existing or create new conversation state.

        Args:
            conversation_id: Unique identifier for conversation.
            system_prompt: Optional system prompt to initialize conversation.

        Returns:
            ConversationState for this conversation.
        """
        with self._lock:
            if conversation_id not in self._conversations:
                state = ConversationState(
                    conversation_id=conversation_id,
                    current_turn=0,
                    message_history=[],
                    pending_user_turn=None,
                    system_prompt=system_prompt,
                    turn_complete_event=threading.Event(),
                )
                if system_prompt:
                    state.message_history.append(
                        {"role": "system", "content": system_prompt}
                    )
                self._conversations[conversation_id] = state
            return self._conversations[conversation_id]

    def wait_for_turn_ready(
        self, conversation_id: str, turn: int, timeout: float | None = None
    ) -> bool:
        """Block until conversation is ready for this turn.

        Args:
            conversation_id: Conversation to wait for.
            turn: Turn number to wait for.
            timeout: Maximum seconds to wait (None = infinite).

        Returns:
            True if ready, False if timeout.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._lock:
            state = self._conversations.get(conversation_id)
            if state is None:
                logger.error(f"Conversation {conversation_id} not found in manager")
                raise KeyError(f"Conversation {conversation_id} not initialized")

        start_time = time.time()

        while True:
            with self._lock:
                if state.is_ready_for_turn(turn):
                    return True

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                remaining_timeout = max(MIN_TIMEOUT_SECONDS, timeout - elapsed)
            else:
                remaining_timeout = None

            state.turn_complete_event.clear()
            if not state.turn_complete_event.wait(timeout=remaining_timeout):
                return False

    def mark_turn_issued(self, conversation_id: str, turn: int, content: str):
        """Mark that user turn has been issued.

        Args:
            conversation_id: Conversation ID.
            turn: Turn number being issued.
            content: User message content.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._lock:
            state = self._conversations.get(conversation_id)
            if state is None:
                raise KeyError(f"Conversation {conversation_id} not initialized")

        state.add_user_turn(turn, content)

    def mark_turn_complete(self, conversation_id: str, response: str):
        """Mark that assistant response has arrived.

        Args:
            conversation_id: Conversation ID.
            response: Assistant response content.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._lock:
            state = self._conversations.get(conversation_id)
            if state is None:
                raise KeyError(f"Conversation {conversation_id} not initialized")

        state.add_assistant_turn(response)
