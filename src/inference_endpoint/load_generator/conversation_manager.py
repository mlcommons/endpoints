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
        expected_user_turns: Expected number of user turns (for completion tracking).
        issued_user_turns: Count of user turns issued.
        completed_user_turns: Count of user turns with assistant responses.
        failed_user_turns: Count of user turns that failed (error/timeout).
        conversation_complete_event: Threading event to signal conversation completion.
    """

    conversation_id: str
    current_turn: int = 0
    message_history: list[dict[str, str]] = field(default_factory=list)
    pending_user_turn: int | None = None
    system_prompt: str | None = None
    turn_complete_event: threading.Event = field(default_factory=threading.Event)
    expected_user_turns: int | None = None
    issued_user_turns: int = 0
    completed_user_turns: int = 0
    failed_user_turns: int = 0
    conversation_complete_event: threading.Event = field(
        default_factory=threading.Event
    )

    def add_user_turn(self, turn: int, content: str):
        """Add user message and mark as pending.

        Args:
            turn: Turn number for this user message.
            content: User message content.
        """
        self.message_history.append({"role": "user", "content": content})
        self.pending_user_turn = turn
        self.issued_user_turns += 1

    def add_assistant_turn(
        self, model_response: str, history_content: str | None = None
    ):
        """Add assistant response and mark turn complete (success).

        Args:
            model_response: Actual model output (used for metrics/timing).
            history_content: Content to append to message history. When provided
                (e.g. dataset reference response), this is used instead of
                model_response so history is deterministic across runs.
        """
        content = history_content if history_content is not None else model_response
        self.message_history.append({"role": "assistant", "content": content})
        # After assistant responds to turn N, conversation is at turn N+1
        # (e.g., after user turn 1 + assistant turn 2, we're ready for turn 3)
        if self.pending_user_turn is not None:
            self.current_turn = self.pending_user_turn + 1
            self.pending_user_turn = None
            self.completed_user_turns += 1
        else:
            # Handle duplicate/orphaned response (response without pending user turn)
            logger.warning(
                f"Received assistant response for {self.conversation_id} "
                f"with no pending user turn (duplicate or out-of-order response)"
            )
            # Increment from current position or start at 1 if no turns yet
            self.current_turn = self.current_turn + 1 if self.current_turn > 0 else 1

        self.turn_complete_event.set()

        # Check if conversation is now complete
        if self.is_complete():
            self.conversation_complete_event.set()
            if self.failed_user_turns > 0:
                logger.info(
                    f"Conversation {self.conversation_id} completed with failures: "
                    f"{self.completed_user_turns - self.failed_user_turns}/"
                    f"{self.expected_user_turns} successful, "
                    f"{self.failed_user_turns} failed"
                )
            else:
                logger.debug(
                    f"Conversation {self.conversation_id} completed: "
                    f"{self.completed_user_turns}/{self.expected_user_turns} turns"
                )

    def mark_turn_failed(self):
        """Mark turn as failed (error/timeout) - still counts as completed for sequencing.

        Failed turns count toward conversation completion to ensure sequential mode
        progresses even when turns fail. An error placeholder is added to message
        history to maintain context for subsequent turns.
        """
        if self.pending_user_turn is not None:
            self.current_turn = self.pending_user_turn + 1
            self.pending_user_turn = None
            self.completed_user_turns += 1
            self.failed_user_turns += 1

            # Add placeholder to message history for future turn context
            self.message_history.append(
                {"role": "assistant", "content": "[ERROR: Turn failed or timed out]"}
            )

            logger.warning(
                f"Turn {self.current_turn - 1} failed for conversation {self.conversation_id}"
            )
        else:
            logger.warning(
                f"Attempted to mark failed turn for {self.conversation_id} "
                f"with no pending user turn"
            )

        self.turn_complete_event.set()

        # Check if conversation is now complete
        if self.is_complete():
            self.conversation_complete_event.set()
            logger.info(
                f"Conversation {self.conversation_id} completed with failures: "
                f"{self.completed_user_turns - self.failed_user_turns}/"
                f"{self.expected_user_turns} successful, "
                f"{self.failed_user_turns} failed"
            )

    def is_complete(self) -> bool:
        """Check if conversation is complete (all turns issued and responses received).

        Returns:
            True if conversation is complete, False otherwise.
        """
        if self.expected_user_turns is None:
            return False  # Unknown completion, can't determine
        return self.completed_user_turns >= self.expected_user_turns

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
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def get_or_create(
        self,
        conversation_id: str,
        system_prompt: str | None,
        expected_user_turns: int | None = None,
    ) -> ConversationState:
        """Get existing or create new conversation state.

        Args:
            conversation_id: Unique identifier for conversation.
            system_prompt: Optional system prompt to initialize conversation.
            expected_user_turns: Expected number of user turns (for completion tracking).

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
                    expected_user_turns=expected_user_turns,
                    issued_user_turns=0,
                    completed_user_turns=0,
                    failed_user_turns=0,
                    conversation_complete_event=threading.Event(),
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
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            state = self._conversations.get(conversation_id)
            if state is None:
                logger.error(f"Conversation {conversation_id} not found in manager")
                raise KeyError(f"Conversation {conversation_id} not initialized")

            while not state.is_ready_for_turn(turn):
                if deadline is not None:
                    remaining_timeout = deadline - time.monotonic()
                    if remaining_timeout <= 0:
                        return state.is_ready_for_turn(turn)
                    remaining_timeout = max(MIN_TIMEOUT_SECONDS, remaining_timeout)
                else:
                    remaining_timeout = None

                self._condition.wait(timeout=remaining_timeout)

            return True

    def wait_for_conversation_complete(
        self, conversation_id: str, timeout: float | None = None
    ) -> bool:
        """Block until conversation is complete (all turns issued and responses received).

        Args:
            conversation_id: Conversation to wait for.
            timeout: Maximum time to wait in seconds (None = infinite).

        Returns:
            True if conversation completed, False if timeout.
        """
        if conversation_id not in self._conversations:
            logger.warning(
                f"Cannot wait for unknown conversation {conversation_id}, returning True"
            )
            return True  # Don't block on unknown conversations

        state = self._conversations[conversation_id]

        # Check if already complete
        if state.is_complete():
            return True

        # Handle unknown expected_user_turns (can't determine completion)
        if state.expected_user_turns is None:
            logger.warning(
                f"Conversation {conversation_id} has no expected_user_turns, "
                "cannot wait for completion"
            )
            return True  # Don't block if we can't determine completion

        # Wait for completion event with timeout
        start_time = time.monotonic()
        while not state.is_complete():
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                remaining_timeout = max(MIN_TIMEOUT_SECONDS, timeout - elapsed)
                if elapsed >= timeout:
                    logger.warning(
                        f"Timeout waiting for conversation {conversation_id} to complete: "
                        f"{state.completed_user_turns}/{state.expected_user_turns} turns"
                    )
                    return False

            if not state.conversation_complete_event.wait(timeout=remaining_timeout):
                return False  # Timeout

            # Re-check in case of spurious wakeup
            if state.is_complete():
                return True

            # Clear event for next wait
            state.conversation_complete_event.clear()

        return True

    def mark_turn_issued(self, conversation_id: str, turn: int, content: str):
        """Mark that user turn has been issued.

        Args:
            conversation_id: Conversation ID.
            turn: Turn number being issued.
            content: User message content.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._condition:
            state = self._conversations.get(conversation_id)
            if state is None:
                raise KeyError(f"Conversation {conversation_id} not initialized")
            state.add_user_turn(turn, content)

    def mark_turn_complete(
        self,
        conversation_id: str,
        response: str,
        history_content: str | None = None,
    ):
        """Mark that assistant response has arrived.

        Args:
            conversation_id: Conversation ID.
            response: Actual model output (used to trigger completion/metrics).
            history_content: Content to store in message history. When provided
                (e.g. dataset reference response), this is used instead of
                response so history is deterministic across runs.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._condition:
            state = self._conversations.get(conversation_id)
            if state is None:
                raise KeyError(f"Conversation {conversation_id} not initialized")
            state.add_assistant_turn(response, history_content)
            self._condition.notify_all()

    def mark_turn_failed(self, conversation_id: str):
        """Mark that assistant response failed (error/timeout).

        Failed turns still count toward conversation completion to ensure
        sequential mode progresses even under errors.

        Args:
            conversation_id: Conversation ID.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        with self._condition:
            state = self._conversations.get(conversation_id)
            if state is None:
                raise KeyError(f"Conversation {conversation_id} not initialized")
            state.mark_turn_failed()
            self._condition.notify_all()
