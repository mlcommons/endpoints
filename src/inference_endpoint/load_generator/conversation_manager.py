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

"""Async conversation state management for multi-turn benchmarking."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Tracks conversation sequencing for multi-turn benchmarking.

    Maintains turn counters and asyncio conditions so the strategy can enforce
    sequential turn ordering within a conversation. Message history is NOT stored
    here — it is pre-computed in MultiTurnDataset and served via load_sample().

    Attributes:
        conversation_id: Unique identifier for this conversation.
        current_turn: Last completed turn number (0 = not started).
        pending_client_turn: Turn number of in-flight client turn (None if idle).
        expected_client_turns: Expected number of client turns (for completion tracking).
        issued_client_turns: Count of client turns issued.
        completed_client_turns: Count of client turns with responses.
        failed_client_turns: Count of client turns that failed (error/timeout).
        message_history: Accumulated message list (only populated when
            use_dataset_history=False; empty otherwise).
        condition: Per-conversation asyncio.Condition for turn-ready and turn-issued waits.
            Scoped to this conversation so that state changes only wake the single
            pipeline task waiting on this conversation, not all pipeline tasks.
    """

    conversation_id: str
    current_turn: int = 0
    pending_client_turn: int | None = None
    expected_client_turns: int | None = None
    issued_client_turns: int = 0
    completed_client_turns: int = 0
    failed_client_turns: int = 0
    message_history: list[dict[str, Any]] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    def add_client_turn(self, turn: int, message: dict[str, Any] | None = None):
        """Record that a client turn has been issued (updates sequencing counters).

        Args:
            turn: Turn number for this client message.
            message: Message dict to append to message_history (only used when
                use_dataset_history=False).
        """
        self.pending_client_turn = turn
        self.issued_client_turns += 1
        if message is not None:
            self.message_history.append(message)

    def add_assistant_turn(self, content: str | None = None):
        """Record assistant response and mark turn complete (success).

        Args:
            content: Response content to append to message_history. Only
                used when use_dataset_history=False; None means no history
                update (pre-built messages path).
        """
        if content is not None:
            self.message_history.append({"role": "assistant", "content": content})
        if self.pending_client_turn is not None:
            self.current_turn = self.pending_client_turn + 1
            self.pending_client_turn = None
            self.completed_client_turns += 1
        elif self.is_complete():
            pass
        else:
            logger.warning(
                f"Received assistant response for {self.conversation_id} "
                f"with no pending client turn (duplicate or out-of-order response)"
            )
            self.current_turn = self.current_turn + 1 if self.current_turn > 0 else 1
            self.completed_client_turns += 1

        if self.is_complete():
            if self.failed_client_turns > 0:
                logger.info(
                    f"Conversation {self.conversation_id} completed with failures: "
                    f"{self.completed_client_turns - self.failed_client_turns}/"
                    f"{self.expected_client_turns} successful, "
                    f"{self.failed_client_turns} failed"
                )
            else:
                logger.debug(
                    f"Conversation {self.conversation_id} completed: "
                    f"{self.completed_client_turns}/{self.expected_client_turns} turns"
                )

    def mark_turn_failed(self, store_in_history: bool = False):
        """Mark turn as failed (error/timeout) - still counts as completed for sequencing."""
        if self.pending_client_turn is not None:
            self.current_turn = self.pending_client_turn + 1
            self.pending_client_turn = None
            self.completed_client_turns += 1
            self.failed_client_turns += 1

            if store_in_history:
                self.message_history.append(
                    {
                        "role": "assistant",
                        "content": "[ERROR: Turn failed or timed out]",
                    }
                )

            logger.warning(
                f"Turn {self.current_turn - 1} failed for conversation {self.conversation_id}"
            )
        else:
            logger.warning(
                f"Attempted to mark failed turn for {self.conversation_id} "
                f"with no pending client turn"
            )

        if self.is_complete():
            logger.info(
                f"Conversation {self.conversation_id} completed with failures: "
                f"{self.completed_client_turns - self.failed_client_turns}/"
                f"{self.expected_client_turns} successful, "
                f"{self.failed_client_turns} failed"
            )

    def is_complete(self) -> bool:
        """Check if conversation is complete (all turns issued and responses received)."""
        if self.expected_client_turns is None:
            return False
        return self.completed_client_turns >= self.expected_client_turns

    def is_ready_for_turn(self) -> bool:
        """Check if the previous turn has completed and the next may be issued."""
        return (
            self.pending_client_turn is None
            and self.issued_client_turns == self.completed_client_turns
            and self.issued_client_turns > 0
        )


class ConversationManager:
    """Manages conversation sequencing for multi-turn benchmarking.

    Async manager that tracks multiple conversations and enforces turn ordering.
    Conversations are identified by unique IDs. Message history is NOT maintained here
    — it is pre-computed in MultiTurnDataset and passed directly to each request.

    The manager ensures that:
    - Turn N+1 cannot be issued until turn N completes
    - Concurrent access to conversation state is async-safe

    Each ConversationState carries its own asyncio.Condition so that state changes
    (turn issued / turn complete) only wake the single pipeline task waiting
    on that conversation, not all pipeline tasks across all conversations.
    All conversation states are pre-created by the strategy before pipeline
    tasks start, so wait_for_turn_issued never races against get_or_create.
    """

    def __init__(self):
        """Initialize conversation manager with empty state."""
        self._conversations: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    def get_state(self, conversation_id: str) -> ConversationState | None:
        """Get conversation state without creating (for read-only access)."""
        return self._conversations.get(conversation_id)

    async def get_or_create(
        self,
        conversation_id: str,
        expected_client_turns: int | None = None,
        system_message: dict[str, Any] | None = None,
    ) -> ConversationState:
        """Get existing or create new conversation state.

        Args:
            conversation_id: Unique identifier for conversation.
            expected_client_turns: Expected number of client turns (for completion tracking).
            system_message: System message dict to pre-populate message_history with.
                Only used when use_dataset_history=False and conversation is new.

        Returns:
            ConversationState for this conversation.
        """
        async with self._lock:
            if conversation_id not in self._conversations:
                initial_history: list[dict[str, Any]] = (
                    [system_message] if system_message is not None else []
                )
                state = ConversationState(
                    conversation_id=conversation_id,
                    current_turn=0,
                    pending_client_turn=None,
                    expected_client_turns=expected_client_turns,
                    issued_client_turns=0,
                    completed_client_turns=0,
                    failed_client_turns=0,
                    message_history=initial_history,
                )
                self._conversations[conversation_id] = state
            return self._conversations[conversation_id]

    async def wait_for_turn_ready(
        self, conversation_id: str, turn: int, timeout: float | None = None
    ) -> bool:
        """Block until conversation is ready for this turn.

        Uses the per-conversation asyncio.Condition so only this conversation's pipeline
        task is woken on state changes, not all pipeline tasks.

        Args:
            conversation_id: Conversation to wait for.
            turn: Turn number to wait for (unused in readiness check; kept for
                call-site compatibility).
            timeout: Maximum seconds to wait (None = infinite).

        Returns:
            True if ready, False if timeout.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            logger.error(f"Conversation {conversation_id} not found in manager")
            raise KeyError(f"Conversation {conversation_id} not initialized")

        async with state.condition:
            if timeout is None:
                await state.condition.wait_for(state.is_ready_for_turn)
                return True
            try:
                async with asyncio.timeout(timeout):
                    await state.condition.wait_for(state.is_ready_for_turn)
                return True
            except TimeoutError:
                return state.is_ready_for_turn()

    async def wait_for_turn_issued(
        self,
        conversation_id: str,
        min_issued: int,
        timeout: float | None = None,
    ) -> bool:
        """Block until at least min_issued client turns have been issued.

        Args:
            conversation_id: Conversation to wait for.
            min_issued: Minimum number of issued turns to wait for.
            timeout: Maximum seconds to wait (None = infinite).

        Returns:
            True if condition met, False if timeout.

        Raises:
            KeyError: If conversation_id not found (programming error — state must be
                pre-created by the strategy before pipeline tasks are spawned).
        """
        state = self._conversations[conversation_id]
        predicate = lambda: state.issued_client_turns >= min_issued  # noqa: E731
        async with state.condition:
            if timeout is None:
                await state.condition.wait_for(predicate)
                return True
            try:
                async with asyncio.timeout(timeout):
                    await state.condition.wait_for(predicate)
                return True
            except TimeoutError:
                return state.issued_client_turns >= min_issued

    async def mark_turn_issued(
        self,
        conversation_id: str,
        turn: int,
        message: dict[str, Any] | None = None,
    ):
        """Mark that a client turn has been issued (updates sequencing counters).

        Args:
            conversation_id: Conversation ID.
            turn: Turn number being issued.
            message: Message dict to append to history (used when
                use_dataset_history=False).

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        async with state.condition:
            state.add_client_turn(turn, message)
            state.condition.notify_all()

    async def mark_turn_complete(
        self,
        conversation_id: str,
        response: str,
        store_in_history: bool = False,
    ):
        """Mark that assistant response has arrived.

        Args:
            conversation_id: Conversation ID.
            response: Model output (stored in history when store_in_history=True).
            store_in_history: When True, append response to message_history.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        async with state.condition:
            state.add_assistant_turn(response if store_in_history else None)
            state.condition.notify_all()

    async def mark_turn_failed(
        self, conversation_id: str, store_in_history: bool = False
    ):
        """Mark that assistant response failed (error/timeout).

        Failed turns still count toward conversation completion to ensure
        turn sequencing progresses even under errors.

        Args:
            conversation_id: Conversation ID.
            store_in_history: When True, append error placeholder to message_history.

        Raises:
            KeyError: If conversation_id not found in manager.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        async with state.condition:
            state.mark_turn_failed(store_in_history=store_in_history)
            state.condition.notify_all()
