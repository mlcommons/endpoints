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

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Per-conversation state for multi-turn benchmarking.

    The pipeline task awaits ``turn_done`` between turns; ``mark_turn_complete``
    and ``mark_turn_failed`` set it synchronously from ``on_sample_complete``.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        turn_done: Event set when a response arrives. Pipeline waits, then clears
            it before issuing the next turn.
        message_history: Accumulated message list (populated only when
            use_dataset_history=False; empty otherwise).
        completed_turns: Turns with responses (success or failure) — observability only.
        failed_turns: Turns that failed — observability only.
        expected_client_turns: Expected total client turns (for completion detection).
    """

    conversation_id: str
    # Python 3.12+: asyncio.Event no longer requires a running loop at construction.
    turn_done: asyncio.Event = field(default_factory=asyncio.Event)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    completed_turns: int = 0
    failed_turns: int = 0
    expected_client_turns: int | None = None

    def is_complete(self) -> bool:
        """Return True when all expected turns have a response."""
        if self.expected_client_turns is None:
            return False
        return self.completed_turns >= self.expected_client_turns


class ConversationManager:
    """Manages per-conversation state for multi-turn benchmarking.

    All methods are synchronous. The pipeline task uses ``ConversationState.turn_done``
    directly for turn-done notification — no locks or condition variables needed.

    All states are pre-created by ``MultiTurnStrategy.execute()`` before any pipeline
    task starts, so ``get_or_create()`` requires no locking.
    """

    def __init__(self):
        """Initialize with empty state."""
        self._conversations: dict[str, ConversationState] = {}

    def get_state(self, conversation_id: str) -> ConversationState | None:
        """Return existing state without creating (read-only access)."""
        return self._conversations.get(conversation_id)

    def get_or_create(
        self,
        conversation_id: str,
        expected_client_turns: int | None = None,
        system_message: dict[str, Any] | None = None,
    ) -> ConversationState:
        """Return existing state or create a new one.

        Args:
            conversation_id: Unique identifier for conversation.
            expected_client_turns: Expected number of client turns.
            system_message: System message to prepend to message_history
                (only used when use_dataset_history=False and state is new).

        Returns:
            ConversationState for this conversation.
        """
        if conversation_id not in self._conversations:
            initial_history: list[dict[str, Any]] = (
                [system_message] if system_message is not None else []
            )
            self._conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                expected_client_turns=expected_client_turns,
                message_history=initial_history,
            )
        return self._conversations[conversation_id]

    def _log_if_complete(self, state: ConversationState, conversation_id: str) -> None:
        """Log completion status once all expected turns have a response."""
        if not state.is_complete():
            return
        if state.failed_turns > 0:
            logger.info(
                f"Conversation {conversation_id} completed with failures: "
                f"{state.completed_turns - state.failed_turns}/"
                f"{state.expected_client_turns} successful, "
                f"{state.failed_turns} failed"
            )
        else:
            logger.debug(
                f"Conversation {conversation_id} completed: "
                f"{state.completed_turns}/{state.expected_client_turns} turns"
            )

    def mark_turn_complete(
        self,
        conversation_id: str,
        response: str,
        store_in_history: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a successful response and wake the pipeline task.

        Args:
            conversation_id: Conversation ID.
            response: Model output (appended to history when store_in_history=True).
            store_in_history: When True, append response to message_history.
            metadata: Optional response metadata; tool_calls are preserved in history
                when present (only used when store_in_history=True).

        Raises:
            KeyError: If conversation_id not found.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        if store_in_history:
            tool_calls = metadata.get("tool_calls") if metadata else None
            if response or tool_calls:
                msg: dict[str, Any] = {"role": "assistant", "content": response or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                state.message_history.append(msg)
        state.completed_turns += 1
        self._log_if_complete(state, conversation_id)
        state.turn_done.set()

    def mark_turn_failed(
        self,
        conversation_id: str,
        store_in_history: bool = False,
    ) -> None:
        """Record a failed response and wake the pipeline task.

        Failed turns count toward completion so sequencing progresses under errors.

        Args:
            conversation_id: Conversation ID.
            store_in_history: When True, append error placeholder to message_history.

        Raises:
            KeyError: If conversation_id not found.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        if store_in_history:
            state.message_history.append(
                {"role": "assistant", "content": "[ERROR: Turn failed or timed out]"}
            )
        state.completed_turns += 1
        state.failed_turns += 1
        logger.warning(f"Turn failed for conversation {conversation_id}")
        self._log_if_complete(state, conversation_id)
        state.turn_done.set()
