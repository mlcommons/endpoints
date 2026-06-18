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

"""Conversation state management for agentic inference benchmarking."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Per-conversation state for agentic inference benchmarking.

    Attributes:
        conversation_id: Unique identifier for this conversation.
        completed_turns: Turns with responses (success or failure) — observability only.
        failed_turns: Turns that failed — observability only.
        expected_client_turns: Expected total client turns (for completion detection).
    """

    conversation_id: str
    completed_turns: int = 0
    failed_turns: int = 0
    expected_client_turns: int | None = None

    def is_complete(self) -> bool:
        """Return True when all expected turns have a response."""
        if self.expected_client_turns is None:
            return False
        return self.completed_turns >= self.expected_client_turns


class ConversationManager:
    """Manages per-conversation state for agentic inference benchmarking.

    All methods are synchronous. Turn sequencing is driven by AgenticInferenceStrategy
    which calls on_sample_complete() → _issue_next_turn() directly.

    All states are pre-created by AgenticInferenceStrategy.execute() before any turns
    are issued, so get_or_create() requires no locking.
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
    ) -> ConversationState:
        """Return existing state or create a new one.

        Args:
            conversation_id: Unique identifier for conversation.
            expected_client_turns: Expected number of client turns.

        Returns:
            ConversationState for this conversation.
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id,
                expected_client_turns=expected_client_turns,
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
    ) -> None:
        """Record a successful response.

        Args:
            conversation_id: Conversation ID.

        Raises:
            KeyError: If conversation_id not found.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        state.completed_turns += 1
        self._log_if_complete(state, conversation_id)

    def mark_turn_failed(
        self,
        conversation_id: str,
    ) -> None:
        """Record a failed response.

        Failed turns count toward completion so sequencing progresses under errors.

        Args:
            conversation_id: Conversation ID.

        Raises:
            KeyError: If conversation_id not found.
        """
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id} not initialized")
        state.completed_turns += 1
        state.failed_turns += 1
        logger.warning(f"Turn failed for conversation {conversation_id}")
        self._log_if_complete(state, conversation_id)
