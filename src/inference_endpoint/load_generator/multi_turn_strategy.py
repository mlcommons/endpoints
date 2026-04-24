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

"""Async multi-turn load strategy implementing the LoadStrategy protocol."""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from ..config.schema import MultiTurnConfig
from ..core.types import QueryResult
from .conversation_manager import ConversationManager, ConversationState
from .strategy import PhaseIssuerProtocol

logger = logging.getLogger(__name__)

# Default turn timeout when no MultiTurnConfig is provided.
_DEFAULT_TURN_TIMEOUT_S = 300.0


class MultiTurnStrategy:
    """Async multi-turn strategy. Spawns per-conversation asyncio.Tasks.

    Each conversation runs as an independent asyncio.Task that enforces
    sequential turn ordering: turn N+1 cannot be issued until turn N completes.
    Conversations run concurrently — no cross-conversation synchronization.

    Optional target_concurrency limits total in-flight requests across all
    conversations using asyncio.Semaphore.

    Integration with BenchmarkSession:
    - execute(): spawns conversation tasks, awaits all to complete
    - on_query_complete(): releases semaphore slot (concurrency control only)
    - on_sample_complete(): routes completed QueryResult to ConversationManager

    The response routing path:
    1. _conv_pipeline issues turn N via phase_issuer.issue(idx) → query_id
    2. _conv_pipeline stores conv_id in _inflight[query_id]
    3. BenchmarkSession calls on_sample_complete(result) with the QueryResult
    4. on_sample_complete looks up conv_id from _inflight, calls mark_turn_complete
    5. mark_turn_complete sets state.turn_done synchronously
    6. _conv_pipeline's await asyncio.wait_for(state.turn_done.wait()) returns
    7. Pipeline clears the event and issues turn N+1
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        dataset_metadata: dict[str, Any],
        multi_turn_config: MultiTurnConfig | None = None,
        target_concurrency: int | None = None,
    ):
        """Initialize multi-turn strategy.

        Args:
            conversation_manager: Manages conversation sequencing state.
            dataset_metadata: Metadata from MultiTurnDataset (samples list).
            multi_turn_config: Multi-turn conversation configuration.
            target_concurrency: Optional maximum concurrent in-flight requests.
        """
        self._conv_manager = conversation_manager
        self._dataset_metadata = dataset_metadata
        self._multi_turn_config = multi_turn_config
        self._turn_timeout_s = (
            multi_turn_config.turn_timeout_s
            if multi_turn_config is not None
            else _DEFAULT_TURN_TIMEOUT_S
        )
        self._target_concurrency = target_concurrency
        self._sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(target_concurrency)
            if target_concurrency is not None and target_concurrency > 0
            else None
        )
        self._store_in_history = (
            not multi_turn_config.use_dataset_history
            if multi_turn_config is not None
            else False
        )

        # Maps query_id -> conversation_id for routing completions.
        self._inflight: dict[str, str] = {}
        # Cached ConversationState refs for O(1) lookup in on_sample_complete.
        self._conv_states: dict[str, ConversationState] = {}

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        """Drive multi-turn sample issuance.

        Args:
            phase_issuer: Interface for issuing samples to the endpoint.

        Returns:
            Total count of samples issued.
        """
        conv_samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for sample_index, sample_meta in enumerate(self._dataset_metadata["samples"]):
            conv_id = sample_meta["conversation_id"]
            conv_samples[conv_id].append((sample_index, sample_meta["turn"]))

        # Pre-create all conversation states before spawning tasks (no locking needed).
        sys_prompts = self._dataset_metadata.get("system_prompts_by_conv", {})
        for conv_id, turns in conv_samples.items():
            sys_content = sys_prompts.get(conv_id) if self._store_in_history else None
            system_message = (
                {"role": "system", "content": sys_content}
                if sys_content is not None
                else None
            )
            state = self._conv_manager.get_or_create(
                conv_id,
                expected_client_turns=len(turns),
                system_message=system_message,
            )
            self._conv_states[conv_id] = state

        tasks = [
            asyncio.create_task(
                self._conv_pipeline(conv_id, turns, phase_issuer),
                name=f"mt-pipeline-{conv_id}",
            )
            for conv_id, turns in conv_samples.items()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        return phase_issuer.issued_count

    async def _conv_pipeline(
        self,
        conv_id: str,
        turns: list[tuple[int, int]],
        phase_issuer: PhaseIssuerProtocol,
    ) -> None:
        """Process all turns for a single conversation sequentially.

        For each turn after the first, waits for state.turn_done before issuing
        the next. This enforces strict sequential ordering within the conversation.
        """
        state = self._conv_states[conv_id]
        sorted_turns = sorted(turns, key=lambda x: x[1])

        for i, (idx, turn) in enumerate(sorted_turns):
            if i > 0:
                try:
                    await asyncio.wait_for(
                        state.turn_done.wait(), timeout=self._turn_timeout_s
                    )
                except TimeoutError:
                    logger.warning(
                        f"Turn {turn} of {conv_id} timed out waiting for previous turn"
                    )
                    state.failed_turns += 1
                    break
                state.turn_done.clear()

            # Acquire concurrency slot before issuing.
            if self._sem is not None:
                await self._sem.acquire()

            # Live-history mode: build messages from accumulated history + current turn.
            data_override: dict[str, Any] | None = None
            current_turn_messages: list[dict[str, Any]] | None = None
            if self._store_in_history:
                current_turn_messages = self._dataset_metadata.get(
                    "current_turn_messages_by_key", {}
                ).get((conv_id, turn))
                if current_turn_messages:
                    live_messages = state.message_history.copy() + current_turn_messages
                    data_override = {"messages": live_messages}

            query_id = phase_issuer.issue(idx, data_override=data_override)
            if query_id is None:
                # Session stopping — release slot and exit.
                if self._sem is not None:
                    self._sem.release()
                break

            self._inflight[query_id] = conv_id

            # Append current-turn messages to history so the next turn sees them.
            if self._store_in_history and current_turn_messages:
                state.message_history.extend(current_turn_messages)

    def on_query_complete(self, query_id: str) -> None:
        """Called by BenchmarkSession when a QueryResult arrives.

        Releases the concurrency semaphore slot. Response routing is done
        via on_sample_complete (which receives the full QueryResult).

        Args:
            query_id: ID of the completed query.
        """
        if self._sem is not None:
            self._sem.release()

    def on_sample_complete(self, result: QueryResult) -> None:
        """Route completed QueryResult to ConversationManager.

        Called by execute.py on_sample_complete hook after each response.
        Event.set() is synchronous — the pipeline task is woken immediately
        without needing asyncio.ensure_future.

        Args:
            result: Completed QueryResult from the endpoint.
        """
        conv_id = self._inflight.pop(result.id, None)
        if conv_id is None:
            return

        response_text = result.get_response_output_string()

        if result.error is not None:
            self._conv_manager.mark_turn_failed(
                conv_id, store_in_history=self._store_in_history
            )
        else:
            self._conv_manager.mark_turn_complete(
                conv_id, response_text, store_in_history=self._store_in_history
            )
