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
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any

from ..config.schema import MultiTurnConfig
from ..core.types import QueryResult
from .conversation_manager import ConversationManager, ConversationState
from .strategy import PhaseIssuerProtocol

logger = logging.getLogger(__name__)

# Default turn timeout when no MultiTurnConfig is provided.
_DEFAULT_TURN_TIMEOUT_S = 300.0


class MultiTurnStrategy:
    """Event-driven multi-turn strategy. Completion of each turn triggers the next.

    execute() seeds the first N conversations (issues turn 1 for each), then
    awaits _all_done. on_sample_complete() is called synchronously from the
    receive coroutine for each response — it issues the next turn immediately
    (zero event-loop iterations between response and next issuance), or starts
    a new conversation when the current one finishes all turns.

    At most target_concurrency conversations are active simultaneously. When
    target_concurrency is None, all conversations start at once.

    Integration with BenchmarkSession:
    - execute(): seeds conversations, awaits completion
    - on_query_complete(): no-op (required by LoadStrategy protocol)
    - on_sample_complete(): routes completed QueryResult, issues next turn

    The response routing path:
    1. _issue_next_turn issues turn N via phase_issuer.issue(idx) → query_id
    2. _issue_next_turn stores conv_id in _inflight[query_id]
    3. BenchmarkSession calls on_sample_complete(result) with the QueryResult
    4. on_sample_complete looks up conv_id from _inflight, calls mark_turn_complete
    5. on_sample_complete calls _issue_next_turn for turn N+1 (synchronously)
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
            target_concurrency: Maximum number of simultaneously active conversations.
                None means all conversations run concurrently.
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
        self._store_in_history = (
            not multi_turn_config.use_dataset_history
            if multi_turn_config is not None
            else False
        )

        # Maps query_id -> conversation_id for routing completions.
        self._inflight: dict[str, str] = {}
        # Cached ConversationState refs for O(1) lookup in on_sample_complete.
        self._conv_states: dict[str, ConversationState] = {}

        # Event-driven state — populated in execute().
        self._pending_convs: deque[tuple[str, list[tuple[int, int]]]] = deque()
        self._active_iters: dict[str, Iterator[tuple[int, int]]] = {}
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}
        self._error: BaseException | None = None
        self._all_done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._phase_issuer: PhaseIssuerProtocol | None = None

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        """Drive multi-turn sample issuance.

        Args:
            phase_issuer: Interface for issuing samples to the endpoint.

        Returns:
            Total count of samples issued.
        """
        self._phase_issuer = phase_issuer
        self._loop = asyncio.get_running_loop()
        self._all_done = asyncio.Event()
        self._error = None

        conv_samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for sample_meta in self._dataset_metadata["samples"]:
            conv_id = sample_meta["conversation_id"]
            conv_samples[conv_id].append(
                (sample_meta["sample_index"], sample_meta["turn"])
            )

        # Pre-create all conversation states before issuing any turns (no locking needed).
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

        # Build pending queue (sorted turns per conversation).
        for conv_id, turns in conv_samples.items():
            self._pending_convs.append((conv_id, sorted(turns, key=lambda x: x[1])))

        n_to_start = (
            min(self._target_concurrency, len(self._pending_convs))
            if self._target_concurrency is not None and self._target_concurrency > 0
            else len(self._pending_convs)
        )
        for _ in range(n_to_start):
            self._start_conversation()

        if not self._active_iters and not self._inflight:
            return phase_issuer.issued_count

        await self._all_done.wait()

        for handle in self._timeout_handles.values():
            handle.cancel()
        self._timeout_handles.clear()

        if self._inflight:
            logger.warning(
                "%d query(ies) never received a response (session stop or transport failure): %s",
                len(self._inflight),
                list(self._inflight.keys()),
            )
            self._inflight.clear()

        if self._error is not None:
            raise self._error
        return phase_issuer.issued_count

    def _start_conversation(self) -> None:
        """Pop the next conversation from the pending queue and issue its first turn."""
        conv_id, turns = self._pending_convs.popleft()
        self._active_iters[conv_id] = iter(turns)
        self._issue_next_turn(conv_id)

    def _issue_next_turn(self, conv_id: str) -> None:
        """Issue the next turn for conv_id, or mark the conversation done."""
        it = self._active_iters.get(conv_id)
        if it is None:
            return

        pair = next(it, None)
        if pair is None:
            del self._active_iters[conv_id]
            self._fill_slot()
            return

        idx, turn = pair
        state = self._conv_states[conv_id]

        data_override: dict[str, Any] | None = None
        current_turn_messages: list[dict[str, Any]] | None = None
        if self._store_in_history:
            current_turn_messages = self._dataset_metadata.get(
                "current_turn_messages_by_key", {}
            ).get((conv_id, turn))
            if current_turn_messages:
                has_tool_msg = any(
                    m.get("role") == "tool" for m in current_turn_messages
                )
                if has_tool_msg:
                    logger.warning(
                        "Live-history mode with tool messages uses dataset "
                        "tool_call_ids; real endpoint IDs will differ "
                        "(conv=%s, turn=%d)",
                        conv_id,
                        turn,
                    )
                live_messages = state.message_history.copy() + current_turn_messages
                data_override = {"messages": live_messages}

        assert self._phase_issuer is not None
        query_id = self._phase_issuer.issue(idx, data_override=data_override)
        if query_id is None:
            # Session stopping — signal done.
            assert self._all_done is not None
            self._all_done.set()
            return

        self._inflight[query_id] = conv_id

        if self._store_in_history and current_turn_messages:
            state.message_history.extend(current_turn_messages)

        assert self._loop is not None
        handle = self._loop.call_later(
            self._turn_timeout_s, self._handle_timeout, query_id, conv_id
        )
        self._timeout_handles[query_id] = handle

    def _fill_slot(self) -> None:
        """Start a new conversation from the pending queue, or signal all done."""
        if self._pending_convs:
            self._start_conversation()
        elif not self._active_iters:
            assert self._all_done is not None
            self._all_done.set()

    def _handle_timeout(self, query_id: str, conv_id: str) -> None:
        """Called by the event loop when a turn response does not arrive in time."""
        if self._inflight.pop(query_id, None) is None:
            return
        self._timeout_handles.pop(query_id, None)

        logger.warning(
            "Turn timed out for conversation %s (query=%s)", conv_id, query_id
        )

        self._conv_manager.mark_turn_failed(
            conv_id, store_in_history=self._store_in_history
        )
        it = self._active_iters.pop(conv_id, None)
        if it is not None:
            for _ in it:
                self._conv_manager.mark_turn_failed(
                    conv_id, store_in_history=self._store_in_history
                )

        self._fill_slot()

    def on_query_complete(self, query_id: str) -> None:
        """No-op. Required by LoadStrategy protocol; called by BenchmarkSession."""
        pass

    def on_sample_complete(self, result: QueryResult) -> None:
        """Route completed QueryResult to ConversationManager and issue next turn.

        Called synchronously from BenchmarkSession._handle_response(). Issues the
        next turn immediately (zero event-loop delay) or starts a new conversation
        when this one finishes all turns.

        Args:
            result: Completed QueryResult from the endpoint.
        """
        conv_id = self._inflight.pop(result.id, None)
        if conv_id is None:
            return

        handle = self._timeout_handles.pop(result.id, None)
        if handle is not None:
            handle.cancel()

        response_text = result.get_response_output_string()

        try:
            if result.error is not None:
                self._conv_manager.mark_turn_failed(
                    conv_id, store_in_history=self._store_in_history
                )
            else:
                self._conv_manager.mark_turn_complete(
                    conv_id,
                    response_text,
                    store_in_history=self._store_in_history,
                    metadata=result.metadata,
                )
        except KeyError:
            logger.warning(
                "on_sample_complete: conversation %s not found in manager (result=%s)",
                conv_id,
                result.id,
            )
            return

        try:
            self._issue_next_turn(conv_id)
        except Exception as exc:
            logger.error("Error issuing next turn for %s: %s", conv_id, exc)
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()
