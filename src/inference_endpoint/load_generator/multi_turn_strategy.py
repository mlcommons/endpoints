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
import hashlib
import logging
import time
from collections import defaultdict
from typing import Any

from ..config.schema import MultiTurnConfig
from ..core.record import ErrorEventType, EventRecord, SampleEventType
from ..core.types import ErrorData, QueryResult
from ..dataset_manager.multi_turn_dataset import ConversationMetadata
from ..exceptions import InputValidationError
from .conversation_manager import ConversationManager
from .strategy import PhaseIssuerProtocol

logger = logging.getLogger(__name__)

# Default turn timeout when no MultiTurnConfig is provided.
_DEFAULT_TURN_TIMEOUT_S = 86400.0

ConversationTurn = tuple[int, int]
ConversationTurns = list[ConversationTurn]
ActiveConversationState = tuple[str, ConversationTurns, int, int]
ConversationInstance = tuple[str, str, ConversationTurns, int]


class MultiTurnStrategy:
    """Event-driven multi-turn strategy. Completion of each turn triggers the next.

    execute() seeds the first N conversations (issues turn 1 for each), then
    awaits _all_done. on_sample_complete() is called synchronously from the
    receive coroutine for each response — it issues the next turn immediately
    (zero event-loop iterations between response and next issuance), or starts
    a new conversation when the current one finishes all turns.

    At most target_concurrency conversations are active simultaneously. When
    target_concurrency is None, all conversations start at once.

    A turn-level timeout aborts the remaining client turns of that conversation
    because subsequent turns depend on the timed-out response. The timed-out
    turn and all downstream turns are marked failed.

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
        dataset_metadata: ConversationMetadata,
        multi_turn_config: MultiTurnConfig | None = None,
        target_concurrency: int | None = None,
        num_trajectories_to_issue: int | None = None,
    ):
        """Initialize multi-turn strategy.

        Args:
            conversation_manager: Manages conversation sequencing state.
            dataset_metadata: ConversationMetadata from MultiTurnDataset (after load()).
            multi_turn_config: Multi-turn conversation configuration.
            target_concurrency: Maximum number of simultaneously active conversations.
                None means all conversations run concurrently.
            num_trajectories_to_issue: Number of complete conversation trajectories
                to run. Defaults to one pass over the dataset conversations.
        """
        self._conv_manager = conversation_manager
        self._dataset_metadata = dataset_metadata
        self._multi_turn_config = multi_turn_config
        self._num_trajectories_to_issue = num_trajectories_to_issue
        self._stop_issuing_on_first_user_complete = (
            multi_turn_config.stop_issuing_on_first_user_complete
            if multi_turn_config is not None
            else False
        )
        self._turn_timeout_s = (
            multi_turn_config.turn_timeout_s
            if multi_turn_config is not None
            else _DEFAULT_TURN_TIMEOUT_S
        )
        self._target_concurrency = target_concurrency
        self._enable_salt = (
            multi_turn_config.enable_salt if multi_turn_config is not None else False
        )

        # Composite on_sample_complete callback set by execute.py; used by
        # _handle_timeout to route synthetic failure results.
        self._session_on_sample_complete: Any | None = None
        self._session_publisher: Any | None = None

        # Maps query_id -> conversation_id for routing completions.
        self._inflight: dict[str, str] = {}

        # Event-driven state — populated in execute().
        self._base_convs: list[tuple[str, ConversationTurns]] = []
        self._active_iters: dict[str, ActiveConversationState] = {}
        self._timeout_handles: dict[str, asyncio.TimerHandle] = {}
        self._delay_handles: dict[str, asyncio.TimerHandle] = {}
        self._error: BaseException | None = None
        self._all_done: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._phase_issuer: PhaseIssuerProtocol | None = None
        self._stopping = False
        self._started_trajectory_count = 0
        self._performance_tracking_stopped = False

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
        self._stopping = False
        self._active_iters.clear()
        self._inflight.clear()
        self._started_trajectory_count = 0
        self._performance_tracking_stopped = False

        conv_samples: dict[str, ConversationTurns] = defaultdict(list)
        for sample_meta in self._dataset_metadata.samples:
            conv_id = sample_meta.conversation_id
            assert sample_meta.sample_index is not None
            conv_samples[conv_id].append((sample_meta.sample_index, sample_meta.turn))

        self._base_convs = [
            (conv_id, sorted(turns, key=lambda x: x[1]))
            for conv_id, turns in conv_samples.items()
        ]
        n_to_start = self._initial_conversations_to_start()
        try:
            for _ in range(n_to_start):
                self._start_conversation()

            if not self._has_work_remaining():
                return phase_issuer.issued_count

            await self._all_done.wait()
            if self._error is not None:
                raise self._error
            return phase_issuer.issued_count
        finally:
            for handle in self._timeout_handles.values():
                handle.cancel()
            self._timeout_handles.clear()
            for handle in self._delay_handles.values():
                handle.cancel()
            self._delay_handles.clear()
            if self._inflight:
                logger.warning(
                    "%d query(ies) never received a response (session stop or transport failure): %s",
                    len(self._inflight),
                    list(self._inflight.keys()),
                )
                self._inflight.clear()

    def _initial_conversations_to_start(self) -> int:
        if not self._base_convs or not self._has_trajectory_budget():
            return 0
        trajectory_budget = self._trajectory_budget()
        if self._target_concurrency is not None and self._target_concurrency > 0:
            n_to_start = self._target_concurrency
        else:
            n_to_start = len(self._base_convs)
        return min(n_to_start, trajectory_budget)

    def _trajectory_budget(self) -> int:
        if self._num_trajectories_to_issue is not None:
            return self._num_trajectories_to_issue
        return len(self._base_convs)

    def _has_trajectory_budget(self) -> bool:
        return self._started_trajectory_count < self._trajectory_budget()

    def _has_more_conversation_instances(self) -> bool:
        return bool(self._base_convs and self._has_trajectory_budget())

    def _next_conversation_instance(self) -> ConversationInstance | None:
        if not self._has_more_conversation_instances():
            return None

        source_index = self._started_trajectory_count % len(self._base_convs)
        source_id, turns = self._base_convs[source_index]
        instance_id = self._started_trajectory_count // len(self._base_convs) + 1
        logical_id = (
            source_id if instance_id == 1 else f"{source_id}__repeat_{instance_id}"
        )
        self._started_trajectory_count += 1
        return logical_id, source_id, turns, instance_id

    def _has_work_remaining(self) -> bool:
        if self._stopping:
            return bool(self._inflight or self._delay_handles)
        return bool(
            self._active_iters
            or self._inflight
            or self._delay_handles
            or self._has_more_conversation_instances()
        )

    def _start_conversation(self) -> None:
        """Pop the next conversation from the pending queue and issue its first turn."""
        if self._stopping:
            return
        instance = self._next_conversation_instance()
        if instance is None:
            self._fill_slot()
            return
        logical_id, source_id, turns, repeat_id = instance
        self._create_conversation_state(logical_id, turns)
        self._active_iters[logical_id] = (source_id, turns, 0, repeat_id)
        self._issue_next_turn(logical_id)

    def _create_conversation_state(
        self,
        logical_id: str,
        turns: ConversationTurns,
    ) -> None:
        self._conv_manager.get_or_create(
            logical_id,
            expected_client_turns=len(turns),
        )

    def _issue_next_turn(self, conv_id: str) -> None:
        """Schedule the next turn for conv_id, applying inter-turn delay if set."""
        if self._stopping:
            return
        state = self._active_iters.get(conv_id)
        if state is None:
            return

        source_id, turns, cursor, repeat_id = state
        if cursor >= len(turns):
            self._finish_conversation(conv_id)
            return

        idx, turn = turns[cursor]

        delay = 0.0
        if (
            self._multi_turn_config is not None
            and self._multi_turn_config.inject_tool_delay
        ):
            delay_map = self._dataset_metadata.delay_seconds_by_key
            delay = float(delay_map.get((source_id, turn), 0.0))

        if delay > 0.0:
            assert self._loop is not None
            handle = self._loop.call_later(
                delay, self._issue_turn_now, conv_id, idx, turn
            )
            self._delay_handles[conv_id] = handle
        else:
            self._issue_turn_now(conv_id, idx, turn)

    def _issue_turn_now(self, conv_id: str, idx: int, turn: int) -> None:
        """Issue a single turn to the phase issuer."""
        self._delay_handles.pop(conv_id, None)
        if self._stopping:
            return

        active_iter = self._active_iters.get(conv_id)
        if active_iter is None:
            return
        source_id, turns, cursor, repeat_id = active_iter
        if cursor >= len(turns):
            return
        expected_idx, expected_turn = turns[cursor]
        if expected_idx != idx or expected_turn != turn:
            logger.debug(
                "dropping stale delayed turn for conv=%s idx=%s turn=%s",
                conv_id,
                idx,
                turn,
            )
            return
        self._active_iters[conv_id] = (source_id, turns, cursor + 1, repeat_id)

        data_override = self._build_data_override(
            source_id=source_id,
            turn=turn,
            repeat_id=repeat_id,
        )

        assert self._phase_issuer is not None
        query_id = self._phase_issuer.issue(
            idx,
            data_override=data_override,
            conversation_id=conv_id,
            turn=turn,
        )
        if query_id is None:
            # Session stopping due to wall-clock limit, signal, or a generic
            # stop check. Do not synthesize failures for unissued turns.
            self._request_stop_issuing()
            return

        self._inflight[query_id] = conv_id

        assert self._loop is not None
        handle = self._loop.call_later(
            self._turn_timeout_s, self._handle_timeout, query_id, conv_id
        )
        self._timeout_handles[query_id] = handle

    def _build_data_override(
        self,
        source_id: str,
        turn: int,
        repeat_id: int,
    ) -> dict[str, Any] | None:
        if not self._enable_salt:
            return None

        messages = self._dataset_metadata.pre_built_messages_by_key.get(
            (source_id, turn)
        )
        if not messages:
            return None

        salted_messages = self._messages_with_trajectory_salt(
            messages,
            repeat_id=repeat_id,
            conversation_id=source_id,
        )
        return {
            "messages": salted_messages,
            "input_tokens": None,
            "token_ids": None,
        }

    def _messages_with_trajectory_salt(
        self,
        messages: list[dict],
        repeat_id: int,
        conversation_id: str,
    ) -> list[dict]:
        salted_messages = [dict(message) for message in messages]
        for message in salted_messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                repeat_salt = hashlib.blake2b(
                    str(repeat_id).encode("utf-8"), digest_size=2
                ).hexdigest()
                conv_salt = hashlib.blake2b(
                    conversation_id.encode("utf-8"), digest_size=2
                ).hexdigest()
                message["content"] = (
                    f"[salt: {repeat_salt}]\n\n" f"{content}\n\n" f"[salt: {conv_salt}]"
                )
                return salted_messages
        raise InputValidationError(
            "multi_turn.enable_salt requires a system prompt for every "
            f"conversation; conversation {conversation_id!r} has no system prompt"
        )

    def _fill_slot(self) -> None:
        """Start a new conversation from the pending queue, or signal all done."""
        # Errors here must not leave _all_done unset — that would hang execute().
        try:
            if self._stopping:
                self._signal_done_if_no_inflight()
                return
            if self._has_more_conversation_instances():
                self._start_conversation()
            elif not self._has_work_remaining():
                assert self._all_done is not None
                self._all_done.set()
        except Exception as exc:
            logger.exception("Error filling slot")
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()

    def _finish_conversation(self, conv_id: str) -> None:
        """Mark one trajectory done and optionally stop tracking/issuing."""
        self._active_iters.pop(conv_id, None)
        if not self._has_more_conversation_instances():
            self._stop_performance_tracking_once()
            if self._stop_issuing_on_first_user_complete:
                self._request_stop_issuing()
                return
        self._fill_slot()

    def _stop_performance_tracking_once(self) -> None:
        if self._performance_tracking_stopped:
            return
        self._performance_tracking_stopped = True
        if self._phase_issuer is None:
            return
        stop_tracking = getattr(self._phase_issuer, "stop_performance_tracking", None)
        if stop_tracking is not None:
            stop_tracking()

    def _request_stop_issuing(self) -> None:
        """Stop issuing future turns and wait only for already in-flight work."""
        if self._stopping:
            self._signal_done_if_no_inflight()
            return
        self._stopping = True
        for handle in self._delay_handles.values():
            handle.cancel()
        self._delay_handles.clear()
        self._active_iters.clear()
        self._signal_done_if_no_inflight()

    def _signal_done_if_no_inflight(self) -> None:
        if not self._inflight and self._all_done is not None:
            self._all_done.set()

    def _handle_timeout(self, query_id: str, conv_id: str) -> None:
        """Called by the event loop when a turn response does not arrive in time."""
        if self._inflight.pop(query_id, None) is None:
            return
        self._timeout_handles.pop(query_id, None)

        conv_id_str: str = ""
        turn_num: int | None = None
        if (
            self._phase_issuer is not None
            and hasattr(self._phase_issuer, "uuid_to_index")
            and query_id in self._phase_issuer.uuid_to_index  # type: ignore[attr-defined]
        ):
            self._phase_issuer.mark_inflight_complete()
            if hasattr(self._phase_issuer, "completed_uuids"):
                self._phase_issuer.completed_uuids.add(query_id)  # type: ignore[attr-defined]
            if hasattr(self._phase_issuer, "uuid_to_conv_info"):
                conv_id_str, turn_num = self._phase_issuer.uuid_to_conv_info.pop(  # type: ignore[attr-defined]
                    query_id, ("", None)
                )

        logger.warning(
            "Turn timed out for conversation %s (query=%s)", conv_id, query_id
        )

        self._conv_manager.mark_turn_failed(conv_id)

        self._publish_synthetic_failure(
            query_id,
            conv_id_str,
            turn_num,
            error_type="TurnTimeout",
            error_message=f"turn timeout after {self._turn_timeout_s}s",
        )

        dropped = self._abort_remaining_turns(
            conv_id, reason=f"prior turn timed out (query={query_id})"
        )
        if dropped:
            logger.warning(
                "turn timeout on conv=%s dropped %d remaining client turn(s)",
                conv_id,
                dropped,
            )

        self._finish_conversation(conv_id)

    def _publish_synthetic_failure(
        self,
        query_id: str,
        conv_id: str,
        turn: int | None,
        error_type: str,
        error_message: str,
    ) -> None:
        synthetic_result = QueryResult(
            id=query_id,
            error=ErrorData(error_type=error_type, error_message=error_message),
        )
        if self._session_publisher is not None:
            try:
                self._session_publisher.publish(
                    EventRecord(
                        event_type=ErrorEventType.GENERIC,
                        timestamp_ns=time.monotonic_ns(),
                        sample_uuid=query_id,
                        data=synthetic_result.error,
                        conversation_id=conv_id,
                        turn=turn,
                    )
                )
                self._session_publisher.publish(
                    EventRecord(
                        event_type=SampleEventType.COMPLETE,
                        timestamp_ns=time.monotonic_ns(),
                        sample_uuid=query_id,
                        data=None,
                        conversation_id=conv_id,
                        turn=turn,
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to publish synthetic-failure EventRecords (query=%s)",
                    query_id,
                )
        if self._session_on_sample_complete is not None:
            try:
                self._session_on_sample_complete(synthetic_result)
            except Exception:
                logger.exception(
                    "on_sample_complete callback raised for synthetic failure (query=%s)",
                    query_id,
                )

    def _abort_remaining_turns(self, conv_id: str, reason: str) -> int:
        delay_handle = self._delay_handles.pop(conv_id, None)
        if delay_handle is not None:
            delay_handle.cancel()
        state = self._active_iters.pop(conv_id, None)
        if state is None:
            return 0
        _source_id, turns, cursor, _repeat_id = state
        assert self._phase_issuer is not None
        dropped = 0
        for idx, turn in turns[cursor:]:
            self._conv_manager.mark_turn_failed(conv_id)
            skipped_id = self._phase_issuer.register_skipped(
                idx, conversation_id=conv_id, turn=turn
            )
            if skipped_id is None:
                break
            self._publish_synthetic_failure(
                skipped_id,
                conv_id,
                turn,
                error_type="TurnAbortedByPriorFailure",
                error_message=reason,
            )
            dropped += 1
        return dropped

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
            logger.debug(
                "dropping late response result=%s (no matching in-flight entry)",
                result.id,
            )
            return

        handle = self._timeout_handles.pop(result.id, None)
        if handle is not None:
            handle.cancel()

        try:
            if result.error is not None:
                self._conv_manager.mark_turn_failed(conv_id)
            else:
                self._conv_manager.mark_turn_complete(conv_id)
        except KeyError:
            self._active_iters.pop(conv_id, None)
            self._fill_slot()
            logger.exception(
                "on_sample_complete routing miss for conv=%s result=%s",
                conv_id,
                result.id,
            )
            return

        # If this turn failed, abandon the rest of the conversation: later
        # client turns depend on the failed prior response, matching timeout
        # handling.
        if result.error is not None:
            err_type = (
                result.error.error_type if result.error is not None else "unknown"
            )
            dropped = self._abort_remaining_turns(
                conv_id, reason=f"prior turn errored: {err_type}"
            )
            if dropped:
                logger.warning(
                    "turn error on conv=%s dropped %d remaining client turn(s)",
                    conv_id,
                    dropped,
                )
            self._finish_conversation(conv_id)
            return

        if self._stopping:
            self._signal_done_if_no_inflight()
            return

        try:
            self._issue_next_turn(conv_id)
        except Exception as exc:
            logger.exception("Error issuing next turn for %s", conv_id)
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()
