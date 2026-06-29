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

"""BFCL v4 multi-turn execution bridge.

Manages the agentic execution loop for BFCL multi-turn evaluation:
model output -> parse tool_calls -> execute locally -> construct tool messages -> repeat.

This bridge sits between Endpoints' HTTP client and the BFCL scoring pipeline,
handling per-conversation state, function execution via bfcl-eval's simulated
classes, and accumulation of results for final scoring.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..dataset_manager.predefined.bfcl_v4 import _convert_bfcl_functions_to_tools
from ..dataset_manager.predefined.bfcl_v4.multi_turn import BFCLv4MultiTurnEntry

logger = logging.getLogger(__name__)

try:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
        is_empty_execute_response,
    )
    from bfcl_eval.model_handler.utils import convert_to_function_call
except ImportError:
    execute_multi_turn_func_call = None
    is_empty_execute_response = None
    convert_to_function_call = None

DEFAULT_MAX_STEPS_PER_TURN = 25
_MODEL_NAME_FOR_EXECUTION = "mlcommons_endpoints_eval"


@dataclass
class ConversationExecState:
    """Tracks execution state for a single multi-turn conversation."""

    entry: BFCLv4MultiTurnEntry
    current_turn: int = 0
    current_step: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    model_results_per_turn: list[list[list[str]]] = field(default_factory=list)
    current_turn_results: list[list[str]] = field(default_factory=list)
    force_terminated: bool = False
    completed: bool = False


class BFCLExecutionBridge:
    """Manages the agentic execution loop for BFCL multi-turn conversations.

    For each conversation (test entry), this bridge:
    1. Provides the initial messages + tools for the first request
    2. On each model response, parses tool_calls, executes them, and determines
       whether to issue another step (tool results feedback) or advance to next turn
    3. Accumulates decoded model results for final scoring via multi_turn_checker

    Usage:
        bridge = BFCLExecutionBridge(max_steps_per_turn=25)
        state = bridge.start_conversation(entry)

        # First request
        messages, tools = bridge.get_initial_request(state)
        # ... send via HTTP client, get response ...

        # Process response and get next action
        action = bridge.process_response(state, tool_calls, tool_call_ids)
        while action.type == "continue":
            # Send action.messages with action.tools
            # ... get response ...
            action = bridge.process_response(state, tool_calls, tool_call_ids)

        # When action.type == "next_turn" or "complete", advance accordingly
    """

    def __init__(self, max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN):
        if convert_to_function_call is None:
            raise ImportError(
                "bfcl-eval is required for BFCL v4 multi-turn execution. "
                "Install with: pip install inference-endpoint[bfcl]"
            )
        self._max_steps = max_steps_per_turn
        self._states: dict[str, ConversationExecState] = {}

    def start_conversation(self, entry: BFCLv4MultiTurnEntry) -> ConversationExecState:
        """Initialize execution state for a new conversation."""
        state = ConversationExecState(
            entry=entry,
            tools=list(entry.tools),
        )
        self._states[entry.entry_id] = state
        return state

    def get_initial_request(
        self, state: ConversationExecState
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Get messages and tools for the first request of the current turn.

        Returns:
            Tuple of (messages, tools) to send to the endpoint.
        """
        entry = state.entry
        turn_idx = state.current_turn

        # Handle holdout functions (miss_func category). Augment only the
        # per-conversation tool list on `state`; never mutate the shared
        # dataset `entry`, or re-runs (retry/re-score) would append the same
        # holdout functions again and corrupt the execution context.
        if str(turn_idx) in entry.holdout_function:
            added_funcs = entry.holdout_function[str(turn_idx)]
            added_tools = _convert_bfcl_functions_to_tools(added_funcs)
            state.tools = state.tools + added_tools

        turn_messages = entry.get_turn_messages(turn_idx)

        # Build full message history
        for msg in turn_messages:
            content = msg.get("content") or ""
            state.messages.append({"role": msg["role"], "content": content})

        state.current_step = 0
        state.current_turn_results = []

        return list(state.messages), list(state.tools)

    def process_response(
        self,
        state: ConversationExecState,
        tool_calls: list[dict[str, Any]] | None,
        tool_call_ids: list[str] | None = None,
        content: str | None = None,
    ) -> "ExecutionAction":
        """Process a model response and determine the next action.

        Args:
            state: Current conversation execution state.
            tool_calls: Parsed tool_calls from model response. Each item is
                {"name": "func_name", "arguments": "{...}" or {...}}.
                None if model did not produce tool calls.
            tool_call_ids: Tool call IDs from the response (for constructing
                tool result messages).
            content: Text content from model response (used when no tool_calls).

        Returns:
            ExecutionAction indicating what to do next.
        """
        entry = state.entry

        # Add assistant message to history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            tc["arguments"]
                            if isinstance(tc["arguments"], str)
                            else json.dumps(tc["arguments"])
                        ),
                    },
                }
                for tc, tid in zip(
                    tool_calls,
                    tool_call_ids or [f"call_{i}" for i in range(len(tool_calls))],
                    strict=False,
                )
            ]
            assistant_msg["content"] = content or ""
        else:
            assistant_msg["content"] = content or ""

        state.messages.append(assistant_msg)

        # If no tool calls, model is done with this turn
        if not tool_calls:
            logger.debug(
                "No tool_calls for %s turn %d step %d — advancing",
                entry.entry_id,
                state.current_turn,
                state.current_step,
            )
            return self._advance_turn(state)

        # Parse tool_calls into bfcl-eval format: [{func_name: args_dict}, ...]
        # and decode into executable format. Both steps are inside one try-except
        # so that invalid JSON in arguments (common for small/quantized models)
        # is handled the same way as an unrecognised function name.
        try:
            model_responses_bfcl = []
            for tc in tool_calls:
                args = tc["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                model_responses_bfcl.append({tc["name"]: args})
            decoded_calls = convert_to_function_call(model_responses_bfcl)
        except Exception as exc:
            logger.warning(
                "Failed to decode function calls for %s turn %d: %s",
                entry.entry_id,
                state.current_turn,
                exc,
            )
            return self._advance_turn(state)

        if is_empty_execute_response(decoded_calls):
            logger.debug(
                "Empty decoded response for %s turn %d — advancing",
                entry.entry_id,
                state.current_turn,
            )
            return self._advance_turn(state)

        # Store decoded calls for scoring
        state.current_turn_results.append(decoded_calls)

        # Execute function calls against simulated classes
        test_category = entry.entry_id.rsplit("_", 1)[0]
        execution_results, _ = execute_multi_turn_func_call(
            func_call_list=decoded_calls,
            initial_config=entry.initial_config,
            involved_classes=entry.involved_classes,
            model_name=_MODEL_NAME_FOR_EXECUTION,
            test_entry_id=entry.entry_id,
            long_context=(
                "long_context" in test_category or "composite" in test_category
            ),
            is_evaL_run=False,
        )

        # Add tool result messages to history
        if tool_call_ids is None:
            tool_call_ids = [f"call_{i}" for i in range(len(tool_calls))]

        for exec_result, call_id in zip(execution_results, tool_call_ids, strict=False):
            tool_msg = {
                "role": "tool",
                "content": str(exec_result),
                "tool_call_id": call_id,
            }
            state.messages.append(tool_msg)

        state.current_step += 1

        # Check force-termination
        if state.current_step >= self._max_steps:
            logger.warning(
                "Force-terminating %s turn %d after %d steps",
                entry.entry_id,
                state.current_turn,
                state.current_step,
            )
            state.force_terminated = True
            state.completed = True
            state.model_results_per_turn.append(state.current_turn_results)
            return ExecutionAction(action_type="force_terminated")

        # Issue another request with updated history (tool results appended)
        return ExecutionAction(
            action_type="continue",
            messages=list(state.messages),
            tools=list(state.tools),
        )

    def _advance_turn(self, state: ConversationExecState) -> "ExecutionAction":
        """Advance to the next turn or mark conversation as complete."""
        state.model_results_per_turn.append(state.current_turn_results)
        state.current_turn += 1

        if state.current_turn >= state.entry.num_turns:
            state.completed = True
            return ExecutionAction(action_type="complete")

        # Prepare next turn
        messages, tools = self.get_initial_request(state)
        return ExecutionAction(
            action_type="next_turn",
            messages=messages,
            tools=tools,
        )

    def get_results(self, state: ConversationExecState) -> dict[str, Any]:
        """Get accumulated results for scoring after conversation completes.

        Returns:
            Dict with entry metadata and model results ready for multi_turn_checker.
        """
        return {
            "entry_id": state.entry.entry_id,
            "subset": state.entry.subset,
            "model_results_per_turn": state.model_results_per_turn,
            "ground_truth": state.entry.ground_truth,
            "initial_config": state.entry.initial_config,
            "involved_classes": state.entry.involved_classes,
            "force_terminated": state.force_terminated,
            "num_turns_completed": state.current_turn,
            "num_turns_expected": state.entry.num_turns,
        }

    def cleanup(self, entry_id: str) -> None:
        """Remove state for a completed conversation."""
        self._states.pop(entry_id, None)


@dataclass
class ExecutionAction:
    """Represents the next action after processing a model response.

    action_type:
        "continue" — issue another request (same turn, tool results appended)
        "next_turn" — advance to next turn (new user message appended)
        "complete" — all turns finished, ready for scoring
        "force_terminated" — max steps exceeded, conversation aborted
    """

    action_type: str
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
