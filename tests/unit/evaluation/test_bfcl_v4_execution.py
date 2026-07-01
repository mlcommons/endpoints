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

"""Unit tests for BFCLExecutionBridge.process_response.

bfcl-eval is an optional, conflicting dependency, so the three symbols the
bridge imports from it are stubbed here. This exercises the decode ->
execute -> tool-message -> force-termination path (and pins the upstream
``is_evaL_run`` kwarg spelling) without requiring bfcl-eval to be installed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from inference_endpoint.evaluation import bfcl_v4_execution as ex


def _state(entry_id="multi_turn_base_0", max_steps=25):
    entry = SimpleNamespace(
        entry_id=entry_id,
        initial_config={},
        involved_classes=[],
    )
    state = SimpleNamespace(
        entry=entry,
        current_turn=0,
        current_step=0,
        current_turn_results=[],
        messages=[],
        tools=[],
        force_terminated=False,
        completed=False,
        model_results_per_turn=[],
    )
    return state


def _make_bridge(max_steps=25):
    # __init__ guards on convert_to_function_call being importable.
    with patch.object(ex, "convert_to_function_call", lambda x: x):
        return ex.BFCLExecutionBridge(max_steps_per_turn=max_steps)


@pytest.mark.unit
def test_process_response_continue_builds_tool_messages_and_passes_kwarg():
    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return (["result_for_call"], None)

    bridge = _make_bridge(max_steps=25)
    state = _state()
    with (
        patch.object(ex, "convert_to_function_call", lambda calls: ["f(x=1)"]),
        patch.object(ex, "is_empty_execute_response", lambda d: False),
        patch.object(ex, "execute_multi_turn_func_call", fake_execute),
    ):
        action = bridge.process_response(
            state,
            tool_calls=[{"name": "f", "arguments": {"x": 1}}],
            tool_call_ids=["call_0"],
            content="",
        )

    assert action.action_type == "continue"
    # Upstream kwarg spelling (capital L) must be honored exactly.
    assert captured["is_evaL_run"] is False
    assert captured["long_context"] is False
    # A tool result message was appended for the executed call.
    assert any(m.get("role") == "tool" for m in state.messages)


@pytest.mark.unit
def test_process_response_force_terminates_at_max_steps():
    bridge = _make_bridge(max_steps=1)
    state = _state()
    with (
        patch.object(ex, "convert_to_function_call", lambda calls: ["f()"]),
        patch.object(ex, "is_empty_execute_response", lambda d: False),
        patch.object(ex, "execute_multi_turn_func_call", lambda **k: (["r"], None)),
    ):
        action = bridge.process_response(
            state, tool_calls=[{"name": "f", "arguments": {}}], tool_call_ids=["c0"]
        )

    assert action.action_type == "force_terminated"
    assert state.force_terminated is True and state.completed is True


@pytest.mark.unit
def test_process_response_no_tool_calls_advances_turn():
    bridge = _make_bridge()
    state = _state()
    state.entry.num_turns = 1  # advancing past the only turn completes
    action = bridge.process_response(state, tool_calls=None, content="all done")
    assert action.action_type == "complete"
    assert state.completed is True
