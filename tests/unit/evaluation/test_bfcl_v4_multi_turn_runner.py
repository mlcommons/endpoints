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

"""Unit tests for BFCLMultiTurnRunner seed forwarding and orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _ok_response(content="done", tool_calls=None):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }
    return resp


def _make_runner(seed=None, **kwargs):
    """Construct a BFCLMultiTurnRunner with BFCLExecutionBridge mocked out."""
    with patch(
        "inference_endpoint.evaluation.bfcl_v4_multi_turn_runner.BFCLExecutionBridge"
    ):
        from inference_endpoint.evaluation.bfcl_v4_multi_turn_runner import (
            BFCLMultiTurnRunner,
        )

        return BFCLMultiTurnRunner(
            endpoint_url="http://localhost:8080",
            model_name=kwargs.pop("model_name", "test-model"),
            seed=seed,
            **kwargs,
        )


@pytest.mark.unit
def test_seed_included_in_payload_when_set():
    """When seed is provided, it is added to the HTTP request payload."""
    runner = _make_runner(seed=42)
    captured: list[dict] = []

    def fake_post(url, json=None, **kwargs):
        captured.append(json or {})
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": None,
                    }
                }
            ]
        }
        return resp

    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value = MagicMock(post=fake_post)
        runner._send_request(messages=[{"role": "user", "content": "hello"}], tools=[])

    assert captured, "No request was sent"
    assert captured[0].get("seed") == 42


@pytest.mark.unit
def test_seed_omitted_from_payload_when_none():
    """When seed is not set, the payload does not include a seed key."""
    runner = _make_runner()
    captured: list[dict] = []

    def fake_post(url, json=None, **kwargs):
        captured.append(json or {})
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        }
        return resp

    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value = MagicMock(post=fake_post)
        runner._send_request(messages=[{"role": "user", "content": "hello"}], tools=[])

    assert captured, "No request was sent"
    assert "seed" not in captured[0]


@pytest.mark.unit
def test_runner_stores_seed():
    """BFCLMultiTurnRunner stores the seed parameter."""
    runner = _make_runner(seed=7, model_name="m")
    assert runner._seed == 7


@pytest.mark.unit
def test_runner_seed_default_none():
    """BFCLMultiTurnRunner seed defaults to None when not provided."""
    runner = _make_runner(model_name="m")
    assert runner._seed is None


@pytest.mark.unit
def test_send_request_includes_tools_only_when_present():
    """tools/tool_choice are sent only when tools is non-empty (avoids 400s)."""
    runner = _make_runner()
    captured: list[dict] = []

    def fake_post(url, json=None, **kwargs):
        captured.append(json or {})
        return _ok_response()

    with patch.object(runner, "_get_client") as mock_client:
        mock_client.return_value = MagicMock(post=fake_post)
        runner._send_request(messages=[{"role": "user", "content": "hi"}], tools=[])
        runner._send_request(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )

    assert "tools" not in captured[0] and "tool_choice" not in captured[0]
    assert captured[1]["tool_choice"] == "auto" and captured[1]["tools"]


@pytest.mark.unit
def test_parse_response_malformed_returns_none_triple():
    """A response missing choices/message yields (None, None, None), not a crash."""
    runner = _make_runner()
    assert runner._parse_response({}) == (None, None, None)
    assert runner._parse_response({"choices": []}) == (None, None, None)


@pytest.mark.unit
def test_parse_response_extracts_tool_calls():
    runner = _make_runner()
    resp = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
            }
        ]
    }
    tool_calls, ids, content = runner._parse_response(resp)
    assert tool_calls == [{"name": "f", "arguments": "{}"}]
    assert ids == ["c1"]


def _runner_with_mock_bridge(state, **bridge_returns):
    """Build a runner whose bridge is a MagicMock primed for run_entry."""
    runner = _make_runner()
    bridge = runner._bridge
    bridge.start_conversation.return_value = state
    bridge.get_initial_request.return_value = ([{"role": "user", "content": "go"}], [])
    bridge.get_results.return_value = {"entry_id": "e"}
    for k, v in bridge_returns.items():
        getattr(bridge, k).return_value = v
    return runner, bridge


@pytest.mark.unit
def test_run_entry_force_terminates_on_failed_request():
    """A None response (request failure) force-terminates and still returns results."""
    state = SimpleNamespace(
        completed=False,
        force_terminated=False,
        model_results_per_turn=[],
        current_turn_results=[],
    )
    runner, bridge = _runner_with_mock_bridge(state)
    with patch.object(runner, "_send_request", return_value=None):
        result = runner.run_entry(MagicMock())
    assert state.force_terminated is True and state.completed is True
    assert result["total_requests"] == 1
    bridge.process_response.assert_not_called()


@pytest.mark.unit
def test_run_entry_breaks_on_complete_action():
    state = SimpleNamespace(
        completed=False,
        force_terminated=False,
        model_results_per_turn=[],
        current_turn_results=[],
    )
    runner, bridge = _runner_with_mock_bridge(
        state,
        process_response=SimpleNamespace(action_type="complete"),
    )
    with patch.object(runner, "_send_request", return_value=_ok_response().json()):
        result = runner.run_entry(MagicMock())
    assert result["total_requests"] == 1


@pytest.mark.unit
def test_run_entry_continues_then_completes():
    state = SimpleNamespace(
        completed=False,
        force_terminated=False,
        model_results_per_turn=[],
        current_turn_results=[],
    )
    runner, bridge = _runner_with_mock_bridge(state)
    bridge.process_response.side_effect = [
        SimpleNamespace(
            action_type="continue",
            messages=[{"role": "user", "content": "again"}],
            tools=[],
        ),
        SimpleNamespace(action_type="complete", messages=None, tools=None),
    ]
    with patch.object(runner, "_send_request", return_value=_ok_response().json()):
        result = runner.run_entry(MagicMock())
    assert result["total_requests"] == 2
