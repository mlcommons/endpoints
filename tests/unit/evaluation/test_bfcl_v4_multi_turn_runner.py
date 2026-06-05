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

"""Unit tests for BFCLMultiTurnRunner seed forwarding."""

from unittest.mock import MagicMock, patch

import pytest


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
        runner._send_request(
            messages=[{"role": "user", "content": "hello"}], tools=[]
        )

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
        runner._send_request(
            messages=[{"role": "user", "content": "hello"}], tools=[]
        )

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
