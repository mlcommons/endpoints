# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-dispatch gates, including the scale rule."""

from __future__ import annotations

import json

import pytest

from inference_endpoint.evaluation.swe_bench_distributed import gates as gates_mod
from inference_endpoint.evaluation.swe_bench_distributed.gates import (
    CheckpointIdentityGate,
    EndpointFingerprintGate,
    GateFailure,
    GateScaleError,
    ToolCallGate,
    build_scale_prompt,
    run_gates,
)

pytestmark = pytest.mark.unit

ENDPOINT = "http://engine-1:8000"


def install_http(monkeypatch, routes):
    """Route ``_http_json`` by URL suffix; a missing route raises like a network error."""

    def fake(url, payload=None, *, timeout_s=60.0, api_key=None):
        for suffix, response in routes.items():
            if url.endswith(suffix):
                if isinstance(response, Exception):
                    raise response
                if callable(response):
                    return response(payload)
                return response
        raise OSError(f"no route for {url}")

    monkeypatch.setattr(gates_mod, "_http_json", fake)


def tool_call_response(command="ls /testbed", name="bash", arguments=None):
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": name,
                                "arguments": (
                                    arguments
                                    if arguments is not None
                                    else json.dumps({"command": command})
                                ),
                            }
                        }
                    ]
                }
            }
        ]
    }


class TestCheckpointIdentity:
    def test_exact_match_passes(self, monkeypatch):
        install_http(monkeypatch, {"/get_model_info": {"model_path": "Org/Model-FP8"}})
        report = CheckpointIdentityGate("Org/Model-FP8").check([ENDPOINT])
        assert report.passed

    def test_a_prefix_is_not_a_match(self, monkeypatch):
        # "Org/Model" is a strict prefix of "Org/Model-FP8", so any
        # startswith/in test would accept an FP8 engine as the BF16 build.
        install_http(monkeypatch, {"/get_model_info": {"model_path": "Org/Model-FP8"}})
        report = CheckpointIdentityGate("Org/Model").check([ENDPOINT])
        assert not report.passed
        assert "Org/Model-FP8" in report.failures[0][1]

    def test_get_model_info_is_preferred_over_v1_models(self, monkeypatch):
        # /v1/models echoes --served-model-name, which operators routinely set
        # identically for two different checkpoints.
        install_http(
            monkeypatch,
            {
                "/get_model_info": {"model_path": "Org/Model-FP8"},
                "/v1/models": {"data": [{"id": "Org/Model"}]},
            },
        )
        assert CheckpointIdentityGate("Org/Model-FP8").check([ENDPOINT]).passed

    def test_v1_models_fallback_warns_about_its_own_ambiguity(self, monkeypatch):
        install_http(
            monkeypatch,
            {
                "/get_model_info": OSError("not sglang"),
                "/v1/models": {"data": [{"id": "Org/Model"}]},
            },
        )
        report = CheckpointIdentityGate("Org/Model").check([ENDPOINT])
        assert report.passed
        assert any("served-model-name" in note for note in report.notes)

    def test_an_unreachable_endpoint_fails_closed(self, monkeypatch):
        install_http(
            monkeypatch,
            {"/get_model_info": OSError("down"), "/v1/models": OSError("down")},
        )
        assert not CheckpointIdentityGate("Org/Model").check([ENDPOINT]).passed

    def test_ambiguous_model_listing_fails(self, monkeypatch):
        install_http(
            monkeypatch,
            {
                "/get_model_info": OSError("not sglang"),
                "/v1/models": {"data": [{"id": "a"}, {"id": "b"}]},
            },
        )
        report = CheckpointIdentityGate("a").check([ENDPOINT])
        assert not report.passed
        assert "ambiguous" in report.failures[0][1]


class TestToolCallScale:
    def test_a_small_prompt_fails_the_scale_assertion(self, monkeypatch):
        install_http(monkeypatch, {"/tokenize": {"count": 278}})
        gate = ToolCallGate("Org/Model", prompt="tiny", min_prompt_tokens=2000)
        # A gate exercising the right operation at a 278-token prompt passed
        # while every prompt above 2000 tokens silently returned nothing.
        with pytest.raises(GateScaleError, match="278 tokens"):
            gate.assert_scale([ENDPOINT])

    def test_a_scale_failure_is_a_gate_failure_not_a_skip(self, monkeypatch):
        install_http(monkeypatch, {"/tokenize": {"count": 100}})
        gate = ToolCallGate("Org/Model", min_prompt_tokens=2000)
        with pytest.raises(GateFailure, match="failing gate"):
            run_gates([gate], [ENDPOINT])

    def test_no_tokenizer_means_the_gate_cannot_prove_its_scale(self, monkeypatch):
        install_http(monkeypatch, {"/tokenize": OSError("404")})
        gate = ToolCallGate("Org/Model", min_prompt_tokens=2000)
        with pytest.raises(GateScaleError, match="cannot prove"):
            gate.assert_scale([ENDPOINT])

    def test_scale_can_be_waived_explicitly(self, monkeypatch):
        install_http(monkeypatch, {"/tokenize": OSError("404")})
        ToolCallGate("Org/Model", min_prompt_tokens=0).assert_scale([ENDPOINT])

    def test_the_default_prompt_is_large(self):
        assert len(build_scale_prompt()) > 20_000


class TestToolCallCheck:
    def _gate(self, monkeypatch, chat_response):
        install_http(
            monkeypatch,
            {"/tokenize": {"count": 4096}, "/v1/chat/completions": chat_response},
        )
        gate = ToolCallGate("Org/Model")
        gate.assert_scale([ENDPOINT])
        return gate

    def test_a_well_formed_call_passes(self, monkeypatch):
        gate = self._gate(monkeypatch, tool_call_response())
        report = gate.check([ENDPOINT])
        assert report.passed
        assert report.data["measured_tokens"][ENDPOINT] == 4096

    def test_an_empty_completion_fails(self, monkeypatch):
        gate = self._gate(monkeypatch, {"choices": [{"message": {"content": ""}}]})
        report = gate.check([ENDPOINT])
        assert not report.passed
        assert "no tool_calls" in report.failures[0][1]

    def test_the_wrong_tool_fails(self, monkeypatch):
        gate = self._gate(monkeypatch, tool_call_response(name="python"))
        assert not gate.check([ENDPOINT]).passed

    def test_unparseable_arguments_fail(self, monkeypatch):
        gate = self._gate(monkeypatch, tool_call_response(arguments="{not json"))
        report = gate.check([ENDPOINT])
        assert "not valid JSON" in report.failures[0][1]

    def test_an_empty_command_fails(self, monkeypatch):
        gate = self._gate(monkeypatch, tool_call_response(command="   "))
        assert not gate.check([ENDPOINT]).passed


class TestFingerprint:
    def test_the_fingerprint_changes_with_the_served_model(self, monkeypatch):
        install_http(monkeypatch, {"/get_model_info": {"model_path": "A"}})
        first = EndpointFingerprintGate().fingerprint(ENDPOINT)
        install_http(monkeypatch, {"/get_model_info": {"model_path": "B"}})
        second = EndpointFingerprintGate().fingerprint(ENDPOINT)
        assert first is not None and first != second

    def test_an_unidentifiable_endpoint_fails(self, monkeypatch):
        install_http(monkeypatch, {})
        assert not EndpointFingerprintGate().check([ENDPOINT]).passed


class TestRunGates:
    def test_every_gate_runs_even_after_one_fails(self, monkeypatch):
        install_http(
            monkeypatch,
            {
                "/get_model_info": {"model_path": "Wrong/Model"},
                "/tokenize": {"count": 4096},
                "/v1/chat/completions": {"choices": [{"message": {}}]},
            },
        )
        with pytest.raises(GateFailure) as excinfo:
            run_gates(
                [CheckpointIdentityGate("Org/Model"), ToolCallGate("Org/Model")],
                [ENDPOINT],
            )
        message = str(excinfo.value)
        assert "checkpoint_identity" in message
        assert "tool_call" in message

    def test_no_targets_is_a_failure_not_a_pass(self, monkeypatch):
        with pytest.raises(GateFailure):
            run_gates([CheckpointIdentityGate("Org/Model")], [])
