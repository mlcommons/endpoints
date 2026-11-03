# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Qwen mini-swe-agent model extension."""

import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_SERVICE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "inference_endpoint"
    / "evaluation"
    / "swebench_service"
)


def _install_minisweagent_stubs(monkeypatch):
    class FormatError(Exception):
        pass

    class LitellmModel:
        def __init__(self, **kwargs):
            defaults = {
                "format_error_template": "{{ error }}",
                "observation_template": "{{ output.output }}",
                "multimodal_regex": "",
                "model_kwargs": {},
            }
            self.config = SimpleNamespace(**(defaults | kwargs))

    litellm = types.ModuleType("litellm")
    litellm.completion = lambda **kwargs: kwargs

    exceptions = types.ModuleType("minisweagent.exceptions")
    exceptions.FormatError = FormatError
    multimodal = types.ModuleType("minisweagent.models.utils.openai_multimodal")
    multimodal.expand_multimodal_content = lambda msg, pattern: {
        **msg,
        "multimodal_pattern": pattern,
    }
    litellm_model = types.ModuleType("minisweagent.models.litellm_model")
    litellm_model.LitellmModel = LitellmModel

    modules = {
        "litellm": litellm,
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.exceptions": exceptions,
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": litellm_model,
        "minisweagent.models.utils": types.ModuleType("minisweagent.models.utils"),
        "minisweagent.models.utils.openai_multimodal": multimodal,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("swebench_service.qwen_tools", "swebench_service.qwen_tools_model"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(_SERVICE_ROOT))


def _load_modules(monkeypatch):
    _install_minisweagent_stubs(monkeypatch)
    tools = importlib.import_module("swebench_service.qwen_tools")
    model = importlib.import_module("swebench_service.qwen_tools_model")
    return tools, model


def _tool_call(name: str, args: object, *, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("finish", []),
        (
            "finish",
            {"files_modified": ["pkg/file.py", 1]},
        ),
        ("str_replace_editor", []),
    ],
)
def test_malformed_tool_arguments_raise_format_error(monkeypatch, name, args):
    tools, _ = _load_modules(monkeypatch)

    with pytest.raises(tools.FormatError):
        tools.parse_toolcall_actions(
            [_tool_call(name, args)],
            format_error_template="{{ error }}",
        )


def test_finish_emits_relative_pathspecs_and_git_add_intent(monkeypatch):
    tools, _ = _load_modules(monkeypatch)

    actions = tools.parse_toolcall_actions(
        [
            _tool_call(
                "finish",
                {
                    "files_modified": [
                        "/testbed/pkg/new file.py",
                        "tests/test_widget.py",
                    ]
                },
            ),
        ],
        format_error_template="{{ error }}",
    )

    command = actions[0]["command"]
    assert (
        command == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
        "git -C /testbed add -N -- 'pkg/new file.py' tests/test_widget.py && "
        "git -C /testbed diff HEAD -- 'pkg/new file.py' tests/test_widget.py"
    )
    assert "/testbed/pkg/new file.py" not in command


def test_str_replace_editor_view_range_emits_clean_awk(monkeypatch):
    tools, _ = _load_modules(monkeypatch)

    actions = tools.parse_toolcall_actions(
        [
            _tool_call(
                "str_replace_editor",
                {
                    "command": "view",
                    "path": "/testbed/pkg/file.py",
                    "view_range": [2, 4],
                },
            )
        ],
        format_error_template="{{ error }}",
    )

    command = actions[0]["command"]
    assert (
        command == "awk 'NR>=2 && NR<=4 "
        '{printf "%6d\\t%s\\n", NR, $0}\' /testbed/pkg/file.py'
    )
    assert r"\'" not in command


def test_str_replace_editor_view_range_rejects_non_integers(monkeypatch):
    tools, _ = _load_modules(monkeypatch)

    with pytest.raises(tools.FormatError, match="view_range values must be integers"):
        tools.parse_toolcall_actions(
            [
                _tool_call(
                    "str_replace_editor",
                    {
                        "command": "view",
                        "path": "/testbed/pkg/file.py",
                        "view_range": ["one", 4],
                    },
                )
            ],
            format_error_template="{{ error }}",
        )


@pytest.mark.parametrize(
    ("contents", "old_str", "expected_contents", "succeeds"),
    [
        ("before unique after", "unique", "before replaced after", True),
        ("duplicate duplicate", "duplicate", "duplicate duplicate", False),
    ],
)
def test_str_replace_editor_command_exit_status_tracks_match_count(
    monkeypatch, tmp_path, contents, old_str, expected_contents, succeeds
):
    tools, _ = _load_modules(monkeypatch)
    target = tmp_path / "file.txt"
    target.write_text(contents)
    action = tools.parse_toolcall_actions(
        [
            _tool_call(
                "str_replace_editor",
                {
                    "command": "str_replace",
                    "path": str(target),
                    "old_str": old_str,
                    "new_str": "replaced",
                },
            )
        ],
        format_error_template="{{ error }}",
    )[0]

    result = subprocess.run(action["command"], shell=True, capture_output=True)

    assert (result.returncode == 0) is succeeds
    assert target.read_text() == expected_contents


def test_qwen_model_query_sends_custom_tool_request(monkeypatch):
    tools, model_mod = _load_modules(monkeypatch)
    calls: list[dict] = []
    monkeypatch.setattr(
        model_mod.litellm,
        "completion",
        lambda **kwargs: calls.append(kwargs) or "response",
    )
    model = model_mod.QwenToolsModel(
        model_name="openai/test-model",
        model_kwargs={"api_base": "http://endpoint/v1", "temperature": 0.2},
    )

    response = model._query([{"role": "user", "content": "task"}], temperature=0.7)

    assert response == "response"
    assert calls == [
        {
            "model": "openai/test-model",
            "messages": [{"role": "user", "content": "task"}],
            "tools": tools.TOOL_SCHEMAS,
            "api_base": "http://endpoint/v1",
            "temperature": 0.7,
        }
    ]


def test_qwen_model_uses_custom_parser_and_observation_formatter(monkeypatch):
    _, model_mod = _load_modules(monkeypatch)
    model = model_mod.QwenToolsModel(
        model_name="openai/test-model",
        model_kwargs={},
        observation_template="{{ output.output }}",
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[_tool_call("bash", {"command": "pwd"})]
                )
            )
        ]
    )

    actions = model._parse_actions(response)
    messages = model.format_observation_messages(
        {"extra": {"actions": actions}},
        [{"output": "/testbed", "returncode": 0}],
    )

    assert actions == [{"command": "pwd", "tool_call_id": "call-1"}]
    assert messages[0]["content"] == "/testbed"
    assert messages[0]["role"] == "tool"
