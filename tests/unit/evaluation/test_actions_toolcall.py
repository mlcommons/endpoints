# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Qwen mini-swe-agent toolcall replacements."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

_ACTIONS_TOOLCALL = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "10_Agentic_Inference"
    / "accuracy"
    / "actions_toolcall.py"
)


def _load_actions_module(monkeypatch):
    class FormatError(Exception):
        pass

    exceptions_mod = types.ModuleType("minisweagent.exceptions")
    exceptions_mod.FormatError = FormatError
    multimodal_mod = types.ModuleType("minisweagent.models.utils.openai_multimodal")
    multimodal_mod.expand_multimodal_content = lambda msg, pattern: msg

    for name in [
        "minisweagent",
        "minisweagent.models",
        "minisweagent.models.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions_mod)
    monkeypatch.setitem(
        sys.modules,
        "minisweagent.models.utils.openai_multimodal",
        multimodal_mod,
    )

    spec = importlib.util.spec_from_file_location(
        "_test_actions_toolcall", _ACTIONS_TOOLCALL
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool_call(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_finish_emits_relative_pathspecs_and_git_add_intent(monkeypatch):
    actions_mod = _load_actions_module(monkeypatch)

    actions = actions_mod.parse_toolcall_actions(
        [
            _tool_call(
                "finish",
                {
                    "files_modified": [
                        "/testbed/pkg/new file.py",
                        "tests/test_widget.py",
                    ]
                },
            )
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
    actions_mod = _load_actions_module(monkeypatch)

    actions = actions_mod.parse_toolcall_actions(
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
    actions_mod = _load_actions_module(monkeypatch)

    with pytest.raises(
        actions_mod.FormatError, match="view_range values must be integers"
    ):
        actions_mod.parse_toolcall_actions(
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
