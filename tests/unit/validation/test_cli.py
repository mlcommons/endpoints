# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy inspection and submission validation CLI contracts."""

import json
import sys

import pytest

from inference_endpoint.validation.__main__ import main

pytestmark = pytest.mark.unit


def test_inspect_bundle(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["validation"])
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["rules"] == 89
    assert len(output["files"]) == 5
    assert "checks" not in output


def test_submission_command_executes_checks(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing"
    monkeypatch.setattr(sys, "argv", ["validation", "--submission", str(missing)])
    assert main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["engine"] == "policy"
    assert not output["passed"]
    assert any(result["rule"] == "path-exists" for result in output["errors"])


def test_submission_execution_rejects_invalid_policy_bundle(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["validation", str(tmp_path), "--submission", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "policy files" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--seed-sets", "--approved-sped-decode-heads"])
def test_cli_rejects_catalog_overrides(option, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["validation", option, str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
