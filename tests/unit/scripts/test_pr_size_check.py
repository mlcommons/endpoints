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

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "pr_size_check.py"


def _load_pr_size_check():
    """Load scripts/pr_size_check.py as a module (it is not a package)."""
    if "pr_size_check" in sys.modules:
        return sys.modules["pr_size_check"]
    spec = importlib.util.spec_from_file_location("pr_size_check", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr_size_check"] = module
    spec.loader.exec_module(module)
    return module


mod = _load_pr_size_check()


@pytest.mark.parametrize(
    ("churn", "files", "expected"),
    [
        (0, 0, "normal"),
        (500, 20, "normal"),  # inclusive upper bound of Normal
        (501, 0, "large"),  # lines cross into Large
        (0, 21, "large"),  # files cross into Large
        (1500, 50, "large"),  # inclusive upper bound of Large
        (1501, 0, "very-large"),  # lines cross into Very large
        (0, 51, "very-large"),  # files cross into Very large
        (200, 60, "very-large"),  # file dimension dominates a small line count
    ],
)
def test_classify_boundaries(churn, files, expected):
    assert mod.classify(churn, files) == expected


def test_parse_numstat_sums_additions_and_deletions():
    text = "10\t5\tsrc/a.py\n3\t0\tsrc/b.py\n"
    assert mod.parse_numstat(text) == (18, 2)


def test_parse_numstat_counts_binary_file_with_zero_lines():
    text = "-\t-\tassets/img.png\n4\t1\tsrc/a.py\n"
    assert mod.parse_numstat(text) == (5, 2)


def test_parse_numstat_empty_is_zero():
    assert mod.parse_numstat("") == (0, 0)


def test_build_summary_is_one_line_and_names_class():
    summary = mod.build_summary("large", 812, 24, "origin/main", "1a2b3c4de")
    assert "\n" not in summary
    assert "LARGE" in summary
    assert "812 lines, 24 files" in summary


def test_build_details_lists_requirements():
    details = mod.build_details("large")
    assert "How to Review" in details
    assert "Advisory only" in details


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")


def test_main_excludes_tests_and_reports_normal(tmp_path, monkeypatch, capsys):
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "feature")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n")  # +2 non-test lines
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_big.py").write_text("\n".join(f"line{i}" for i in range(600)))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "feature")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_BASE_REF", "main")
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    rc = mod.main([])
    out = capsys.readouterr().out

    assert rc == 0
    # The 600-line test file is excluded; only a.py's +2 lines count -> Normal.
    assert "PR size: NORMAL" in out
    assert "2 lines, 1 files" in out
    # Default is terse: the threshold grid / requirements are not printed.
    assert "Advisory only" not in out


def test_main_verbose_prints_details(tmp_path, monkeypatch, capsys):
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "feature")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "feature")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_BASE_REF", "main")
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    rc = mod.main(["--verbose"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "PR size: NORMAL" in out
    assert "Advisory only" in out  # details present under --verbose


def test_main_writes_github_output(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "feature")
    (tmp_path / "a.py").write_text("x = 1\n" + "\n".join(f"c{i}" for i in range(600)))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "feature")

    output_file = tmp_path / "gh_output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_BASE_REF", "main")
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    rc = mod.main(["--github-output"])
    assert rc == 0
    written = output_file.read_text()
    assert "size_class=large" in written
    assert "files=1" in written


def test_main_skips_when_base_unresolvable(tmp_path, monkeypatch, capsys):
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PR_BASE_REF", "does/not/exist")
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    rc = mod.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipping" in out
