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

"""Tests for version utilities."""

import shutil
import subprocess
from pathlib import Path

import pytest
from inference_endpoint import __version__
from inference_endpoint.utils import version as version_mod
from inference_endpoint.utils.version import _REPO_ROOT, get_git_sha, get_version_info


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.unit
def test_get_git_sha():
    """Test that get_git_sha returns a string or None."""
    sha = get_git_sha()
    if sha is not None:
        assert isinstance(sha, str)
        # --short=7 is a minimum width; git lengthens it on prefix collision.
        assert 7 <= len(sha) <= 40
        assert sha.isalnum()  # Should only contain alphanumeric chars


@pytest.mark.unit
def test_get_version_info():
    """Test that get_version_info returns correct structure."""
    info = get_version_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "git_sha" in info
    assert info["version"] == __version__
    # git_sha can be None if not in a git repo
    if info["git_sha"] is not None:
        assert isinstance(info["git_sha"], str)
        assert 7 <= len(info["git_sha"]) <= 40


@pytest.mark.unit
def test_version_info_cached():
    """Test that get_version_info is properly cached."""
    info1 = get_version_info()
    info2 = get_version_info()
    # Should return the same object due to lru_cache
    assert info1 is info2


@pytest.mark.unit
def test_git_sha_is_endpoints_repo_not_cwd(tmp_path, monkeypatch):
    """get_git_sha reports the endpoints repo SHA, not the launch dir's repo.

    Without anchoring to the package location, running the CLI from an unrelated
    git repo would record that repo's SHA into run provenance.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    # The endpoints repo SHA, resolved independently of the process CWD.
    try:
        expected = _git("rev-parse", "--short=7", "HEAD", cwd=_REPO_ROOT)
    except subprocess.CalledProcessError:
        pytest.skip("Source tree is not a git repository")

    # A distinct, unrelated git repo the CLI is pretend-launched from.
    other = tmp_path / "other_repo"
    other.mkdir()
    _git("init", cwd=other)
    _git(
        "-c",
        "user.email=t@t.co",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-m",
        "unrelated",
        cwd=other,
    )
    other_sha = _git("rev-parse", "--short=7", "HEAD", cwd=other)
    assert other_sha != expected  # sanity: the two repos differ

    monkeypatch.chdir(other)
    get_git_sha.cache_clear()
    try:
        sha = get_git_sha()
    finally:
        # Don't leak the cache-cleared value into other tests' assumptions.
        get_git_sha.cache_clear()

    assert sha == expected
    assert sha != other_sha


@pytest.mark.unit
@pytest.mark.parametrize(
    "fake",
    [
        # git missing / cwd gone / timed out
        FileNotFoundError(),
        subprocess.TimeoutExpired(cmd="git", timeout=1.0),
        # not a git repo (non-zero exit)
        subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="x"),
        # malformed output (not exactly two lines)
        subprocess.CompletedProcess(args=[], returncode=0, stdout="only-one-line\n"),
        # discovered repo is NOT this source tree (e.g. wheel nested in another repo)
        subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/some/other/repo\ndeadbee\n"
        ),
    ],
)
def test_git_sha_returns_none_on_untrusted_or_missing_repo(fake, monkeypatch):
    """A foreign/absent repo yields None rather than a wrong provenance SHA."""

    def fake_run(*args, **kwargs):
        if isinstance(fake, BaseException):
            raise fake
        return fake

    monkeypatch.setattr(version_mod.subprocess, "run", fake_run)
    get_git_sha.cache_clear()
    try:
        assert get_git_sha() is None
    finally:
        get_git_sha.cache_clear()


@pytest.mark.unit
def test_git_sha_returned_when_toplevel_matches(monkeypatch):
    """Happy path without a real git repo: toplevel == _REPO_ROOT -> return sha.

    Covers the success branch deterministically so it holds in a no-git
    environment (installed wheel / clean container).
    """

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"{_REPO_ROOT}\nabc1234\n"
        )

    monkeypatch.setattr(version_mod.subprocess, "run", fake_run)
    get_git_sha.cache_clear()
    try:
        assert get_git_sha() == "abc1234"
    finally:
        get_git_sha.cache_clear()


@pytest.mark.unit
def test_resolve_repo_root_falls_back_on_shallow_tree(monkeypatch):
    """A too-shallow module path yields the filesystem root, not IndexError."""
    monkeypatch.setattr(version_mod, "__file__", "/inference_endpoint/utils/version.py")
    root = version_mod._resolve_repo_root()
    assert root == Path("/inference_endpoint/utils/version.py").resolve().parents[-1]
