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
import sys
from pathlib import Path

import inference_endpoint
import pytest
from inference_endpoint import __version__
from inference_endpoint.utils import version as version_mod
from inference_endpoint.utils.version import (
    _REPO_ROOT,
    _valid_sha,
    get_git_sha,
    get_version_info,
    resolve_git_sha,
)


def _fake_git_run(sha: str, *, dirty: bool):
    """subprocess.run stand-in dispatching on the git subcommand.

    ``rev-parse`` returns a matching toplevel + ``sha``; ``diff-index`` returns
    exit 1 (dirty) or 0 (clean).
    """

    def fake_run(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=f"{_REPO_ROOT}\n{sha}\n"
            )
        if "diff-index" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1 if dirty else 0, stdout=""
            )
        raise AssertionError(f"unexpected git invocation: {cmd}")

    return fake_run


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# _valid_sha
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "abc1234",  # 7-char short SHA-1
        "abc1234-dirty",  # with dirty marker
        "a" * 40,  # full SHA-1
        "A1B2C3D",  # uppercase accepted
        "f" * 64,  # full SHA-256
        "  abc1234  ",  # surrounding whitespace stripped
    ],
)
def test_valid_sha_accepts(value):
    assert _valid_sha(value) == value.strip()


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        None,
        1234567,  # non-string (unquoted GIT_SHA in a hand-edited _build_info.py)
        "",
        "   ",
        "unknown",  # the display sentinel
        "not-a-sha",
        "abc123",  # too short (6)
        "a" * 65,  # too long (>64)
        "abc1234\ndef5678",  # embedded newline (injection) — fullmatch anchors
        "abc1234-dirty-dirty",  # trailing junk
        "abc1234\x1b[31m",  # ANSI escape
    ],
)
def test_valid_sha_rejects(value):
    assert _valid_sha(value) is None


# ---------------------------------------------------------------------------
# get_git_sha (live-git channel)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_git_sha():
    """Test that get_git_sha returns a string or None."""
    sha = get_git_sha()
    if sha is not None:
        assert isinstance(sha, str)
        # A dirty working tree appends a "-dirty" suffix; strip it before the
        # shape checks on the bare SHA.
        base = sha.removesuffix("-dirty")
        # --short=7 is a minimum width; git lengthens it on prefix collision.
        assert 7 <= len(base) <= 40
        assert base.isalnum()


@pytest.mark.unit
def test_get_git_sha_appends_dirty_when_tree_has_tracked_changes(monkeypatch):
    """Uncommitted tracked changes surface as a ``-dirty`` suffix on the SHA."""
    monkeypatch.setattr(
        version_mod.subprocess, "run", _fake_git_run("abc1234", dirty=True)
    )
    assert get_git_sha() == "abc1234-dirty"


@pytest.mark.unit
def test_get_git_sha_no_dirty_suffix_when_clean(monkeypatch):
    """A clean tree yields the bare SHA with no suffix."""
    monkeypatch.setattr(
        version_mod.subprocess, "run", _fake_git_run("abc1234", dirty=False)
    )
    assert get_git_sha() == "abc1234"


@pytest.mark.unit
def test_git_tree_dirty_fails_to_dirty_on_error_returncode(monkeypatch):
    """A non-0/1 diff-index code (e.g. 128, no HEAD/lock) errs to dirty, not clean."""

    def fake_run(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=f"{_REPO_ROOT}\nabc1234\n"
            )
        if "diff-index" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=128, stdout="")
        raise AssertionError(f"unexpected git invocation: {cmd}")

    monkeypatch.setattr(version_mod.subprocess, "run", fake_run)
    assert get_git_sha() == "abc1234-dirty"


@pytest.mark.unit
def test_git_tree_dirty_fails_to_dirty_on_exception(monkeypatch):
    """If the dirty probe raises (timeout/OSError), err on the side of dirty."""

    def fake_run(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=f"{_REPO_ROOT}\nabc1234\n"
            )
        if "diff-index" in cmd:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)
        raise AssertionError(f"unexpected git invocation: {cmd}")

    monkeypatch.setattr(version_mod.subprocess, "run", fake_run)
    assert get_git_sha() == "abc1234-dirty"


@pytest.mark.unit
@pytest.mark.parametrize(
    "fake",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired(cmd="git", timeout=1.0),
        subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="x"),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="only-one-line\n"),
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
    assert get_git_sha() is None


@pytest.mark.unit
def test_git_sha_returned_when_toplevel_matches(monkeypatch):
    """Happy path without a real git repo: toplevel == _REPO_ROOT -> return sha."""
    monkeypatch.setattr(
        version_mod.subprocess, "run", _fake_git_run("abc1234", dirty=False)
    )
    assert get_git_sha() == "abc1234"


@pytest.mark.unit
def test_git_sha_is_endpoints_repo_not_cwd(tmp_path, monkeypatch):
    """get_git_sha reports the endpoints repo SHA, not the launch dir's repo.

    Without anchoring to the package location, running the CLI from an unrelated
    git repo would record that repo's SHA into run provenance.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")

    try:
        expected = _git("rev-parse", "--short=7", "HEAD", cwd=_REPO_ROOT)
    except subprocess.CalledProcessError:
        pytest.skip("Source tree is not a git repository")

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
    assert other_sha != expected

    monkeypatch.chdir(other)
    sha = get_git_sha()

    # A dirty source tree (e.g. mid-development) appends "-dirty"; compare bases.
    assert sha is not None
    base = sha.removesuffix("-dirty")
    assert base == expected
    assert base != other_sha


_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


@pytest.mark.unit
def test_git_env_drops_git_location_family(monkeypatch):
    """_git_env scrubs the whole git-location env family but preserves the rest."""
    for var in _GIT_LOCATION_VARS:
        monkeypatch.setenv(var, "/foreign")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = version_mod._git_env()
    for var in _GIT_LOCATION_VARS:
        assert var not in env
    assert env["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_get_git_sha_scrubs_git_env_and_preserves_dirty(monkeypatch):
    """Both git calls get a scrubbed env, and a stray index can't fake a clean tree.

    Pins the fix directly: GIT_DIR/GIT_INDEX_FILE set in the ambient env must not
    reach either subprocess, and a dirty diff-index still yields a -dirty SHA.
    """
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("GIT_INDEX_FILE", "/foreign/.git/index")
    seen_envs = []

    def fake_run(cmd, *args, **kwargs):
        seen_envs.append(kwargs.get("env"))
        if "rev-parse" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=f"{_REPO_ROOT}\nabc1234\n"
            )
        if "diff-index" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="")
        raise AssertionError(f"unexpected git invocation: {cmd}")

    monkeypatch.setattr(version_mod.subprocess, "run", fake_run)
    assert get_git_sha() == "abc1234-dirty"
    assert len(seen_envs) == 2  # rev-parse + diff-index
    for env in seen_envs:
        assert env is not None
        assert "GIT_DIR" not in env
        assert "GIT_INDEX_FILE" not in env


@pytest.mark.unit
def test_get_git_sha_ignores_stray_git_dir(tmp_path, monkeypatch):
    """A stray GIT_DIR must not slip a foreign repo's SHA past the toplevel guard.

    With GIT_DIR set (and GIT_WORK_TREE unset) git defaults the work-tree to cwd,
    so ``--show-toplevel`` reports our repo (guard passes) while ``HEAD`` would
    otherwise resolve the foreign repo. Scrubbing GIT_DIR closes that.
    """
    if shutil.which("git") is None:
        pytest.skip("git not available")
    try:
        our_sha = _git("rev-parse", "--short=7", "HEAD", cwd=_REPO_ROOT)
    except subprocess.CalledProcessError:
        pytest.skip("Source tree is not a git repository")

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git("init", cwd=foreign)
    _git(
        "-c",
        "user.email=t@t.co",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-m",
        "foreign",
        cwd=foreign,
    )
    foreign_sha = _git("rev-parse", "--short=7", "HEAD", cwd=foreign)
    assert foreign_sha != our_sha

    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    sha = get_git_sha()

    assert sha is not None
    base = sha.removesuffix("-dirty")
    assert base == our_sha
    assert base != foreign_sha


# ---------------------------------------------------------------------------
# resolve_git_sha (channel precedence + validation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_git_sha_prefers_baked_over_env_and_git(monkeypatch):
    """A baked build-info SHA wins over env and live git; source is 'baked'."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", "abc1234")
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "0000fff")
    assert resolve_git_sha() == ("abc1234", "baked")


@pytest.mark.unit
def test_resolve_git_sha_uses_env_when_no_baked(monkeypatch):
    """With no baked SHA, the ENDPOINTS_GIT_SHA env var wins; source is 'env'."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "0000fff")
    assert resolve_git_sha() == ("def5678", "env")


@pytest.mark.unit
def test_resolve_git_sha_accepts_dirty_and_uppercase_env(monkeypatch):
    """A dirty-suffixed / uppercase env attestation is honored verbatim."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "ABC1234-dirty")
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "0000fff")
    assert resolve_git_sha() == ("ABC1234-dirty", "env")


@pytest.mark.unit
def test_resolve_git_sha_falls_back_to_live_git(monkeypatch):
    """With no baked SHA and no env var, live git wins; source is 'git'."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234")
    assert resolve_git_sha() == ("abc1234", "git")


@pytest.mark.unit
def test_resolve_git_sha_none_when_nothing_resolves(monkeypatch):
    """No baked SHA, no env, no git checkout -> (None, 'none')."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: None)
    assert resolve_git_sha() == (None, "none")


@pytest.mark.unit
@pytest.mark.parametrize("blank", ["", "   "])
def test_resolve_git_sha_ignores_blank_env(monkeypatch, blank):
    """An empty / whitespace-only env var is treated as unset (falls to git)."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", blank)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234")
    assert resolve_git_sha() == ("abc1234", "git")


@pytest.mark.unit
def test_resolve_git_sha_rejects_unknown_sentinel_env(monkeypatch):
    """A stray/sentinel ENDPOINTS_GIT_SHA=unknown is rejected, not attested."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "unknown")
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: None)
    assert resolve_git_sha() == (None, "none")


@pytest.mark.unit
def test_resolve_git_sha_rejects_non_hex_env_falls_through(monkeypatch):
    """A non-SHA env value is ignored; resolution continues to live git."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "not-a-sha")
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234")
    assert resolve_git_sha() == ("abc1234", "git")


@pytest.mark.unit
def test_resolve_git_sha_rejects_garbage_baked_falls_through(monkeypatch):
    """A malformed baked value is ignored rather than reported as provenance."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", "not-a-real-sha")
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234")
    assert resolve_git_sha() == ("abc1234", "git")


@pytest.mark.unit
def test_resolve_git_sha_rejects_non_str_baked_without_crashing(monkeypatch):
    """A non-string baked GIT_SHA (unquoted int) is ignored, not a crash."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", 1234567)
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234")
    assert resolve_git_sha() == ("abc1234", "git")


@pytest.mark.unit
def test_resolve_git_sha_validates_live_git_channel(monkeypatch):
    """Even the git channel is validated: polluted git output -> not trusted."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: "abc1234\x1b[31m")
    assert resolve_git_sha() == (None, "none")


@pytest.mark.unit
def test_resolve_git_sha_reflects_later_env_change(monkeypatch):
    """resolve_git_sha is not cached: a later env change is observed."""
    monkeypatch.setattr(version_mod, "_BAKED_GIT_SHA", None)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: None)
    monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    assert resolve_git_sha() == (None, "none")
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")
    assert resolve_git_sha() == ("def5678", "env")


# ---------------------------------------------------------------------------
# get_version_info
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_version_info():
    """Test that get_version_info returns correct structure."""
    info = get_version_info()
    assert isinstance(info, dict)
    assert "version" in info
    assert "git_sha" in info
    assert "git_sha_source" in info
    assert info["version"] == __version__
    assert info["git_sha_source"] in {"baked", "env", "git", "none"}
    if info["git_sha"] is not None:
        assert isinstance(info["git_sha"], str)
        assert 7 <= len(info["git_sha"].removesuffix("-dirty")) <= 64


@pytest.mark.unit
def test_version_info_is_deterministic():
    """Two calls (now uncached) return equal structures."""
    assert get_version_info() == get_version_info()


@pytest.mark.unit
def test_corrupt_baked_file_degrades_not_crashes():
    """A corrupt _build_info.py must degrade (fall through), not crash version import.

    A hand-edited unquoted ``GIT_SHA = abc1234`` raises NameError at import — not
    ImportError/SyntaxError — so the baked-import handler must be broad enough to
    catch it, honoring the "degrade, don't crash Report.from_snapshot" invariant.
    Run in a subprocess (import-time behavior can't be re-exercised in-process).
    """
    pkg_dir = Path(inference_endpoint.__file__).resolve().parent
    baked = pkg_dir / "_build_info.py"
    if baked.exists():
        pytest.skip("a real _build_info.py is present; not overwriting it")
    baked.write_text("GIT_SHA = abc1234\n")  # unquoted -> NameError on import
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from inference_endpoint.utils.version import resolve_git_sha;"
                "print(resolve_git_sha()[1])",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        baked.unlink(missing_ok=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"baked", "env", "git", "none"}


@pytest.mark.unit
def test_resolve_repo_root_falls_back_on_shallow_tree(monkeypatch):
    """A too-shallow module path yields the filesystem root, not IndexError."""
    monkeypatch.setattr(version_mod, "__file__", "/inference_endpoint/utils/version.py")
    root = version_mod._resolve_repo_root()
    assert root == Path("/inference_endpoint/utils/version.py").resolve().parents[-1]
