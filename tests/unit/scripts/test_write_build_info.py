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

"""Tests for the scripts/write_build_info.py build-time SHA generator."""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "write_build_info.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_build_info", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _exec_git_sha(path: Path) -> str:
    """Execute the generated module and return its GIT_SHA (proves importability)."""
    namespace: dict = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace["GIT_SHA"]


@pytest.fixture
def git_repo(tmp_path):
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "t@t.co", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "tracked.txt").write_text("v1\n")
    _git("add", "tracked.txt", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)
    return tmp_path


@pytest.mark.unit
def test_writes_importable_hex_sha_on_clean_tree(git_repo):
    mod = _load_module()
    out = git_repo / "_build_info.py"
    sha = mod.write_build_info(repo_root=git_repo, output=out)

    assert out.exists()
    assert re.fullmatch(r"[0-9a-f]{7}", sha)  # clean tree -> bare short SHA
    assert _exec_git_sha(out) == sha  # generated file is importable and matches
    assert "SPDX-License-Identifier" in out.read_text()  # license header emitted


@pytest.mark.unit
def test_appends_dirty_on_modified_tracked_file(git_repo):
    mod = _load_module()
    (git_repo / "tracked.txt").write_text("v2-uncommitted\n")  # modify tracked
    out = git_repo / "_build_info.py"
    sha = mod.write_build_info(repo_root=git_repo, output=out)

    assert sha.endswith("-dirty")
    assert _exec_git_sha(out) == sha


@pytest.mark.unit
def test_untracked_file_does_not_mark_dirty(git_repo):
    mod = _load_module()
    (git_repo / "scratch.jsonl").write_text("noise\n")  # untracked scratch
    out = git_repo / "_build_info.py"
    sha = mod.write_build_info(repo_root=git_repo, output=out)

    assert not sha.endswith("-dirty")


@pytest.mark.unit
def test_short_sha_raises_without_git(tmp_path):
    mod = _load_module()
    with pytest.raises(subprocess.CalledProcessError):
        mod._short_sha(tmp_path)  # no .git -> check=True fails loudly


@pytest.mark.unit
def test_write_is_atomic_leaves_no_tmp_file(git_repo):
    mod = _load_module()
    out = git_repo / "_build_info.py"
    mod.write_build_info(repo_root=git_repo, output=out)
    leftovers = list(git_repo.glob("_build_info.py*.tmp")) + list(
        git_repo.glob("*.tmp")
    )
    assert leftovers == []


@pytest.mark.unit
def test_short_sha_rejects_foreign_toplevel(git_repo):
    """Running against a subdir whose git toplevel != repo_root is refused."""
    mod = _load_module()
    subdir = git_repo / "sub"
    subdir.mkdir()
    with pytest.raises(RuntimeError, match="different repo"):
        mod._short_sha(subdir)


@pytest.mark.unit
def test_short_sha_ignores_stray_git_dir(git_repo, tmp_path, monkeypatch):
    """A stray GIT_DIR must not bake a foreign SHA (the toplevel guard alone passes).

    With GIT_DIR set + GIT_WORK_TREE unset, ``--show-toplevel`` reports repo_root
    (guard passes) while HEAD would resolve the foreign repo; _git_env's scrub is
    what re-anchors the bake to repo_root's own commit.
    """
    mod = _load_module()
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
    our_sha = _git("rev-parse", "--short=7", "HEAD", cwd=git_repo)
    assert foreign_sha != our_sha

    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    assert mod._short_sha(git_repo) == our_sha


@pytest.mark.unit
def test_tree_dirty_fails_to_dirty_on_probe_error(git_repo, monkeypatch):
    """A dirty-probe timeout/error is reported as dirty, mirroring runtime."""
    mod = _load_module()

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5.0)

    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod._tree_dirty(git_repo) is True


@pytest.mark.unit
def test_tree_dirty_scrubs_git_env(git_repo, monkeypatch):
    """The generator's dirty probe also runs with the git-location family scrubbed."""
    mod = _load_module()
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("GIT_INDEX_FILE", "/foreign/.git/index")
    seen = {}

    def fake_run(cmd, *args, **kwargs):
        seen["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=b"")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod._tree_dirty(git_repo) is True
    assert seen["env"] is not None
    assert "GIT_DIR" not in seen["env"]
    assert "GIT_INDEX_FILE" not in seen["env"]


@pytest.mark.unit
def test_git_location_env_vars_in_sync_with_runtime():
    """The generator's scrub list must not drift from utils.version's."""
    from inference_endpoint.utils import version

    mod = _load_module()
    assert mod._GIT_LOCATION_ENV_VARS == version._GIT_LOCATION_ENV_VARS


@pytest.mark.unit
def test_atomic_write_preserves_prior_output_on_failure(git_repo, monkeypatch):
    """A mid-write failure leaves the previous _build_info.py intact, not truncated."""
    mod = _load_module()
    out = git_repo / "_build_info.py"
    out.write_text('GIT_SHA = "previous"\n')

    original_replace = Path.replace

    def failing_replace(self, target):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        mod.write_build_info(repo_root=git_repo, output=out)
    monkeypatch.setattr(Path, "replace", original_replace)

    assert out.read_text() == 'GIT_SHA = "previous"\n'  # untouched
    assert list(git_repo.glob("*.tmp")) == []  # temp cleaned up
