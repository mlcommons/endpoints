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

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from inference_endpoint import __version__
from inference_endpoint.utils import version as version_mod

# Fixed mtime for files committed by _commit_tree. Keeping it well before the
# index write means no entry is "racily clean", so git trusts the index stat data.
_PAST_MTIME = 1_000_000_000


@pytest.fixture(autouse=True)
def _fresh_process_provenance():
    version_mod._process_git_sha.cache_clear()
    yield
    version_mod._process_git_sha.cache_clear()


def _hermetic_git_env() -> dict[str, str]:
    """Env for git calls that build throwaway repos.

    Drops every ambient ``GIT_*`` variable (a stray ``GIT_DIR`` would redirect
    commits into a foreign repo) and the user's global/system config (commit
    signing, hooks, templates), and supplies an identity.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t.co",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t.co",
    )
    return env


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_hermetic_git_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit_tree(repo: Path, files: dict[str, str], *add_flags: str) -> Path:
    """Create a repo at ``repo`` holding ``files`` in one commit."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo.mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        os.utime(path, (_PAST_MTIME, _PAST_MTIME))
    _git(*add_flags, "add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    return repo


def _real_checkout_sha() -> str:
    """Short SHA of the checkout this package runs from; skip if there is none.

    Reads the checkout the way get_git_sha does: ambient config (so CI's
    safe.directory applies) without a stray GIT_DIR.
    """
    root = version_mod._REPO_ROOT
    if shutil.which("git") is None or root is None:
        pytest.skip("not running from a git source checkout")

    def rev_parse(*args: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", *args],
            cwd=root,
            env=version_mod._git_env(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        toplevel = rev_parse("--show-toplevel")
        sha = rev_parse("--short=7", "HEAD")
    except subprocess.CalledProcessError:
        pytest.skip("source tree is not a git checkout")
    if Path(toplevel).resolve() != root:
        pytest.skip("source tree is nested inside another repo")
    return sha


def _fake_git(rev_parse=None, status=b""):
    """subprocess.run stand-in dispatching on the git subcommand.

    Each outcome is stdout bytes, a CompletedProcess, or an exception to raise;
    ``rev_parse`` defaults to this checkout's toplevel plus ``abc1234``. Output is
    decoded whenever the caller's arguments would make the real call decode it,
    with the same encoding and error handler, so a probe switched to text mode
    fails here too.
    """
    if rev_parse is None:
        rev_parse = os.fsencode(version_mod._REPO_ROOT) + b"\nabc1234\n"

    def run(cmd, *args, **kwargs):
        if "rev-parse" in cmd:
            outcome = rev_parse
        elif "status" in cmd:
            outcome = status
        else:
            raise AssertionError(f"unexpected git invocation: {cmd}")
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, bytes):
            outcome = subprocess.CompletedProcess(cmd, 0, stdout=outcome)
        text_keys = ("text", "universal_newlines", "encoding", "errors")
        if any(kwargs.get(key) for key in text_keys):
            stdout = outcome.stdout.decode(
                kwargs.get("encoding") or "utf-8", kwargs.get("errors") or "strict"
            )
            outcome = subprocess.CompletedProcess(
                outcome.args, outcome.returncode, stdout=stdout, stderr=outcome.stderr
            )
        return outcome

    return run


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
    assert version_mod._valid_sha(value) == value.strip()


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        None,
        1234567,  # non-string
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
    assert version_mod._valid_sha(value) is None


# ---------------------------------------------------------------------------
# get_git_sha (live-git channel)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_git_sha_reports_a_valid_sha_or_none():
    sha = version_mod.get_git_sha()
    assert sha is None or version_mod._valid_sha(sha) == sha


@pytest.mark.unit
def test_get_git_sha_returns_bare_sha_for_clean_tree(monkeypatch):
    monkeypatch.setattr(version_mod.subprocess, "run", _fake_git(status=b""))
    assert version_mod.get_git_sha() == "abc1234"


@pytest.mark.unit
def test_get_git_sha_appends_dirty_when_tree_has_tracked_changes(monkeypatch):
    monkeypatch.setattr(
        version_mod.subprocess, "run", _fake_git(status=b" M src/x.py\n")
    )
    assert version_mod.get_git_sha() == "abc1234-dirty"


@pytest.mark.unit
@pytest.mark.parametrize(
    "status",
    [
        subprocess.CompletedProcess(
            [], 128, stdout=b"", stderr=b"fatal: index file corrupt"
        ),
        subprocess.TimeoutExpired(cmd="git", timeout=5.0),
        OSError("exec failed"),
    ],
    ids=["error-returncode", "timeout", "oserror"],
)
def test_dirty_probe_failure_reports_dirty_and_warns(monkeypatch, caplog, status):
    """An unverifiable tree is flagged -dirty, and the log says it wasn't verified."""
    monkeypatch.setattr(version_mod.subprocess, "run", _fake_git(status=status))
    with caplog.at_level("WARNING", logger=version_mod.__name__):
        assert version_mod.get_git_sha() == "abc1234-dirty"
    assert "Could not verify" in caplog.text


@pytest.mark.unit
def test_git_probes_use_cold_path_timeout(monkeypatch):
    timeouts = []
    fake = _fake_git()

    def run(cmd, *args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return fake(cmd, *args, **kwargs)

    monkeypatch.setattr(version_mod.subprocess, "run", run)
    version_mod.get_git_sha()
    assert timeouts == [5.0, 5.0]


@pytest.mark.unit
def test_dirty_probe_reads_non_utf8_status_output(tmp_path, monkeypatch):
    """A non-UTF-8 path in status output (core.quotePath=false) reads as dirty."""
    monkeypatch.setattr(
        version_mod.subprocess, "run", _fake_git(status=b" M caf\xe9.txt\n")
    )
    assert version_mod._git_tree_dirty(tmp_path) is True


@pytest.mark.unit
def test_get_git_sha_reads_non_utf8_checkout_path(monkeypatch):
    """A checkout under a non-UTF-8 directory name still resolves (Linux allows it)."""
    raw_root = b"/nonexistent/caf\xe9/endpoints"
    monkeypatch.setattr(
        version_mod, "_REPO_ROOT", Path(os.fsdecode(raw_root)).resolve()
    )
    monkeypatch.setattr(
        version_mod.subprocess,
        "run",
        _fake_git(rev_parse=raw_root + b"\nabc1234\n", status=b""),
    )
    assert version_mod.get_git_sha() == "abc1234"


@pytest.mark.unit
def test_non_ascii_sha_line_is_read_and_then_rejected(monkeypatch):
    """A wrapping git printing garbage for the SHA can't crash resolution."""
    root = os.fsencode(version_mod._REPO_ROOT)
    monkeypatch.setattr(
        version_mod.subprocess,
        "run",
        _fake_git(rev_parse=root + b"\n\xff\xfe\n", status=b""),
    )
    sha = version_mod.get_git_sha()
    assert sha is not None
    assert version_mod._valid_sha(sha) is None


@pytest.mark.unit
def test_get_git_sha_none_outside_source_layout(monkeypatch):
    """Without a source-checkout root, git is never consulted."""

    def fail_run(*args, **kwargs):
        raise AssertionError("git must not run outside a source checkout")

    monkeypatch.setattr(version_mod, "_REPO_ROOT", None)
    monkeypatch.setattr(version_mod.subprocess, "run", fail_run)
    assert version_mod.get_git_sha() is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "rev_parse",
    [
        FileNotFoundError(),
        subprocess.TimeoutExpired(cmd="git", timeout=5.0),
        subprocess.CompletedProcess([], 128, stdout=b"", stderr=b"x"),
        b"only-one-line\n",
        b"/some/other/repo\ndeadbee\n",
    ],
    ids=["no-git", "timeout", "not-a-repo", "short-output", "foreign-toplevel"],
)
def test_get_git_sha_none_on_untrusted_or_missing_repo(rev_parse, monkeypatch):
    """A foreign/absent repo yields None rather than a wrong provenance SHA."""
    monkeypatch.setattr(version_mod.subprocess, "run", _fake_git(rev_parse=rev_parse))
    assert version_mod.get_git_sha() is None


@pytest.mark.unit
def test_get_git_sha_is_endpoints_repo_not_cwd(tmp_path, monkeypatch):
    """Launching from an unrelated repo must not record that repo's SHA."""
    expected = _real_checkout_sha()
    other = _commit_tree(tmp_path / "other_repo", {"f.txt": "unrelated\n"})
    other_sha = _git("rev-parse", "--short=7", "HEAD", cwd=other)
    assert other_sha != expected

    monkeypatch.chdir(other)
    sha = version_mod.get_git_sha()

    assert sha is not None
    assert sha.removesuffix("-dirty") == expected


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
    for var in _GIT_LOCATION_VARS:
        monkeypatch.setenv(var, "/foreign")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = version_mod._git_env()
    for var in _GIT_LOCATION_VARS:
        assert var not in env
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["PATH"] == "/usr/bin"


@pytest.mark.unit
def test_get_git_sha_passes_scrubbed_env_to_both_probes(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("GIT_INDEX_FILE", "/foreign/.git/index")
    seen_envs = []
    fake = _fake_git(status=b" M src/x.py\n")

    def run(cmd, *args, **kwargs):
        seen_envs.append(kwargs.get("env"))
        return fake(cmd, *args, **kwargs)

    monkeypatch.setattr(version_mod.subprocess, "run", run)
    assert version_mod.get_git_sha() == "abc1234-dirty"
    assert len(seen_envs) == 2  # rev-parse + status
    for env in seen_envs:
        assert env is not None
        assert "GIT_DIR" not in env
        assert "GIT_INDEX_FILE" not in env


@pytest.mark.unit
def test_get_git_sha_ignores_stray_git_dir(tmp_path, monkeypatch):
    """A stray GIT_DIR must not slip a foreign repo's SHA past the toplevel guard.

    With GIT_DIR set (and GIT_WORK_TREE unset) git defaults the work-tree to cwd,
    so ``--show-toplevel`` reports our repo (guard passes) while ``HEAD`` would
    resolve the foreign repo.
    """
    our_sha = _real_checkout_sha()
    foreign = _commit_tree(tmp_path / "foreign", {"f.txt": "foreign\n"})
    foreign_sha = _git("rev-parse", "--short=7", "HEAD", cwd=foreign)
    assert foreign_sha != our_sha

    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.delenv("GIT_WORK_TREE", raising=False)
    sha = version_mod.get_git_sha()

    assert sha is not None
    assert sha.removesuffix("-dirty") == our_sha


# ---------------------------------------------------------------------------
# _git_tree_dirty (real git)
# ---------------------------------------------------------------------------

_VBENCH_RUNNER = "examples/09_Wan22_VideoGen_Example/accuracy/vbench_runner.py"
_LAYOUT = {
    "src/inference_endpoint/mod.py": "x = 1\n",
    "pyproject.toml": "[project]\n",
    "uv.lock": "version = 1\n",
    _VBENCH_RUNNER: "run()\n",
    "examples/data.bin": "payload\n",
    "docs/notes.md": "notes\n",
}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "dirty"),
    [
        ("src/inference_endpoint/mod.py", True),
        ("pyproject.toml", True),
        ("uv.lock", True),
        (_VBENCH_RUNNER, True),
        ("examples/data.bin", False),
        ("docs/notes.md", False),
    ],
)
def test_dirty_probe_covers_code_that_runs_and_its_pins(tmp_path, path, dirty):
    repo = _commit_tree(tmp_path / "repo", _LAYOUT)
    (repo / path).write_text("edited\n")
    assert version_mod._git_tree_dirty(repo) is dirty


@pytest.mark.unit
@pytest.mark.parametrize(
    "script", ["scripts/push_docker_image.sh", "scripts/Dockerfile.dev"]
)
def test_image_build_stamps_sha_like_the_runtime_probe(script):
    """Image builds compute GIT_SHA with the runtime's SHA length and dirty scope."""
    text = (Path(__file__).resolve().parents[2] / script).read_text()
    assert "rev-parse --short=7 HEAD" in text
    assert f"git diff --quiet HEAD -- {' '.join(version_mod._DIRTY_PATHSPEC)}" in text


@pytest.mark.unit
def test_dirty_probe_treats_copied_checkout_as_clean(tmp_path):
    """New inodes/ctimes with unchanged content are clean; untracked files don't count."""
    repo = _commit_tree(tmp_path / "repo", _LAYOUT)
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    (copy / "src" / "inference_endpoint" / "scratch.jsonl").write_text("untracked\n")

    assert version_mod._git_tree_dirty(copy) is False
    (copy / "src" / "inference_endpoint" / "mod.py").write_text("x = 2\n")
    assert version_mod._git_tree_dirty(copy) is True


@pytest.mark.unit
def test_dirty_probe_never_rewrites_the_index(tmp_path):
    """A probe that refreshes stale stat data leaves the caller's index untouched."""
    repo = _commit_tree(tmp_path / "repo", _LAYOUT)
    stale = repo / "src" / "inference_endpoint" / "mod.py"
    os.utime(stale, (_PAST_MTIME + 60, _PAST_MTIME + 60))  # new mtime, same content
    index_before = (repo / ".git" / "index").read_bytes()

    assert version_mod._git_tree_dirty(repo) is False
    assert (repo / ".git" / "index").read_bytes() == index_before


@pytest.mark.unit
def test_dirty_probe_skips_rehash_of_filtered_file_on_stat_only_change(tmp_path):
    """A filtered file (git-lfs style) with only inode/ctime changes stays clean.

    The repo is committed through a clean filter that isn't configured afterwards,
    like an LFS checkout opened where git-lfs isn't installed (a container over a
    Docker Desktop bind mount, which presents new inode numbers). Re-hashing the
    raw file can never match the filtered blob, so only a stat check limited to
    mtime and size reports the tree correctly.
    """
    repo = _commit_tree(
        tmp_path / "repo",
        {
            ".gitattributes": "src/inference_endpoint/blob.bin filter=fake\n",
            "src/inference_endpoint/blob.bin": "aaaa\n",
        },
        "-c",
        "filter.fake.clean=tr a b",
    )
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    assert version_mod._git_tree_dirty(copy) is False


# ---------------------------------------------------------------------------
# resolve_git_sha (channel precedence + validation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("live", "env", "expected"),
    [
        ("0000fff-dirty", "def5678", ("0000fff-dirty", "git")),
        (None, "def5678", ("def5678", "env")),
        (None, "ABC1234-dirty", ("ABC1234-dirty", "env")),
        (None, None, (None, "none")),
        (None, "", (None, "none")),
        (None, "   ", (None, "none")),
        (None, "unknown", (None, "none")),
        (None, "not-a-sha", (None, "none")),
        ("abc1234\x1b[31m", "def5678", ("def5678", "env")),
    ],
    ids=[
        "git-wins",
        "env",
        "env-dirty-uppercase",
        "nothing",
        "env-blank",
        "env-whitespace",
        "env-sentinel",
        "env-non-hex",
        "polluted-git-falls-through",
    ],
)
def test_resolve_git_sha_outside_a_checkout(tmp_path, monkeypatch, live, env, expected):
    """Without a .git at the source root, live git then env, both validated."""
    monkeypatch.setattr(version_mod, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(version_mod, "get_git_sha", lambda: live)
    if env is None:
        monkeypatch.delenv("ENDPOINTS_GIT_SHA", raising=False)
    else:
        monkeypatch.setenv("ENDPOINTS_GIT_SHA", env)
    assert version_mod.resolve_git_sha() == expected


@pytest.mark.unit
def test_resolve_git_sha_uses_env_for_installed_package(monkeypatch):
    monkeypatch.setattr(version_mod, "_REPO_ROOT", None)
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")
    assert version_mod.resolve_git_sha() == ("def5678", "env")


@pytest.mark.unit
@pytest.mark.parametrize(
    "make_git",
    [
        lambda git: git.write_text("gitdir: /nonexistent/.git/worktrees/x\n"),
        lambda git: git.symlink_to("/nonexistent/.git"),
    ],
    ids=["gitfile-outside-mount", "dangling-symlink"],
)
def test_unreadable_checkout_reports_none_not_env(
    tmp_path, monkeypatch, caplog, make_git
):
    """A checkout git can't read reports no SHA; the env SHA describes other code.

    Models a bind-mounted checkout whose ``.git`` points outside the mount: the
    image's ENDPOINTS_GIT_SHA names the image's commit, not this tree's.
    """
    root = tmp_path / "mnt"
    (root / "src" / "inference_endpoint").mkdir(parents=True)
    make_git(root / ".git")
    monkeypatch.setattr(version_mod, "_REPO_ROOT", root.resolve())
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")

    with caplog.at_level("WARNING", logger=version_mod.__name__):
        assert version_mod.resolve_git_sha() == (None, "none")
    assert str(root.resolve()) in caplog.text


@pytest.mark.unit
def test_source_tree_without_git_uses_env(tmp_path, monkeypatch):
    """A source layout with no .git (an image's COPY of src/) takes the env SHA."""
    root = tmp_path / "image"
    (root / "src" / "inference_endpoint").mkdir(parents=True)
    monkeypatch.setattr(version_mod, "_REPO_ROOT", root.resolve())
    monkeypatch.setenv("ENDPOINTS_GIT_SHA", "def5678")
    assert version_mod.resolve_git_sha() == ("def5678", "env")


# ---------------------------------------------------------------------------
# get_version_info
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_version_info_shape():
    info = version_mod.get_version_info()
    assert set(info) == {"version", "git_sha", "git_sha_source"}
    assert info["version"] == __version__
    assert info["git_sha_source"] in {"env", "git", "none"}
    assert (info["git_sha"] is None) == (info["git_sha_source"] == "none")


@pytest.mark.unit
def test_get_version_info_resolves_once_per_process(monkeypatch):
    """Every report a process writes describes the code it started with."""
    calls = []

    def resolve():
        calls.append(None)
        return ("abc1234", "git")

    monkeypatch.setattr(version_mod, "resolve_git_sha", resolve)
    first = version_mod.get_version_info()
    first["git_sha"] = "mutated"
    second = version_mod.get_version_info()

    assert second == {
        "version": __version__,
        "git_sha": "abc1234",
        "git_sha_source": "git",
    }
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# _resolve_repo_root
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_repo_root_in_source_layout(monkeypatch):
    monkeypatch.setattr(
        version_mod, "__file__", "/repo/src/inference_endpoint/utils/version.py"
    )
    assert version_mod._resolve_repo_root() == Path("/repo").resolve()


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_path",
    [
        "/inference_endpoint/utils/version.py",
        "/venv/lib/python3.12/site-packages/inference_endpoint/utils/version.py",
        "/other_repo/vendor/inference_endpoint/utils/version.py",
    ],
    ids=["too-shallow", "site-packages", "vendored"],
)
def test_resolve_repo_root_none_outside_source_layout(monkeypatch, module_path):
    monkeypatch.setattr(version_mod, "__file__", module_path)
    assert version_mod._resolve_repo_root() is None
