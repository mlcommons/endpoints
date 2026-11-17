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

"""Version and git information utilities."""

import functools
import logging
import os
import re
import subprocess
from pathlib import Path

from .. import __version__

logger = logging.getLogger(__name__)

# Attestation for code running outside a git checkout, e.g. a container image
# built with `--build-arg GIT_SHA=...` (scripts/Dockerfile.dev).
_GIT_SHA_ENV_VAR = "ENDPOINTS_GIT_SHA"

# A git hex object name (SHA-1 short/full or SHA-256, any case) with an optional
# -dirty marker. Every channel is validated so a sentinel ("unknown") or an
# injected newline/ANSI sequence can't pose as a SHA or corrupt report.txt.
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,64}(?:-dirty)?")

# Per-probe budget. Provenance is resolved once per process before the run
# starts, so slow filesystems (NFS/HPC homes) can afford it.
_GIT_TIMEOUT_S = 5.0

# What "dirty" covers: code a run executes and its dependency pins, i.e. the
# package plus the VBench scorer subproject it launches from examples/. Edits
# elsewhere (other examples, docs, LFS datasets) don't change what runs.
# scripts/push_docker_image.sh and scripts/Dockerfile.dev repeat this list.
_DIRTY_PATHSPEC = (
    "src/inference_endpoint",
    "pyproject.toml",
    "uv.lock",
    "examples/09_Wan22_VideoGen_Example/accuracy",
)


def _valid_sha(value: object) -> str | None:
    """Return ``value`` stripped if it is a well-formed (optionally dirty) SHA."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if _SHA_RE.fullmatch(value) else None


def _resolve_repo_root() -> Path | None:
    """Repo root of this source checkout, anchored to the module location.

    Only the source layout ``<root>/src/inference_endpoint/utils/version.py`` has
    one. Any other location (an installed wheel, a copy vendored inside another
    repo) yields None, so get_git_sha never asks git about a foreign repo.
    """
    parents = Path(__file__).resolve().parents
    if len(parents) < 4 or [p.name for p in parents[2::-1]] != [
        "src",
        "inference_endpoint",
        "utils",
    ]:
        return None
    return parents[3]


_REPO_ROOT = _resolve_repo_root()


# Git's repo-location env family. A stray value (a CI wrapper, ``git submodule
# foreach``) can point git at a foreign repo/index while ``--show-toplevel`` still
# reports our cwd, slipping a foreign SHA past the toplevel guard or a foreign
# index past the dirty probe. Scrubbing it anchors every probe to its ``cwd``.
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _git_env() -> dict[str, str]:
    """Environment for read-only git probes.

    Drops the repo-location family and sets ``GIT_OPTIONAL_LOCKS=0`` so
    ``git status`` never rewrites the caller's index. A git too old to know the
    variable ignores it, where it would reject ``--no-optional-locks``.
    """
    env = dict(os.environ)
    for var in _GIT_LOCATION_ENV_VARS:
        env.pop(var, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _git_tree_dirty(repo_root: Path) -> bool:
    """True iff ``_DIRTY_PATHSPEC`` has uncommitted tracked changes.

    Untracked files don't count. ``core.checkStat=minimal`` compares only mtime
    and size, so a copied checkout, or a bind mount presenting new inode numbers,
    isn't re-hashed: re-hashing a git-lfs file where git-lfs isn't installed would
    always report it modified.

    Fails to dirty: a git error (e.g. a dubious-ownership refusal), a timeout or
    an OSError reports dirty, with a warning, so an unverifiable tree is never
    blessed as clean.
    """
    try:
        # Bytes, not text: with core.quotePath=false a non-UTF-8 path would
        # otherwise raise UnicodeDecodeError.
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.checkStat=minimal",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                *_DIRTY_PATHSPEC,
            ],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            cwd=repo_root,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        reason = str(e)
    else:
        if result.returncode == 0:
            return bool(result.stdout.strip())
        stderr = result.stderr.decode(errors="replace").strip()
        reason = f"git status exited {result.returncode}: {stderr}"
    logger.warning(
        "Could not verify %s is clean (%s); reporting -dirty", repo_root, reason
    )
    return True


def get_git_sha() -> str | None:
    """Get the git commit SHA of the endpoints source checkout via live git.

    The query is anchored to this package's own location (``_REPO_ROOT``) rather
    than the process working directory, so the SHA reflects the endpoints repo
    even when the CLI is launched from an unrelated repo.

    Returns:
        The short git SHA (at least 7 chars; git lengthens it if a 7-char
        prefix is ambiguous), suffixed with ``-dirty`` when ``_DIRTY_PATHSPEC`` has
        uncommitted tracked changes; or None if the package is not in a source
        checkout (e.g. an installed wheel) or git is unavailable.
    """
    repo_root = _REPO_ROOT
    if repo_root is None:
        return None
    try:
        # Bytes, not text: --show-toplevel prints the path raw, and a directory
        # name need not be UTF-8.
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--short=7", "HEAD"],
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            cwd=repo_root,
            env=_git_env(),
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
        if len(lines) != 2:
            return None
        toplevel = os.fsdecode(lines[0])
        sha = lines[1].decode("ascii", "replace").strip()
        # Only trust the SHA if the discovered repo IS this source tree (e.g. not
        # an enclosing repo whose subdirectory merely holds a checkout's src/) —
        # a wrong provenance SHA is worse than None.
        if Path(toplevel).resolve() != repo_root:
            return None
        return f"{sha}-dirty" if _git_tree_dirty(repo_root) else sha
    except (OSError, subprocess.TimeoutExpired):
        return None


def _in_git_checkout() -> bool:
    # lexists: a .git symlink whose target isn't visible (outside a bind mount)
    # still marks a checkout.
    return _REPO_ROOT is not None and os.path.lexists(_REPO_ROOT / ".git")


def resolve_git_sha() -> tuple[str | None, str]:
    """Resolve the source SHA and record which channel provided it.

    1. ``"git"`` — a live query of this package's own source checkout,
       ``-dirty`` included.
    2. ``"env"`` — ``ENDPOINTS_GIT_SHA``, honored only when the package is not in
       a git checkout (an installed package, or an image's copy of ``src/``).
       A checkout git can't read (git missing, a timeout, a dubious-ownership
       refusal, a ``.git`` pointing outside a bind mount) reports nothing
       instead: the variable describes whichever image or launcher set it, not
       this tree. Code mounted without its ``.git`` (a tarball, or ``src/``
       alone) can't be told apart from the image's copy and gets the variable.
    3. ``"none"`` — nothing resolved.

    Every value is validated, so a wrapping git on PATH can't carry ANSI or
    newlines into the report.

    Returns:
        ``(sha, source)`` where ``source`` is one of the labels above. ``sha`` is
        None only when ``source == "none"``.
    """
    live_sha = _valid_sha(get_git_sha())
    if live_sha:
        return live_sha, "git"
    if _in_git_checkout():
        logger.warning(
            "Not reporting a git SHA: %s is a git checkout but git could not "
            "read it (e.g. git missing, a timeout, dubious ownership, or a .git "
            "pointing outside a mount)",
            _REPO_ROOT,
        )
        return None, "none"
    env_sha = _valid_sha(os.environ.get(_GIT_SHA_ENV_VAR))
    if env_sha:
        return env_sha, "env"
    return None, "none"


@functools.cache
def _process_git_sha() -> tuple[str | None, str]:
    return resolve_git_sha()


def get_version_info() -> dict[str, str | None]:
    """Get version and git provenance information.

    The SHA is resolved once per process, on the first call (``setup_benchmark``
    makes it before the run starts), so every report the process writes
    describes the checkout as it was at setup rather than whenever a report is
    built. Subprocesses started later (e.g. audit-phase services) import
    whatever the checkout holds then.

    Returns:
        Dictionary with 'version', 'git_sha', and 'git_sha_source' keys.
    """
    git_sha, git_sha_source = _process_git_sha()
    return {
        "version": __version__,
        "git_sha": git_sha,
        "git_sha_source": git_sha_source,
    }
