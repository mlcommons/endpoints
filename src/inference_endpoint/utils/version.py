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

import os
import re
import subprocess
from pathlib import Path

from .. import __version__

# Build-time provenance for installed WHEELS: scripts/write_build_info.py writes
# this file and uv_build packs it (it selects by module tree, not git tracking,
# so the gitignored file is still included). Container images do NOT use it — the
# file is .dockerignored and images carry the SHA via the ENDPOINTS_GIT_SHA env
# channel instead, so a stray local bake can't override the build-arg. Absent in
# a plain source checkout (generated + gitignored).
try:
    from .._build_info import GIT_SHA as _BAKED_GIT_SHA
except Exception:
    # Broad by design: a missing (ImportError), truncated (SyntaxError) or
    # hand-edited unquoted (NameError) generated file must degrade to "no baked
    # SHA" rather than break version import and, with it, Report.from_snapshot.
    # KeyboardInterrupt/SystemExit are BaseException and still propagate.
    _BAKED_GIT_SHA = None  # type: ignore[assignment]

# Explicit attestation, primarily for containers/CI: `docker build --build-arg
# GIT_SHA=...` -> ENV (or a launcher) sets this for code that has neither a git
# checkout nor a baked file.
_GIT_SHA_ENV_VAR = "ENDPOINTS_GIT_SHA"

# A resolved SHA is a git hex object name (SHA-1 short/full or SHA-256, any case)
# with an optional -dirty marker. EVERY channel is validated against this so a
# sentinel ("unknown"), a stray value, a non-string baked value, or an injected
# newline/ANSI sequence cannot masquerade as an attestation or corrupt report.txt.
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,64}(?:-dirty)?")


def _valid_sha(value: object) -> str | None:
    """Return the value if it is a well-formed (optionally dirty) git SHA, else None.

    Guards against a non-string baked value (e.g. an unquoted ``GIT_SHA = 123``
    in a hand-edited _build_info.py that imports cleanly) as well as sentinels
    and injected control characters.
    """
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
# foreach``, a parent-process hook) can point git at a foreign repo/index while
# ``--show-toplevel`` still reports our cwd — slipping a foreign SHA past the
# toplevel guard, or a foreign index past the dirty probe. Scrubbing the whole
# family re-anchors every git call to its ``cwd``. Kept in sync with the
# generator's copy in scripts/write_build_info.py (stdlib-only there).
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

    Drops the repo-location family (see the note above) and sets
    ``GIT_OPTIONAL_LOCKS=0`` so ``git status`` never rewrites the caller's index.
    The env var, unlike the ``--no-optional-locks`` flag, is ignored rather than
    rejected by a git too old to know it (common on HPC hosts).
    """
    env = dict(os.environ)
    for var in _GIT_LOCATION_ENV_VARS:
        env.pop(var, None)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _git_tree_dirty(repo_root: Path) -> bool:
    """True iff tracked files have uncommitted changes (via ``git status``).

    Untracked files are ignored — only modifications to tracked content mark the
    tree dirty, so scratch artifacts a dev checkout accumulates (e.g. run output
    ``*.jsonl``) don't taint the provenance SHA. Stat-only differences (a copied,
    rsynced or re-mounted checkout) are content-checked in memory.

    Fails to dirty: anything other than exit 0 with no entries — a git error
    (e.g. a corrupt index or a dubious-ownership refusal), a timeout, or an
    OSError — is reported as dirty. An unverifiable tree is flagged ``-dirty``
    rather than silently blessed as clean, so provenance never understates
    uncertainty (e.g. a slow probe on NFS/HPC homes).
    """
    try:
        # Bytes, not text: with core.quotePath=false a non-UTF-8 path would
        # otherwise raise UnicodeDecodeError out of report building.
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            timeout=1.0,
            check=False,
            cwd=repo_root,
            env=_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return result.returncode != 0 or bool(result.stdout.strip())


def get_git_sha() -> str | None:
    """Get the git commit SHA of the endpoints source checkout via live git.

    The query is anchored to this package's own location (``_REPO_ROOT``) rather
    than the process working directory, so the SHA reflects the endpoints repo
    even when the CLI is launched from an unrelated repo.

    Returns:
        The short git SHA (at least 7 chars; git lengthens it if a 7-char
        prefix is ambiguous), suffixed with ``-dirty`` when the working tree has
        uncommitted tracked changes; or None if the package is not in a source
        checkout (e.g. an installed wheel) or git is unavailable.
    """
    repo_root = _REPO_ROOT
    if repo_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
            cwd=repo_root,
            env=_git_env(),
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
        if len(lines) != 2:
            return None
        toplevel, sha = lines
        # Only trust the SHA if the discovered repo IS this source tree (e.g. not
        # an enclosing repo whose subdirectory merely holds a checkout's src/) —
        # a wrong provenance SHA is worse than None.
        if Path(toplevel).resolve() != repo_root:
            return None
        sha = sha.strip()
        return f"{sha}-dirty" if _git_tree_dirty(repo_root) else sha
    except (OSError, subprocess.TimeoutExpired):
        return None


def resolve_git_sha() -> tuple[str | None, str]:
    """Resolve the source SHA and record which channel provided it.

    Not cached: the env var and the working-tree dirty state can change within a
    process (a launcher exporting ``ENDPOINTS_GIT_SHA`` after import; a run that
    dirties the tree; a transient dirty-probe failure that must not stick), and
    resolution is cold-path (built once per report), so it is re-evaluated on
    each call rather than frozen at first use.

    Priority (first hit wins), ordered so the value most tightly bound to the
    running code wins over a claim recorded about it elsewhere:

    1. ``"git"`` — a live ``git`` query against this package's own source
       checkout. It observes the code actually running, ``-dirty`` included, so
       while git can read the checkout, a stale bake left in it or an image env
       value under a bind-mounted checkout can't mask it. If git can't (no git,
       a dubious-ownership refusal, a timeout), resolution falls through.
    2. ``"baked"`` — ``_build_info.GIT_SHA`` packed into the wheel at build time
       from the exact packaged tree; travels with the artifact, no runtime git.
    3. ``"env"`` — the ``ENDPOINTS_GIT_SHA`` environment variable (container
       build-arg or launch-script attestation).
    4. ``"none"`` — nothing resolved; SHA is None.

    Returns:
        ``(sha, source)`` where ``source`` is one of the labels above. ``sha`` is
        None only when ``source == "none"``.
    """
    # Validate the live-git result too: get_git_sha reads git stdout, so a wrapping
    # git on PATH or polluted output could otherwise carry ANSI/newline into the
    # report. Uniform validation also keeps all three channels to one grammar.
    live_sha = _valid_sha(get_git_sha())
    if live_sha:
        return live_sha, "git"
    baked_sha = _valid_sha(_BAKED_GIT_SHA)
    if baked_sha:
        return baked_sha, "baked"
    env_sha = _valid_sha(os.environ.get(_GIT_SHA_ENV_VAR))
    if env_sha:
        return env_sha, "env"
    return None, "none"


def get_version_info() -> dict[str, str | None]:
    """Get version and git provenance information.

    Returns:
        Dictionary with 'version', 'git_sha', and 'git_sha_source' keys.
    """
    git_sha, git_sha_source = resolve_git_sha()
    return {
        "version": __version__,
        "git_sha": git_sha,
        "git_sha_source": git_sha_source,
    }
