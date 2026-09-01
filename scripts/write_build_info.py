#!/usr/bin/env python3
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

"""Bake the build-time git SHA into ``src/inference_endpoint/_build_info.py``.

Run from a source checkout (where ``.git`` exists) BEFORE building a wheel so the
SHA travels with the artifact and is readable at runtime without git::

    python scripts/write_build_info.py
    uv build

The generated file is gitignored but uv_build still packs it into the wheel (it
selects by module tree, not git tracking). ``utils.version.resolve_git_sha``
imports it as the highest-priority provenance source; without it, runtime falls
back to the ``ENDPOINTS_GIT_SHA`` env var (containers) then a live ``git`` query
(dev). Container images do NOT use this file (it is ``.dockerignore``d); they
carry the SHA via the ``ENDPOINTS_GIT_SHA`` build-arg instead.

The SHA format mirrors ``get_git_sha``: the 7-char short SHA plus a ``-dirty``
suffix when tracked files have uncommitted changes. Stdlib-only on purpose — this
runs before ``inference_endpoint`` is installed, so it cannot import from it.
"""

import os
import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO_ROOT / "src" / "inference_endpoint" / "_build_info.py"

# Build-time git budget: more lenient than runtime's snappy 1.0s (the build can
# afford to wait); intentionally not shared with utils.version (stdlib-only here).
_GIT_TIMEOUT_S = 5.0
# Bare hex object name (SHA-1 short/full or SHA-256). This is the hex CORE of
# utils.version._SHA_RE, which additionally allows a trailing "-dirty"; here the
# SHA is validated before write_build_info appends "-dirty" itself.
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,64}")

_LICENSE_HEADER = """\
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
"""


# Git's repo-location env family. A CI wrapper or ``git submodule foreach`` can
# export any of these to point git at a foreign repo/index; dropping the whole
# family (plus the toplevel check below) stops the generator from baking a SHA or
# dirty bit that isn't this source tree's. Mirrors utils.version's copy (that
# module can't be imported here — this script runs pre-install, stdlib-only).
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def _git_env() -> dict[str, str]:
    """Environment with the git repo-location family dropped (see the note above)."""
    env = dict(os.environ)
    for var in _GIT_LOCATION_ENV_VARS:
        env.pop(var, None)
    return env


def _short_sha(repo_root: Path) -> str:
    """Return the validated short SHA of the repo rooted at ``repo_root``.

    Verifies the discovered toplevel IS ``repo_root`` (mirrors get_git_sha's
    foreign-repo guard) and that the value is hex, so the generator cannot bake a
    foreign or malformed SHA that runtime would otherwise trust as baked ground
    truth. Raises on any mismatch so a bad bake fails loudly instead of writing
    a value runtime silently discards.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "--short=7", "HEAD"],
        cwd=repo_root,
        env=_git_env(),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=True,
    )
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        raise RuntimeError(f"unexpected git rev-parse output: {result.stdout!r}")
    toplevel, sha = lines
    if Path(toplevel).resolve() != repo_root.resolve():
        raise RuntimeError(
            f"refusing to bake a SHA from a different repo: git toplevel "
            f"{toplevel!r} != {repo_root}"
        )
    sha = sha.strip()
    if not _SHA_RE.fullmatch(sha):
        raise RuntimeError(f"git returned a non-hex SHA: {sha!r}")
    return sha


def _tree_dirty(repo_root: Path) -> bool:
    """True iff tracked files have uncommitted changes; fails to dirty.

    Mirrors ``utils.version._git_tree_dirty``: untracked files are ignored, and
    any outcome other than a clean exit 0 (non-zero code, timeout, OSError) is
    reported as dirty so a build never bakes a clean SHA it could not verify.

    Caveat (matches ``git describe --dirty`` semantics): a NEW untracked file
    under the packaged tree (``src/inference_endpoint/``) is NOT flagged, yet
    uv_build would pack it — so baking + building a new module without committing
    it yields a clean SHA for a wheel whose code differs from that commit. Commit
    new modules before baking.
    """
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=repo_root,
            env=_git_env(),
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    return result.returncode != 0


def write_build_info(repo_root: Path = _REPO_ROOT, output: Path = _OUTPUT) -> str:
    """Generate ``output`` with the (validated) resolved SHA and return that SHA."""
    sha = _short_sha(repo_root)
    if _tree_dirty(repo_root):
        sha = f"{sha}-dirty"
    content = (
        f"{_LICENSE_HEADER}\n"
        '"""Generated by scripts/write_build_info.py at build time. Do not edit or commit."""\n'
        "\n"
        # _short_sha validated the SHA is hex; !r additionally guarantees the
        # generated module parses regardless of the value.
        f"GIT_SHA = {sha!r}\n"
    )
    # Atomic: write a sibling temp file then rename, so a crash mid-write cannot
    # leave a truncated _build_info.py that would SyntaxError on import. The
    # finally cleans up the temp on any failure before the rename.
    tmp = output.with_name(output.name + ".tmp")
    try:
        tmp.write_text(content)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return sha


def main() -> None:
    sha = write_build_info()
    print(f"Wrote {_OUTPUT} (GIT_SHA={sha})")


if __name__ == "__main__":
    main()
