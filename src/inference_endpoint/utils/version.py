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

import subprocess
from functools import lru_cache
from pathlib import Path

from .. import __version__


def _resolve_repo_root() -> Path:
    """Repo root of this source checkout, anchored to the module location.

    ``src/inference_endpoint/utils/version.py`` -> ``parents[3]`` is the repo root.
    Falls back to the filesystem root when the package lives in a shallower tree
    (e.g. an installed wheel copied to ``/inference_endpoint``) so import never
    crashes; get_git_sha's toplevel guard then yields None, not a foreign SHA.
    """
    parents = Path(__file__).resolve().parents
    return parents[3] if len(parents) > 3 else parents[-1]


_REPO_ROOT = _resolve_repo_root()


@lru_cache(maxsize=1)
def get_git_sha() -> str | None:
    """Get the git commit SHA of the endpoints source checkout.

    The query is anchored to this package's own location (``_REPO_ROOT``) rather
    than the process working directory, so the SHA reflects the endpoints repo
    even when the CLI is launched from an unrelated repo.

    Returns:
        The short git SHA (at least 7 chars; git lengthens it if a 7-char
        prefix is ambiguous), or None if the package is not in a git checkout
        (e.g. an installed wheel) or git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
            cwd=_REPO_ROOT,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.splitlines()
        if len(lines) != 2:
            return None
        toplevel, sha = lines
        # Only trust the SHA if the discovered repo IS this source tree. Guards against
        # an installed wheel nested inside an unrelated repo reporting a foreign SHA —
        # a wrong provenance SHA is worse than None.
        if Path(toplevel).resolve() != _REPO_ROOT:
            return None
        return sha.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=1)
def get_version_info() -> dict[str, str | None]:
    """Get version and git information.

    Returns:
        Dictionary with 'version' and 'git_sha' keys.
    """
    return {
        "version": __version__,
        "git_sha": get_git_sha(),
    }
