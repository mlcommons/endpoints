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

# Substituted with the full commit SHA by `git archive` export-subst (see
# .gitattributes); left as the literal "$Format:%H$" placeholder in a working tree.
_GIT_ARCHIVAL = Path(__file__).resolve().parent.parent / "_git_archival.txt"


@lru_cache(maxsize=1)
def get_git_sha() -> str | None:
    """Get the SHA of the *endpoints package's* git commit (short, 7 chars).

    Resolution order:
      1. ``_git_archival.txt``, substituted by ``git archive`` export-subst at build
         time — sdist / git-archive builds (e.g. the container image) carry no
         ``.git``, so this is the only source there.
      2. ``git rev-parse`` anchored to this package's own directory, so a dev/editable
         checkout reports the endpoints repo's HEAD — not the caller's cwd, which may
         be an unrelated git repo.

    Returns the short SHA, or None if neither source is available.
    """
    try:
        baked = _GIT_ARCHIVAL.read_text().strip().removeprefix("node:").strip()
    except OSError:
        baked = ""
    if len(baked) == 40 and all(c in "0123456789abcdef" for c in baked.lower()):
        return baked[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=_GIT_ARCHIVAL.parent,
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
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
