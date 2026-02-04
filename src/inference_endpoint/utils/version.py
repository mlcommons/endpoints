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

from .. import __version__


@lru_cache(maxsize=1)
def get_git_sha() -> str | None:
    """Get the current git commit SHA.

    Returns:
        The short git SHA (7 chars) or None if not in a git repo or git unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
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
