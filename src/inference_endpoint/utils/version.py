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

from functools import lru_cache

from .. import __version__


@lru_cache(maxsize=1)
def get_git_sha() -> str | None:
    """Short commit SHA of the build, parsed from the VCS-derived ``__version__``.

    hatch-vcs encodes the commit as the PEP 440 local segment ``+g<sha>`` (e.g.
    ``0.6.dev3+g6eac351`` or ``...+g6eac351.d20260626`` when dirty). A clean tagged
    release has no local segment and returns None — ``__version__`` itself is the
    canonical, self-identifying version, so this is just a convenience accessor.
    """
    local = __version__.partition("+")[2]
    for part in local.split("."):
        if part.startswith("g") and len(part) > 1:
            return part[1:]
    return None


@lru_cache(maxsize=1)
def get_version_info() -> dict[str, str | None]:
    """Get version and git information.

    Returns:
        Dictionary with 'version' (VCS-derived, SHA-embedded) and 'git_sha' keys.
    """
    return {
        "version": __version__,
        "git_sha": get_git_sha(),
    }
