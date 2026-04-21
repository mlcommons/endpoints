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

"""Filesystem type detection for mmap ordering decisions.

On tmpfs (/dev/shm), msync() is a no-op because there is no backing store.
On a real on-disk filesystem, msync() flushes dirty pages to the shared page
cache, which provides write ordering for cross-process mmap readers.

On ARM (weak memory model), we need msync() to act as an ordering mechanism
between the value write and the count update in _SeriesItem.append(). This
only works on a real filesystem — not tmpfs. Detecting the filesystem type
lets us:
  - Skip the useless msync() syscall on tmpfs (any architecture)
  - Warn if ARM code is running on tmpfs (msync won't provide ordering)
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)

_TMPFS_MAGIC = 0x01021994
"""Special tmpfs filesystem header value."""


def _is_tmpfs_via_statfs(path: str) -> bool | None:
    """Check filesystem type via libc statfs(2). Returns None if unavailable."""
    try:
        lib_name = ctypes.util.find_library("c")
        if lib_name is None:
            return None
        libc = ctypes.CDLL(lib_name, use_errno=True)

        # Allocate a large buffer to account for differently sized statfs
        # structs across architectures. f_type is always the first field
        # (__SWORD_TYPE / long) at offset 0 on all Linux archs.
        buf = ctypes.create_string_buffer(256)
        if libc.statfs(path.encode(), buf) != 0:
            return None
        # f_type is a native-endian long at offset 0
        f_type = ctypes.c_long.from_buffer(buf, 0).value
        return f_type == _TMPFS_MAGIC
    except (OSError, AttributeError, ValueError):
        return None


def _is_tmpfs_via_proc_mounts(path: str) -> bool | None:
    """Check filesystem type via /proc/mounts. Returns None if unavailable."""
    try:
        resolved = str(Path(path).resolve())
        best_match = ""
        best_fstype = ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fstype = parts[1], parts[2]
                if resolved.startswith(mount_point) and len(mount_point) > len(
                    best_match
                ):
                    best_match = mount_point
                    best_fstype = fstype
        if not best_match:
            return None
        return best_fstype == "tmpfs"
    except OSError:
        return None


def is_tmpfs(path: str | Path) -> bool:
    """Check if a path resides on a tmpfs filesystem.

    Tries statfs(2) via ctypes first, falls back to /proc/mounts.
    Returns False if detection fails (safe default — will call msync).
    """
    path_str = str(path)

    result = _is_tmpfs_via_statfs(path_str)
    if result is not None:
        return result

    result = _is_tmpfs_via_proc_mounts(path_str)
    if result is not None:
        return result

    logger.warning(
        "Could not determine filesystem type for %s "
        "(statfs and /proc/mounts both unavailable). "
        "Assuming non-tmpfs (msync will be called on every series append).",
        path_str,
    )
    return False


def needs_msync(path: str | Path) -> bool:
    """Determine if msync() is needed for mmap write ordering at this path.

    Returns True if msync should be called between value write and count
    update in series append. This is needed on ARM when the backing store
    is a real filesystem (not tmpfs).

    On x86-64 (TSO), store ordering is guaranteed by hardware — msync is
    never needed regardless of filesystem type.

    On ARM with tmpfs, msync is a no-op and won't help — log a warning
    since the caller should use an on-disk directory for correct ordering.
    """
    if platform.machine() == "x86_64":
        return False

    on_tmpfs = is_tmpfs(path)
    if on_tmpfs:
        logger.warning(
            "ARM platform with tmpfs-backed metrics at %s. "
            "Python does not support memory fences. "
            "Use an on-disk metrics directory for correct cross-process reads.",
            path,
        )
        return False

    return True
