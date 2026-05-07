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

"""Info command — display system information."""

import logging
import os
import platform
import socket
import sys

import psutil

from .. import __version__

logger = logging.getLogger(__name__)


def execute_info() -> None:
    """Display system information and tool version."""
    lines = [
        f"Inference Endpoint Benchmarking Tool v{__version__}",
        "",
        "=== System Information ===",
        "",
        "Python Environment:",
        f"  Version: {sys.version}",
        f"  Implementation: {platform.python_implementation()}",
        f"  Compiler: {platform.python_compiler()}",
        "",
        "Operating System:",
        f"  System: {platform.system()}",
        f"  Release: {platform.release()}",
        f"  Version: {platform.version()}",
        f"  Architecture: {platform.machine()}",
    ]

    hostname = socket.gethostname()
    lines.append(f"  Hostname: {hostname}")
    lines.append("")

    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    lines.extend(
        [
            "CPU:",
            f"  Physical cores: {cpu_physical}",
            f"  Logical cores: {cpu_logical}",
        ]
    )

    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        lines.append(
            f"  Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)"
        )
    lines.append("")

    mem = psutil.virtual_memory()
    lines.extend(
        [
            "Memory:",
            f"  Total: {mem.total / (1024**3):.2f} GB",
            f"  Available: {mem.available / (1024**3):.2f} GB",
            f"  Used: {mem.used / (1024**3):.2f} GB ({mem.percent}%)",
            "",
        ]
    )

    disk = psutil.disk_usage("/")
    lines.extend(
        [
            "Disk (root):",
            f"  Total: {disk.total / (1024**3):.2f} GB",
            f"  Used: {disk.used / (1024**3):.2f} GB ({disk.percent}%)",
            f"  Free: {disk.free / (1024**3):.2f} GB",
            "",
        ]
    )

    lines.append("Network:")
    try:
        ip_address = socket.gethostbyname(hostname)
        lines.append(f"  IP Address: {ip_address}")
    except Exception:
        lines.append("  IP Address: Unable to determine")

    lines.extend(
        [
            "",
            "Environment:",
            f"  Working Directory: {os.getcwd()}",
            f"  User: {os.getenv('USER', 'unknown')}",
        ]
    )

    if os.getenv("VIRTUAL_ENV"):
        lines.append(f"  Virtual Env: {os.getenv('VIRTUAL_ENV')}")

    logger.info("\n".join(lines))
