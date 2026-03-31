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

"""Utility functions for the endpoint client module."""

import subprocess
import sys

# Platform detection
_IS_LINUX = sys.platform.startswith("linux")
_IS_DARWIN = sys.platform == "darwin"


def get_ephemeral_port_range() -> tuple[int, int]:
    """Get the ephemeral port range from system config.

    On Linux, reads from /proc/sys/net/ipv4/ip_local_port_range.
    On macOS, reads from sysctl net.inet.ip.portrange.first/last.

    Returns:
        Tuple of (low, high) port numbers.
    """
    if _IS_LINUX:
        try:
            with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
                low, high = map(int, f.read().split())
                return low, high
        except (OSError, ValueError):
            return 32768, 60999

    if _IS_DARWIN:
        try:
            low = int(
                subprocess.check_output(
                    ["sysctl", "-n", "net.inet.ip.portrange.first"], text=True
                ).strip()
            )
            high = int(
                subprocess.check_output(
                    ["sysctl", "-n", "net.inet.ip.portrange.last"], text=True
                ).strip()
            )
            return low, high
        except (OSError, ValueError, subprocess.CalledProcessError):
            return 49152, 65535

    raise OSError(f"Ephemeral port range detection is not supported on {sys.platform}.")


def get_used_port_count() -> int:
    """Count TCP sockets using ephemeral ports.
    On Linux, reads from /proc/net/tcp and /proc/net/tcp6.

    Returns:
        Number of TCP sockets using ephemeral ports.
    """
    if _IS_DARWIN:
        return 0
    if not _IS_LINUX:
        raise OSError(f"TCP socket counting is not supported on {sys.platform}.")

    low, high = get_ephemeral_port_range()
    count = 0

    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_file) as f:
                next(f)  # Skip header
                for line in f:
                    # local_address is second field: "IP:PORT" in hex
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    local_addr = parts[1]  # e.g., "0100007F:1F90"
                    port_hex = local_addr.split(":")[1]
                    port = int(port_hex, 16)
                    if low <= port <= high:
                        count += 1
        except (OSError, ValueError, IndexError):
            pass

    return count


def get_ephemeral_port_limit() -> int:
    """Get the number of available ephemeral ports.

    Reads the configured port range and subtracts currently used ports
    to determine how many new connections can be established.

    Returns:
        Number of available ephemeral ports.
    """
    low, high = get_ephemeral_port_range()
    total_range = high - low + 1
    used = get_used_port_count()
    available = max(0, total_range - used)
    return available
