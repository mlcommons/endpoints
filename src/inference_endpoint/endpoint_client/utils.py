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

import sys

# Platform detection
_IS_LINUX = sys.platform.startswith("linux")


def get_ephemeral_port_range() -> tuple[int, int]:
    """Get the ephemeral port range from system config.

    On Linux, reads from /proc/sys/net/ipv4/ip_local_port_range.

    Returns:
        Tuple of (low, high) port numbers.

    Raises:
        OSError: If not running on Linux.
    """
    if not _IS_LINUX:
        raise OSError(
            f"Ephemeral port range detection is only supported on Linux, not {sys.platform}."
        )

    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = map(int, f.read().split())
            return low, high
    except (OSError, ValueError):
        # Fallback to typical Linux default
        return 32768, 60999


def get_used_port_count() -> int:
    """Count TCP sockets using ephemeral ports.

    Only counts sockets whose local port is in the ephemeral range.
    Excludes LISTEN sockets (servers) and sockets on well-known ports.
    On Linux, reads from /proc/net/tcp and /proc/net/tcp6.

    /proc/net/tcp format (hex values):
      sl  local_address rem_address   st tx_queue rx_queue ...
       0: 0100007F:1F90 00000000:0000 0A ...
          ^^^^^^^^:^^^^
          IP addr  port (hex)

    Returns:
        Number of TCP sockets using ephemeral ports.

    Raises:
        OSError: If not running on Linux.
    """
    if not _IS_LINUX:
        raise OSError(
            f"TCP socket counting is only supported on Linux, not {sys.platform}."
        )

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
                    # Port is after the colon, in hex
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

    Raises:
        OSError: If not running on Linux.
    """
    low, high = get_ephemeral_port_range()
    total_range = high - low + 1
    used = get_used_port_count()
    available = max(0, total_range - used)
    return available
