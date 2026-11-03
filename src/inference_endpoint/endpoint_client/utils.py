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
