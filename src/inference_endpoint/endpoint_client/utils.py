# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions for endpoint client."""


def get_ephemeral_port_limit() -> int:
    """Get the number of available ephemeral ports from ip_local_port_range.

    Reads /proc/sys/net/ipv4/ip_local_port_range to determine the range of
    ports available for outbound TCP connections.

    Returns:
        Number of available ephemeral ports.
    """
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = map(int, f.read().split())
            return high - low + 1
    except (OSError, ValueError):
        # Fallback to typical Linux default (32768-60999 = 28232 ports)
        return 28232
