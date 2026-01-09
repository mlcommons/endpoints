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

"""CPU affinity utilities for process pinning."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Capture available CPUs at module load time, before any pinning restricts the process
try:
    AVAILABLE_CPUS: set[int] = set(os.sched_getaffinity(0))
except (OSError, AttributeError):
    AVAILABLE_CPUS = set(range(os.cpu_count() or 1))

# Track the loadgen CPU for NUMA-aware worker pinning
LOADGEN_CPU: int | None = None


def get_fastest_cpu() -> int | None:
    """Return the CPU core with the highest max frequency.

    Reads max frequency from sysfs on Linux. Falls back to min available CPU
    if detection fails (e.g., in containers, non-Linux, or missing cpufreq).

    Returns:
        CPU core number with highest max frequency, or None if no CPUs found.
    """
    if not AVAILABLE_CPUS:
        return None

    sysfs_base = Path("/sys/devices/system/cpu")
    best_cpu = min(AVAILABLE_CPUS)
    best_freq = 0

    for cpu in AVAILABLE_CPUS:
        freq_path = sysfs_base / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq"
        try:
            freq = int(freq_path.read_text().strip())
            if freq > best_freq:
                best_freq = freq
                best_cpu = cpu
        except (OSError, ValueError):
            continue

    if best_freq > 0:
        logger.debug(f"Fastest CPU detected: cpu{best_cpu} ({best_freq} kHz)")
    else:
        logger.debug(f"CPU frequency detection unavailable, using cpu{best_cpu}")

    return best_cpu


def get_numa_node(cpu: int) -> int | None:
    """Get the NUMA node for a given CPU.

    Reads NUMA topology from sysfs on Linux. Uses multiple detection methods
    for robustness across different kernel versions and configurations.

    Detection methods (in order of preference):
    1. Look for nodeX symlink in /sys/devices/system/cpu/cpuN/
    2. Scan /sys/devices/system/node/nodeX/cpulist for matching CPU

    Args:
        cpu: CPU core number.

    Returns:
        NUMA node number, or None if detection fails (e.g., non-Linux,
        containers without sysfs access, or single-node systems without
        NUMA topology exposed).
    """
    # Method 1: Look for nodeX symlink in CPU's sysfs directory
    # Standard location: /sys/devices/system/cpu/cpuN/nodeX -> ../../node/nodeX
    cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    try:
        for entry in cpu_path.iterdir():
            name = entry.name
            # nodeX entries are symlinks to the NUMA node directory
            if name.startswith("node") and name[4:].isdigit():
                return int(name[4:])
    except OSError:
        pass

    # Method 2: Scan all NUMA nodes and check their cpulists
    # Location: /sys/devices/system/node/nodeX/cpulist
    node_base = Path("/sys/devices/system/node")
    try:
        for node_dir in node_base.iterdir():
            name = node_dir.name
            if not (name.startswith("node") and name[4:].isdigit()):
                continue

            node_id = int(name[4:])
            cpulist_path = node_dir / "cpulist"
            try:
                cpulist_str = cpulist_path.read_text().strip()
                if cpu in _parse_cpulist(cpulist_str):
                    return node_id
            except OSError:
                continue
    except OSError:
        pass

    return None


def get_cpus_in_numa_node(node: int) -> set[int]:
    """Get all CPUs in a given NUMA node.

    Reads NUMA topology from sysfs on Linux.

    Args:
        node: NUMA node number.

    Returns:
        Set of CPU core numbers in the NUMA node, or empty set if detection fails.
    """
    sysfs_path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    try:
        cpulist_str = sysfs_path.read_text().strip()
        return _parse_cpulist(cpulist_str)
    except (OSError, ValueError):
        return set()


def _parse_cpulist(cpulist_str: str) -> set[int]:
    """Parse a CPU list string (e.g., "0-3,8-11") into a set of CPU numbers.

    Args:
        cpulist_str: CPU list in kernel format (e.g., "0-3,8-11,16").

    Returns:
        Set of CPU core numbers.
    """
    cpus = set()
    if not cpulist_str:
        return cpus

    for part in cpulist_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(part))

    return cpus


def get_cpus_sorted_by_numa_preference(
    prefer_numa_node: int | None = None,
) -> list[int]:
    """Get available CPUs sorted by NUMA preference.

    CPUs in the preferred NUMA node come first, followed by CPUs in other nodes.

    Args:
        prefer_numa_node: NUMA node to prefer. If None, uses loadgen's NUMA node.

    Returns:
        List of available CPU core numbers, sorted by NUMA preference.
    """
    if prefer_numa_node is None and LOADGEN_CPU is not None:
        prefer_numa_node = get_numa_node(LOADGEN_CPU)

    if prefer_numa_node is None:
        # No NUMA preference, return sorted available CPUs
        return sorted(AVAILABLE_CPUS)

    preferred_cpus = get_cpus_in_numa_node(prefer_numa_node) & AVAILABLE_CPUS
    other_cpus = AVAILABLE_CPUS - preferred_cpus

    # Preferred NUMA node CPUs first, then others
    return sorted(preferred_cpus) + sorted(other_cpus)


def set_loadgen_cpu(cpu: int) -> bool:
    """Pin loadgen to CPU and remove it from available pool.

    After calling this, AVAILABLE_CPUS will no longer contain the loadgen CPU,
    so worker auto-pinning will automatically exclude it. LOADGEN_CPU is set
    for NUMA-aware worker pinning.

    Args:
        cpu: CPU core to pin loadgen process to.

    Returns:
        True if affinity was successfully set, False otherwise.
    """
    global LOADGEN_CPU
    result = set_cpu_affinity(os.getpid(), {cpu})
    if result:
        AVAILABLE_CPUS.discard(cpu)
        LOADGEN_CPU = cpu
    return result


def set_cpu_affinity(pid: int, cpus: set[int]) -> bool:
    """Set CPU affinity for a process.

    Args:
        pid: Process ID to set affinity for.
        cpus: Set of CPU cores to pin the process to.

    Returns:
        True if affinity was successfully set, False otherwise.
    """
    if not cpus:
        return False

    try:
        # Validate against launch-time CPUs, not current process affinity
        valid_cpus = cpus & AVAILABLE_CPUS
        if not valid_cpus:
            logger.warning(f"CPUs {cpus} not available, skipping CPU pinning")
            return False

        os.sched_setaffinity(pid, valid_cpus)
        logger.debug(f"Process {pid} pinned to CPU {valid_cpus}")
        return True

    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to set CPU affinity for pid {pid}: {e}")

    return False
