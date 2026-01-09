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


def set_loadgen_cpu(cpu: int) -> bool:
    """Pin loadgen to CPU and remove it from available pool.

    After calling this, AVAILABLE_CPUS will no longer contain the loadgen CPU,
    so worker auto-pinning will automatically exclude it.

    Args:
        cpu: CPU core to pin loadgen process to.

    Returns:
        True if affinity was successfully set, False otherwise.
    """
    result = set_cpu_affinity(os.getpid(), {cpu})
    if result:
        AVAILABLE_CPUS.discard(cpu)
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
