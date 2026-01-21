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

"""CPU affinity management.

Partitions physical cores between LoadGen (main process) and Workers.
Each process gets all hyperthreads (SMT siblings) of its assigned physical
cores to prevent cross-process cache thrashing.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default physical cores for LoadGen (main process):
#   - 1 core: Session thread (scheduler, busy-wait timing)
#   - 1 core: Event loop thread (uvloop, response handling)
#   - 4 cores: ZMQ I/O threads (matches default io_threads in transport/zmq/transport.py)
DEFAULT_LOADGEN_CORES = 6


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AffinityPlan:
    """CPU affinity assignment plan.

    Attributes:
        loadgen_cpus: All logical CPUs for loadgen (faster physical cores).
        worker_cpu_sets: List of CPU sets, one per worker. Each set contains
            both hyperthreads of a dedicated physical core, e.g. [[20,21], [22,23], ...].
        _loadgen_physical_cores: Number of physical cores assigned to loadgen.
    """

    loadgen_cpus: list[int]
    worker_cpu_sets: list[list[int]]
    _loadgen_physical_cores: int = 0

    @property
    def num_loadgen_physical_cores(self) -> int:
        """Number of physical cores assigned to loadgen."""
        return self._loadgen_physical_cores

    @property
    def num_worker_physical_cores(self) -> int:
        """Number of physical cores available for workers."""
        return len(self.worker_cpu_sets)

    def get_worker_cpus(self, worker_id: int) -> list[int]:
        """Get the CPUs for a specific worker (round-robins if more workers than cores)."""
        if not self.worker_cpu_sets:
            return []
        return self.worker_cpu_sets[worker_id % len(self.worker_cpu_sets)]


# =============================================================================
# Main API
# =============================================================================


def compute_affinity_plan(
    num_workers: int, loadgen_cores: int = DEFAULT_LOADGEN_CORES
) -> AffinityPlan:
    """Compute CPU affinity plan with full physical core isolation.

    Strategy:
    - LoadGen: First N fastest physical cores
    - Workers: Remaining physical cores (spills across NUMA if needed)
    - Each gets all hyperthreads of its physical cores (no sharing)

    Args:
        num_workers: Number of worker processes.
        loadgen_cores: Number of physical cores for loadgen.

    Returns:
        AffinityPlan with CPU assignments.
    """
    all_cpus = get_all_online_cpus()
    if not all_cpus:
        logger.warning("No CPUs available for affinity plan")
        return AffinityPlan(loadgen_cpus=[], worker_cpu_sets=[])

    logger.debug(
        f"Building affinity plan: {len(all_cpus)} online CPUs, {num_workers} workers"
    )

    # Build physical core -> logical CPUs mapping (across all NUMA nodes)
    phys_to_logical: dict[
        tuple[int, int], list[int]
    ] = {}  # (numa_node, core_id) -> cpus
    for cpu in all_cpus:
        phys = get_physical_core_id(cpu)
        numa = get_numa_node(cpu)
        if phys is None:
            phys = cpu
        if numa is None:
            numa = 0
        key = (numa, phys)
        if key not in phys_to_logical:
            phys_to_logical[key] = []
        phys_to_logical[key].append(cpu)

    # Rank physical cores by performance (fastest first)
    ranked_cpus = get_cpus_ranked_by_performance(all_cpus)
    cpu_rank = {cpu: i for i, cpu in enumerate(ranked_cpus)}

    sorted_phys_cores = sorted(
        phys_to_logical.keys(),
        key=lambda p: min(
            cpu_rank.get(c, len(ranked_cpus)) for c in phys_to_logical[p]
        ),
    )

    total_phys_cores = len(sorted_phys_cores)
    loadgen_cores = min(loadgen_cores, total_phys_cores)
    loadgen_phys_cores = sorted_phys_cores[:loadgen_cores]
    worker_phys_cores = sorted_phys_cores[loadgen_cores:]

    logger.debug(
        f"Affinity plan: {total_phys_cores} physical cores, "
        f"{loadgen_cores} for loadgen, {len(worker_phys_cores)} for workers"
    )

    if num_workers > len(worker_phys_cores):
        logger.warning(
            f"Not enough physical cores for {num_workers} workers "
            f"(have {len(worker_phys_cores)}). Some workers will share cores."
        )

    # Build CPU sets (all hyperthreads per physical core)
    worker_cpu_sets = [sorted(phys_to_logical[p]) for p in worker_phys_cores]
    loadgen_cpus = sorted(cpu for p in loadgen_phys_cores for cpu in phys_to_logical[p])

    return AffinityPlan(
        loadgen_cpus=loadgen_cpus,
        worker_cpu_sets=worker_cpu_sets,
        _loadgen_physical_cores=len(loadgen_phys_cores),
    )


def pin_loadgen(
    num_workers: int, loadgen_cores: int = DEFAULT_LOADGEN_CORES
) -> AffinityPlan | None:
    """Compute plan and pin current process (loadgen) to its assigned CPUs.

    Args:
        num_workers: Number of workers to account for in the plan.
        loadgen_cores: Number of physical cores for loadgen.

    Returns:
        The affinity plan used, or None if pinning failed.
    """
    plan = compute_affinity_plan(num_workers, loadgen_cores)

    if not plan.loadgen_cpus:
        logger.warning("No CPUs available for loadgen pinning")
        return None

    try:
        os.sched_setaffinity(os.getpid(), set(plan.loadgen_cpus))
        logger.info(
            f"LoadGen pinned to {len(plan.loadgen_cpus)} CPUs "
            f"({plan.num_loadgen_physical_cores} physical cores)"
        )
        return plan
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to pin loadgen: {e}")
        return None


def set_cpu_affinity(pid: int, cpus: set[int]) -> bool:
    """Set CPU affinity for a process.

    Args:
        pid: Process ID.
        cpus: Set of CPU cores to pin to.

    Returns:
        True if successful.
    """
    if not cpus:
        return False

    try:
        os.sched_setaffinity(pid, cpus)
        logger.debug(f"Process {pid} pinned to CPUs {sorted(cpus)}")
        return True
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to set affinity for pid {pid}: {e}")
        return False


# =============================================================================
# Topology Utilities
# =============================================================================


def get_current_numa_node() -> int | None:
    """Get the NUMA node of the current process."""
    try:
        current_cpus = os.sched_getaffinity(0)
    except (OSError, AttributeError):
        return None
    if not current_cpus:
        return None
    return get_numa_node(min(current_cpus))


def get_cpus_in_numa_node(node: int) -> set[int]:
    """Get all CPUs in a given NUMA node."""
    sysfs_path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    try:
        return _parse_cpulist(sysfs_path.read_text().strip())
    except (OSError, ValueError):
        return set()


def get_numa_node(cpu: int) -> int | None:
    """Get the NUMA node for a given CPU."""
    cpu_path = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    try:
        for entry in cpu_path.iterdir():
            name = entry.name
            if name.startswith("node") and name[4:].isdigit():
                return int(name[4:])
    except OSError:
        pass

    # Fallback: scan NUMA node cpulists
    node_base = Path("/sys/devices/system/node")
    try:
        for node_dir in node_base.iterdir():
            name = node_dir.name
            if not (name.startswith("node") and name[4:].isdigit()):
                continue
            cpulist_path = node_dir / "cpulist"
            try:
                if cpu in _parse_cpulist(cpulist_path.read_text().strip()):
                    return int(name[4:])
            except OSError:
                continue
    except OSError:
        pass

    return None


def get_physical_core_id(cpu: int) -> int | None:
    """Get physical core ID for a logical CPU."""
    core_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id")
    try:
        return int(core_path.read_text().strip())
    except (OSError, ValueError):
        return None


def get_all_online_cpus() -> set[int]:
    """Get all online CPUs from the system."""
    online_path = Path("/sys/devices/system/cpu/online")
    try:
        return _parse_cpulist(online_path.read_text().strip())
    except (OSError, ValueError):
        pass

    # Fallback: list cpu directories
    try:
        cpus = set()
        cpu_base = Path("/sys/devices/system/cpu")
        for entry in cpu_base.iterdir():
            name = entry.name
            if name.startswith("cpu") and name[3:].isdigit():
                cpus.add(int(name[3:]))
        if cpus:
            return cpus
    except OSError:
        pass

    return set(range(os.cpu_count() or 1))


def get_cpus_ranked_by_performance(cpus: set[int] | None = None) -> list[int]:
    """Get CPUs ranked by performance (fastest first)."""
    if cpus is None:
        cpus = get_all_online_cpus()
    if not cpus:
        return []

    sysfs_base = Path("/sys/devices/system/cpu")
    cpu_scores: dict[int, int] = {}

    # Try ACPI CPPC highest_perf
    for cpu in cpus:
        perf_path = sysfs_base / f"cpu{cpu}" / "acpi_cppc" / "highest_perf"
        try:
            cpu_scores[cpu] = int(perf_path.read_text().strip())
        except (OSError, ValueError):
            pass

    if cpu_scores:
        return sorted(cpus, key=lambda c: cpu_scores.get(c, 0), reverse=True)

    # Try ARM cpu_capacity
    for cpu in cpus:
        capacity_path = sysfs_base / f"cpu{cpu}" / "cpu_capacity"
        try:
            cpu_scores[cpu] = int(capacity_path.read_text().strip())
        except (OSError, ValueError):
            pass

    if cpu_scores:
        return sorted(cpus, key=lambda c: cpu_scores.get(c, 0), reverse=True)

    # Try max frequency
    for cpu in cpus:
        freq_path = sysfs_base / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq"
        try:
            cpu_scores[cpu] = int(freq_path.read_text().strip())
        except (OSError, ValueError):
            pass

    if cpu_scores:
        return sorted(cpus, key=lambda c: cpu_scores.get(c, 0), reverse=True)

    return sorted(cpus)


def _parse_cpulist(cpulist_str: str) -> set[int]:
    """Parse a CPU list string (e.g., '0-3,8-11') into a set."""
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
