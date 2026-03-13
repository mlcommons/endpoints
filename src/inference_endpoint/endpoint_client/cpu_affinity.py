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

Overview In: docs/PERF_ARCHITECTURE.md
TODO:(vir): dump out hw-view in verbose mode
"""

import logging
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Platform check - most APIs in this module require Linux
_CURRENT_PLATFORM = platform.system().lower()
_IS_LINUX = _CURRENT_PLATFORM == "linux"


class UnsupportedPlatformError(RuntimeError):
    """Raised when CPU affinity APIs are used on non-Linux systems."""

    pass


def require_linux(func: Callable) -> Callable:
    """Decorator to wrap functions that should error if not on Linux."""

    def wrapper(*args, **kwargs):
        if not _IS_LINUX:
            raise UnsupportedPlatformError(
                f"{func.__name__}() requires Linux (current platform: {_CURRENT_PLATFORM})"
            )
        return func(*args, **kwargs)

    return wrapper


# NOTE(vir): seeing high jitter when loadgen has <2 Physical CPUs
# Default physical cores for LoadGen (main process):
#   - Session thread (scheduler, busy-wait timing)
#   - Event loop thread (uvloop, response handling)
DEFAULT_LOADGEN_CORES = 5


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AffinityPlan:
    """CPU affinity assignment plan.

    Attributes:
        loadgen_cpus: All logical CPUs for loadgen (from Primary NUMA).
        worker_cpu_sets: List of CPU sets, one per worker. Each set contains
            both hyperthreads of a dedicated physical core, e.g. [[20,21], [22,23], ...].
        _loadgen_physical_cores: Number of physical cores assigned to loadgen.
        _primary_numa: The NUMA node containing loadgen (fastest core's NUMA).
    """

    loadgen_cpus: list[int]
    worker_cpu_sets: list[list[int]]
    _loadgen_physical_cores: int = 0
    _primary_numa: int = 0

    @property
    def num_loadgen_physical_cores(self) -> int:
        """Number of physical cores assigned to loadgen."""
        return self._loadgen_physical_cores

    @property
    def num_worker_physical_cores(self) -> int:
        """Number of physical cores available for workers."""
        return len(self.worker_cpu_sets)

    @property
    def primary_numa(self) -> int:
        """The NUMA node containing loadgen (fastest core's NUMA)."""
        return self._primary_numa

    def get_worker_cpus(self, worker_id: int) -> list[int]:
        """Get the CPUs for a specific worker (round-robins if more workers than cores)."""
        if not self.worker_cpu_sets:
            return []
        return self.worker_cpu_sets[worker_id % len(self.worker_cpu_sets)]

    def summary(self) -> str:
        """Return a human-readable summary of the affinity plan."""
        lines = [
            f"Primary NUMA: {self._primary_numa}",
            f"LoadGen: {self._loadgen_physical_cores} physical cores, "
            f"CPUs {self.loadgen_cpus}",
            f"Workers: {len(self.worker_cpu_sets)} physical cores available",
        ]
        if self.worker_cpu_sets:
            # Show first few worker CPU sets
            preview = self.worker_cpu_sets[:5]
            preview_str = ", ".join(str(cpus) for cpus in preview)
            if len(self.worker_cpu_sets) > 5:
                preview_str += f", ... (+{len(self.worker_cpu_sets) - 5} more)"
            lines.append(f"Worker CPU sets: {preview_str}")
        return "\n".join(lines)


# =============================================================================
# Main API
# =============================================================================


@require_linux
def compute_affinity_plan(
    num_workers: int, loadgen_cores: int = DEFAULT_LOADGEN_CORES
) -> AffinityPlan:
    """Compute NUMA-aware CPU affinity plan with physical core isolation.

    Strategy (see module docstring for decision tree):
    - Phase 1: Discover topology, group cores by NUMA, rank by performance
    - Phase 2: LoadGen gets N cores from Primary NUMA (contains fastest core)
    - Phase 3: Workers get remaining cores, Primary NUMA first, then spill
    - Each process gets all hyperthreads of its physical cores (no sharing)

    Args:
        num_workers: Number of worker processes.
        loadgen_cores: Number of physical cores for loadgen.

    Returns:
        AffinityPlan with CPU assignments.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    all_cpus = get_all_online_cpus()
    if not all_cpus:
        logger.warning("No CPUs available for affinity plan")
        return AffinityPlan(loadgen_cpus=[], worker_cpu_sets=[])

    logger.info(f"CPU affinity: {len(all_cpus)} online CPUs available to process")

    # =========================================================================
    # PHASE 1: TOPOLOGY DISCOVERY
    # =========================================================================

    # Build NUMA -> {phys_core_id -> [logical_cpus]} mapping
    numa_cores: dict[int, dict[int, list[int]]] = {}
    for cpu in all_cpus:
        phys = get_physical_core_id(cpu)
        numa = get_numa_node(cpu)
        if phys is None:
            phys = cpu  # Fallback: treat each CPU as its own physical core
        if numa is None:
            numa = 0
        if numa not in numa_cores:
            numa_cores[numa] = {}
        if phys not in numa_cores[numa]:
            numa_cores[numa][phys] = []
        numa_cores[numa][phys].append(cpu)

    total_phys_cores = sum(len(cores) for cores in numa_cores.values())
    logger.info(
        f"CPU affinity: {total_phys_cores} physical cores across {len(numa_cores)} NUMA nodes, "
        f"requesting {loadgen_cores} for loadgen, {num_workers} workers"
    )

    # Rank all CPUs by performance (fastest first)
    ranked_cpus = get_cpus_ranked_by_performance(all_cpus)
    cpu_rank = {cpu: i for i, cpu in enumerate(ranked_cpus)}

    def core_perf_rank(numa: int, phys: int) -> int:
        """Get performance rank of a physical core (lower = faster)."""
        cpus = numa_cores[numa][phys]
        return min(cpu_rank.get(c, len(ranked_cpus)) for c in cpus)

    # =========================================================================
    # PHASE 2: LOADGEN ASSIGNMENT (single NUMA)
    # =========================================================================

    # Find fastest core globally -> its NUMA becomes Primary NUMA
    fastest_cpu = ranked_cpus[0] if ranked_cpus else min(all_cpus)
    primary_numa = get_numa_node(fastest_cpu)
    if primary_numa is None:
        primary_numa = 0

    logger.debug(f"Primary NUMA: {primary_numa} (contains fastest CPU {fastest_cpu})")

    # Sort cores within Primary NUMA by performance
    primary_cores = sorted(
        numa_cores.get(primary_numa, {}).keys(),
        key=lambda p: core_perf_rank(primary_numa, p),
    )

    # Take loadgen cores from Primary NUMA only
    actual_loadgen_cores = min(loadgen_cores, len(primary_cores))
    loadgen_phys_cores = [
        (primary_numa, p) for p in primary_cores[:actual_loadgen_cores]
    ]
    remaining_primary_cores = primary_cores[actual_loadgen_cores:]

    if actual_loadgen_cores < loadgen_cores:
        logger.warning(
            f"Primary NUMA {primary_numa} has only {len(primary_cores)} cores, "
            f"loadgen requested {loadgen_cores}, using {actual_loadgen_cores}"
        )

    # =========================================================================
    # PHASE 3: WORKER ASSIGNMENT (NUMA-aware spillover)
    # =========================================================================

    # Build worker core list:
    # 1. Remaining cores from Primary NUMA (sorted by perf)
    # 2. Cores from other NUMAs (sorted by NUMA's best core perf, then by core perf)

    worker_phys_cores: list[tuple[int, int]] = []

    # Add remaining Primary NUMA cores first
    for phys in remaining_primary_cores:
        worker_phys_cores.append((primary_numa, phys))

    # Sort other NUMA nodes by their best core's performance
    other_numas: list[int] = sorted(
        (numa for numa in numa_cores if numa != primary_numa),
        key=lambda n: min(core_perf_rank(n, p) for p in numa_cores[n]),
    )

    # Add cores from other NUMAs (each NUMA's cores sorted by perf)
    for numa in other_numas:
        numa_id = numa  # Capture numa in local variable for lambda closure
        sorted_cores = sorted(
            numa_cores[numa].keys(),
            key=lambda p: core_perf_rank(numa_id, p),
        )
        for phys in sorted_cores:
            worker_phys_cores.append((numa, phys))

    if num_workers > len(worker_phys_cores):
        logger.warning(
            f"Not enough physical cores for {num_workers} workers "
            f"(have {len(worker_phys_cores)}, {total_phys_cores} total). Workers will share cores."
        )

    # Log NUMA distribution for workers
    if worker_phys_cores:
        numa_distribution: dict[int, int] = {}
        for numa, _ in worker_phys_cores:
            numa_distribution[numa] = numa_distribution.get(numa, 0) + 1
        logger.debug(f"Worker cores by NUMA: {numa_distribution}")

    # Build CPU sets (all hyperthreads per physical core)
    loadgen_cpus = sorted(
        cpu for numa, phys in loadgen_phys_cores for cpu in numa_cores[numa][phys]
    )
    worker_cpu_sets = [
        sorted(numa_cores[numa][phys]) for numa, phys in worker_phys_cores
    ]

    return AffinityPlan(
        loadgen_cpus=loadgen_cpus,
        worker_cpu_sets=worker_cpu_sets,
        _loadgen_physical_cores=len(loadgen_phys_cores),
        _primary_numa=primary_numa,
    )


@require_linux
def pin_loadgen(
    num_workers: int, loadgen_cores: int = DEFAULT_LOADGEN_CORES
) -> AffinityPlan | None:
    """Compute plan and pin current process (loadgen) to its assigned CPUs.

    Args:
        num_workers: Number of workers to account for in the plan.
        loadgen_cores: Number of physical cores for loadgen.

    Returns:
        The affinity plan used, or None if pinning failed.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    plan = compute_affinity_plan(num_workers, loadgen_cores)

    if not plan.loadgen_cpus:
        logger.warning("No CPUs available for loadgen pinning")
        return None

    try:
        os.sched_setaffinity(os.getpid(), set(plan.loadgen_cpus))  # type: ignore[attr-defined]
        logger.info(
            f"LoadGen pinned to {len(plan.loadgen_cpus)} CPUs "
            f"({plan.num_loadgen_physical_cores} physical cores)"
        )
        return plan
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to pin loadgen: {e}")
        return None


@require_linux
def set_cpu_affinity(pid: int, cpus: set[int]) -> bool:
    """Set CPU affinity for a process.

    Args:
        pid: Process ID.
        cpus: Set of CPU cores to pin to.

    Returns:
        True if successful.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    if not cpus:
        return False

    try:
        os.sched_setaffinity(pid, cpus)  # type: ignore[attr-defined]
        logger.debug(f"Process {pid} pinned to CPUs {sorted(cpus)}")
        return True
    except (OSError, AttributeError) as e:
        logger.warning(f"Failed to set affinity for pid {pid}: {e}")
        return False


# =============================================================================
# Topology Utilities
# =============================================================================

# Common sysfs base path
_SYSFS_CPU = Path("/sys/devices/system/cpu")
_SYSFS_NODE = Path("/sys/devices/system/node")


def _read_sysfs_int(path: Path) -> int | None:
    """Read integer from sysfs file, return None on failure."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_sysfs_cpulist(path: Path) -> set[int] | None:
    """Read and parse cpulist from sysfs file, return None on failure."""
    try:
        return _parse_cpulist(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _parse_cpulist(cpulist_str: str) -> set[int]:
    """Parse a CPU list string (e.g., '0-3,8-11') into a set."""
    if not cpulist_str:
        return set()
    cpus: set[int] = set()
    for part in cpulist_str.split(","):
        if "-" in (p := part.strip()):
            start, end = p.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(p))
    return cpus


@require_linux
def get_current_numa_node() -> int | None:
    """Get the NUMA node of the current process.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    try:
        if current_cpus := os.sched_getaffinity(0):  # type: ignore[attr-defined]
            return get_numa_node(min(current_cpus))
    except OSError:
        pass
    return None


@require_linux
def get_cpus_in_numa_node(node: int) -> set[int]:
    """Get all CPUs in a given NUMA node.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    return _read_sysfs_cpulist(_SYSFS_NODE / f"node{node}" / "cpulist") or set()


@require_linux
def get_numa_node(cpu: int) -> int | None:
    """Get the NUMA node for a given CPU.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """

    # Try direct lookup via cpu's node symlink
    try:
        for entry in (_SYSFS_CPU / f"cpu{cpu}").iterdir():
            if entry.name.startswith("node") and entry.name[4:].isdigit():
                return int(entry.name[4:])
    except OSError:
        pass

    # Fallback: scan NUMA node cpulists
    try:
        for node_dir in _SYSFS_NODE.iterdir():
            if not (node_dir.name.startswith("node") and node_dir.name[4:].isdigit()):
                continue
            if (cpus := _read_sysfs_cpulist(node_dir / "cpulist")) and cpu in cpus:
                return int(node_dir.name[4:])
    except OSError:
        pass

    return None


@require_linux
def get_physical_core_id(cpu: int) -> int | None:
    """Get physical core ID for a logical CPU.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    return _read_sysfs_int(_SYSFS_CPU / f"cpu{cpu}" / "topology" / "core_id")


@require_linux
def get_all_online_cpus() -> set[int]:
    """Get all online CPUs available to the current process.

    Returns intersection of system online CPUs and cgroup-allowed CPUs.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """

    # Get cgroup-allowed CPUs
    try:
        allowed = os.sched_getaffinity(0)  # type: ignore[attr-defined]
    except OSError:
        allowed = None

    def _intersect(cpus: set[int]) -> set[int]:
        """Intersect with allowed CPUs, log if restricted."""
        if allowed is None:
            return cpus
        result = cpus & allowed
        if len(result) < len(cpus):
            logger.debug(
                f"CPU restriction: system={len(cpus)}, allowed={len(allowed)}, using={len(result)}"
            )
        return result

    # Try sysfs online cpulist
    if system_cpus := _read_sysfs_cpulist(_SYSFS_CPU / "online"):
        return _intersect(system_cpus)

    # Fallback: enumerate cpu directories
    try:
        cpus = {
            int(e.name[3:])
            for e in _SYSFS_CPU.iterdir()
            if e.name.startswith("cpu") and e.name[3:].isdigit()
        }
        if cpus:
            return _intersect(cpus)
    except OSError:
        pass

    # Final fallback: use allowed directly
    return allowed or set()


@require_linux
def get_cpus_ranked_by_performance(cpus: set[int] | None = None) -> list[int]:
    """Get CPUs ranked by performance (fastest first).

    Tries ACPI CPPC, ARM cpu_capacity, then max frequency.

    Raises:
        UnsupportedPlatformError: If not running on Linux.
    """
    if cpus is None:
        cpus = get_all_online_cpus()
    if not cpus:
        return []

    # Performance metric paths to try (in priority order)
    metric_paths = [
        "acpi_cppc/highest_perf",  # x86 hybrid (P-core vs E-core)
        "cpu_capacity",  # ARM big.LITTLE
        "cpufreq/cpuinfo_max_freq",  # Fallback: max frequency
    ]

    for metric in metric_paths:
        scores = {
            cpu: score
            for cpu in cpus
            if (score := _read_sysfs_int(_SYSFS_CPU / f"cpu{cpu}" / metric)) is not None
        }
        if scores:
            return sorted(cpus, key=lambda c: scores.get(c, 0), reverse=True)

    return sorted(cpus)
