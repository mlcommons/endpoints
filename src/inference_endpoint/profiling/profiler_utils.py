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

"""Shared utilities for all profilers.

Common functionality without requiring inheritance or abstract methods.
"""

import multiprocessing
import os
import sys
import time
from pathlib import Path

from . import ENV_VAR_PROFILER_OUT_DIR

# Global run-specific output directory (created once per process)
_run_output_dir: Path | None = None


def get_run_output_dir() -> Path:
    """Get the run-specific output directory for all profilers.

    Creates a unique directory for this run in /tmp (or user-specified location).
    Directory is shared across main process and all workers.

    Returns:
        Path to the run-specific output directory
    """
    global _run_output_dir

    if _run_output_dir is None:
        # Check if user specified a custom directory or main process already set one
        custom_dir = os.environ.get(ENV_VAR_PROFILER_OUT_DIR)

        if custom_dir:
            # Use directory from environment (set by main process or user)
            _run_output_dir = Path(custom_dir)
        else:
            # Main process creates run-specific directory in /tmp
            timestamp = int(time.time() * 1000)
            _run_output_dir = Path(f"/tmp/mlperf_profiles_run_{timestamp}")

            # Set environment variable so workers inherit the same directory
            os.environ[ENV_VAR_PROFILER_OUT_DIR] = str(_run_output_dir)

        # Create directory
        _run_output_dir.mkdir(parents=True, exist_ok=True)

    return _run_output_dir


def is_pytest_mode() -> bool:
    """Check if we're running under pytest."""
    return "pytest" in sys.modules


def is_worker_process() -> bool:
    """Check if we're in a worker process (not MainProcess)."""
    try:
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:
        return False


def get_output_path(
    output_dir: str | Path, profiler_name: str, suffix: str = "txt"
) -> Path:
    """Get output file path for current process.

    Args:
        output_dir: Directory for output files
        profiler_name: Name of the profiler (for filename)
        suffix: File extension (without dot)

    Returns:
        Path to output file
    """
    import time

    output_dir = Path(output_dir)
    pid = os.getpid()
    timestamp = int(time.time() * 1000)
    filename = f"profile_pid{pid}_{timestamp}.{suffix}"
    return output_dir / filename


def log_profiler_start(profiler_name: str):
    """Log that profiler started."""
    if is_worker_process():
        print(
            f"[{profiler_name.capitalize()}] Started profiling for worker PID {os.getpid()}",
            file=sys.stderr,
        )

    else:
        print(
            f"[{profiler_name.capitalize()}] Started profiling for main process {os.getpid()}",
            file=sys.stderr,
        )


def collect_all_profiles(
    profiler_name: str,
    output_dir: str | Path,
) -> tuple[list[Path], list[Path]]:
    """Collect all profile files for a profiler, separated by main and workers.

    Args:
        profiler_name: Name of profiler (for logging)
        output_dir: Directory containing profiles

    Returns:
        Tuple of (main_process_files, worker_files)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return [], []

    # Find all profile files
    all_files = list(output_dir.glob("profile_pid*.txt"))

    # Separate main process from workers based on current PID
    main_pid = os.getpid()
    main_files = [f for f in all_files if f"_pid{main_pid}_" in f.name]
    worker_files = [f for f in all_files if f"_pid{main_pid}_" not in f.name]

    return sorted(main_files), sorted(worker_files)


def print_all_profiles(
    profiler_name: str,
    output_dir: str | Path,
):
    """Collect and print all profiles (main and workers) for pytest.

    Args:
        profiler_name: Name of profiler (for header)
        output_dir: Directory containing profiles
    """
    main_files, worker_files = collect_all_profiles(profiler_name, output_dir)

    if not main_files and not worker_files:
        return

    print("\n" + "=" * 80, file=sys.stderr)
    print(f"{profiler_name.upper()} PROFILING RESULTS", file=sys.stderr)
    print(f"Main: {len(main_files)}, Workers: {len(worker_files)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)

    # Print main process profiles first
    if main_files:
        print("--- Main Process ---\n", file=sys.stderr)
        for profile_file in main_files:
            try:
                content = profile_file.read_text()
                if content.strip():
                    print(content, file=sys.stderr)
                    print("\n", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {profile_file}: {e}", file=sys.stderr)

    # Then print worker profiles
    if worker_files:
        print("--- Workers ---\n", file=sys.stderr)
        for profile_file in worker_files:
            try:
                content = profile_file.read_text()
                if content.strip():
                    print(content, file=sys.stderr)
                    print("\n", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {profile_file}: {e}", file=sys.stderr)
