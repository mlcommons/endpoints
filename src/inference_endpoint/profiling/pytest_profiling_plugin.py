"""
Pytest plugin for HTTP client profiling.

Automatically configures profiling for pytest test runs when ENABLE_LINE_PROFILER=1:
- Enables profiling at test session start
- Collects and displays main process profile stats
- Aggregates and displays worker process profiles from files
- Ensures clean output even on test failures
"""

import atexit
import glob
import os
import shutil
import sys

from inference_endpoint.profiling import (
    ENV_VAR_ENABLE_LINE_PROFILER,
    ENV_VAR_LINE_PROFILER_LOGFILE,
    profiler_shutdown,
)


def pytest_configure(config):
    """Initialize profiling at pytest session start."""
    if os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER) != "1":
        return

    print(f"\n[Profiling] {ENV_VAR_ENABLE_LINE_PROFILER}=1 detected, profiling enabled")

    # Set default output location for worker profiles
    # Workers write to files to avoid stderr interleaving
    if not os.environ.get(ENV_VAR_LINE_PROFILER_LOGFILE):
        os.environ[ENV_VAR_LINE_PROFILER_LOGFILE] = (
            "/tmp/mlperf_client_profiles/profile"
        )

    # Suppress stderr during interpreter shutdown to hide line_profiler internal errors
    atexit.register(_suppress_stderr_during_shutdown)


def pytest_sessionfinish(session, exitstatus):
    """Print profiling results after test session completes."""
    if os.environ.get(ENV_VAR_ENABLE_LINE_PROFILER) != "1":
        return

    # Shutdown profiler completely (disables atexit handler and prints stats)
    profiler_shutdown()

    # Collect and display worker profiles
    _print_worker_profiles()


def _print_worker_profiles():
    """Read and display worker profile files."""
    output_file = os.environ.get(
        ENV_VAR_LINE_PROFILER_LOGFILE, "/tmp/mlperf_client_profiles/profile"
    )

    # Find all profile files (exclude .lprof binary files)
    all_files = glob.glob(f"{output_file}.*")
    profile_files = [f for f in all_files if not f.endswith(".lprof")]

    # Exclude main process (already printed)
    main_pid = os.getpid()
    worker_files = sorted([f for f in profile_files if not f.endswith(f".{main_pid}")])

    if not worker_files:
        return

    print("\n" + "=" * 80, file=sys.stderr)
    print("WORKER PROFILER RESULTS", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)

    for worker_file in worker_files:
        try:
            with open(worker_file) as f:
                content = f.read()
                if content.strip():  # Only print non-empty files
                    print(content, file=sys.stderr)
        except Exception as e:
            print(f"Error reading {worker_file}: {e}", file=sys.stderr)

    # Cleanup profile directory
    _cleanup_profile_files(output_file)


def _cleanup_profile_files(output_file: str):
    """Remove profile directory and files after displaying results."""
    try:
        profile_dir = os.path.dirname(output_file)
        if profile_dir and os.path.exists(profile_dir):
            shutil.rmtree(profile_dir, ignore_errors=True)
    except Exception:
        pass  # Silently fail cleanup


def _suppress_stderr_during_shutdown():
    """Suppress stderr at OS level to hide harmless line_profiler shutdown errors."""
    try:
        # Redirect stderr file descriptor to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        pass  # Silently fail if stderr redirection fails
