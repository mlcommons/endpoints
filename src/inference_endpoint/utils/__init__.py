"""
Utility functions for the MLPerf Inference Endpoint Benchmarking System.

This module contains common utilities used throughout the system.
"""

import ctypes
import time
import warnings

try:
    libc = ctypes.CDLL("libc.so.6")
    if not hasattr(libc, "usleep"):
        raise RuntimeError("libc.so.6 does not contain the usleep function")

    def sleep_ns(nanoseconds: int):
        microseconds = int(nanoseconds // 1000)
        libc.usleep(microseconds)
except (OSError, AttributeError, RuntimeError):
    warnings.warn(
        "libc.usleep() unavailable, falling back to time.sleep() with reduced precision. "
        "This may impact timing-sensitive benchmarks.",
        RuntimeWarning,
        stacklevel=2,
    )

    def sleep_ns(nanoseconds: int):
        time.sleep(nanoseconds / 1e9)


def byte_quantity_to_str(
    n_bytes: int,
    max_unit: str = "GB",
) -> str:
    """Convert a byte quantity to a human-readable string.

    Args:
        n_bytes: The byte quantity to convert.
        max_unit: The maximum unit to reduce to. Supports up to Terabytes (TB). (Default: "GB")

    Returns:
        A human-readable string representing the byte quantity.
    """
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    if max_unit not in suffixes:
        raise ValueError(f"Invalid max_unit: {max_unit}. Must be one of {suffixes}")
    suffix_idx = 0
    while n_bytes >= 1024:
        if suffixes[suffix_idx] == max_unit or suffix_idx >= len(suffixes) - 1:
            break
        n_bytes /= 1024
        suffix_idx += 1
    n_bytes = int(n_bytes)
    suffix = suffixes[suffix_idx]
    return f"{n_bytes}{suffix}"
