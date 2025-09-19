"""
Utility functions for the MLPerf Inference Endpoint Benchmarking System.

This module contains common utilities used throughout the system.
"""


import ctypes
import time
import warnings


try:
    libc = ctypes.CDLL('libc.so.6')
    if not hasattr(libc, 'usleep'):
        raise RuntimeError("libc.so.6 does not contain the usleep function")

    def sleep_ns(nanoseconds: int):
        microseconds = int(nanoseconds // 1000)
        libc.usleep(microseconds)
except (OSError, AttributeError, RuntimeError):
    warnings.warn(
        "libc.usleep() unavailable, falling back to time.sleep() with reduced precision. "
        "This may impact timing-sensitive benchmarks.",
        RuntimeWarning,
        stacklevel=2
    )

    def sleep_ns(nanoseconds: int):
        time.sleep(nanoseconds / 1e9)
