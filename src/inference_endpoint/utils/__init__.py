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

"""
Utility functions for the MLPerf Inference Endpoint Benchmarking System.

This module contains common utilities used throughout the system.
"""

from __future__ import annotations

import ctypes
import threading
import time
import warnings
from datetime import datetime
from typing import Any, ClassVar

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
        n_bytes //= 1024
        suffix_idx += 1
    n_bytes = int(n_bytes)
    suffix = suffixes[suffix_idx]
    return f"{n_bytes}{suffix}"


_G_MONOTIME_DELTA = time.time_ns() - time.monotonic_ns()
"""Approximate delta between monotonic and wall-clock time in nanoseconds. See
monotime_to_datetime() for more details.
"""


def monotime_to_datetime(monotime_ns: int) -> datetime:
    """Monotonic clock has an undefined starting point. To convert to human readable timestamp,
    we can add a constant delta to any monotonic timestamp to get an approximate equivalent wall-clock
    timestamp. Note that the result will not be completely accurate, but it will be a consistent
    offset from the real time, as long as this function is called in the same process. Any durations
    and deltas calculated from resulting datetimes will be accurate, but absolute times will not be.

    Args:
        monotime_ns: The monotonic timestamp in nanoseconds.

    Returns:
        The datetime object corresponding to the approximate wall-clock timestamp.
    """
    wall_time = (monotime_ns + _G_MONOTIME_DELTA) / 1e9
    return datetime.fromtimestamp(wall_time)


class WithUpdatesMixin:
    """Mixin for Pydantic models that need ``with_updates(**overrides)``.

    Reconstructs with overrides, re-running all validators.
    """

    def with_updates(self, **updates: object) -> Any:
        """Reconstruct with updates, re-running all validators."""
        return type(self).model_validate(self.model_dump() | updates)  # type: ignore[attr-defined]


class SingletonMixin:
    """Mixin that makes a class a singleton.

    The first call to the constructor creates the instance; subsequent calls
    return the same instance. Subclasses must guard their __init__ body with::

        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        # ... rest of init
    """

    _instance: ClassVar[Any] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    # Inject _initialized attribute if child class does not define it.
                    obj._initialized = False  # type: ignore[attr-defined]
                    cls._instance = obj
        return cls._instance
