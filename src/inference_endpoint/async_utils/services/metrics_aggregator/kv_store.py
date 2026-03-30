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

"""Key-value store for metrics with per-key /dev/shm backing files.

Each key in the store maps to a KVItem backed by an individual mmap'd file.
Two item types are supported:

- **counter**: A single float64 value (e.g., error_count, n_in_flight).
  File layout: [value: 8B float64]

- **series**: An append-only list of float64 values with a length header
  (e.g., ttft_ns, sample_latency_ns). Rollup stats are computed lazily on read.
  File layout: [count: 8B uint64] [v0: 8B float64] [v1: 8B float64] ...

Write protocol (single writer):
    Counter: overwrite the 8-byte value.
    Series: write float64 at HEADER + count*8, then update count.
    On x86-64, aligned 8-byte stores are atomic (TSO), so readers always
    see a consistent state.

Read protocol (any process):
    Counter: read 8 bytes.
    Series: read count, then values[:count]. Rollup computed lazily with
    incremental progress tracking (_last_rollup_idx).
"""

from __future__ import annotations

import logging
import math
import mmap
import os
import shutil
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Series rollup stats (computed on read)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_HEADER_BYTES = 8  # uint64 count for series
_VALUE_BYTES = 8  # float64 per value
_DEFAULT_CAPACITY = 128 * 1024  # pre-allocate for 128k values (~1 MB)


class SeriesStats:
    """Lazily-computed statistics over a series of values.

    Rollup stats (count, total, min, max, sum_sq) are computed on read,
    not on write. ``_last_rollup_idx`` caches progress so subsequent
    reads only process newly appended values.
    """

    __slots__ = (
        "count",
        "total",
        "min_val",
        "max_val",
        "sum_sq",
        "values",
        "_last_rollup_idx",
    )

    def __init__(self, values: list[float] | None = None) -> None:
        self.values: list[float] = values if values is not None else []
        self.count: int = 0
        self.total: float = 0.0
        self.min_val: float = math.inf
        self.max_val: float = -math.inf
        self.sum_sq: float = 0.0
        self._last_rollup_idx: int = 0
        if self.values:
            self._update_rollup()

    def _update_rollup(self) -> None:
        """Incrementally update rollup stats from _last_rollup_idx onward."""
        for v in self.values[self._last_rollup_idx :]:
            self.total += v
            self.sum_sq += v * v
            if v < self.min_val:
                self.min_val = v
            if v > self.max_val:
                self.max_val = v
        self.count = len(self.values)
        self._last_rollup_idx = self.count


# ---------------------------------------------------------------------------
# KVStore ABC
# ---------------------------------------------------------------------------


class KVStore(ABC):
    """Abstract key-value store for metrics.

    Keys are created with a type (counter or series). Values are updated
    via update() and read via get() or snapshot(). Implementations may
    back keys with /dev/shm files, Prometheus, or in-memory dicts.
    """

    @abstractmethod
    def create_key(self, key: str, type: Literal["series", "counter"]) -> None:
        """Register a new key in the store."""
        raise NotImplementedError

    @abstractmethod
    def update(self, key: str, value: float) -> None:
        """Update a key. For counters, sets the value. For series, appends."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str) -> float | SeriesStats:
        """Read the current value of a key."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> dict[str, float | SeriesStats]:
        """Return a dict of all keys and their current values."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# KVItem implementations (per-key mmap files)
# ---------------------------------------------------------------------------


class _CounterItem:
    """Single float64 value backed by an 8-byte mmap file."""

    __slots__ = ("_mm", "_path", "_closed")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._closed = False
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            os.ftruncate(fd, _VALUE_BYTES)
            self._mm = mmap.mmap(fd, _VALUE_BYTES)
        finally:
            os.close(fd)
        struct.pack_into("<d", self._mm, 0, 0.0)

    def set(self, value: float) -> None:
        if not self._closed:
            struct.pack_into("<d", self._mm, 0, value)

    def get(self) -> float:
        return struct.unpack_from("<d", self._mm, 0)[0]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._mm.close()


class _CounterReader:
    """Reader for a counter item."""

    __slots__ = ("_fd", "_mm", "_path")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        if path.exists():
            self._open()

    def _open(self) -> None:
        fd = os.open(str(self._path), os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, _VALUE_BYTES, prot=mmap.PROT_READ)
            self._fd = fd
        except Exception:
            os.close(fd)
            raise

    def get(self) -> float:
        if self._mm is None:
            if self._path.exists():
                self._open()
            if self._mm is None:
                return 0.0
        return struct.unpack_from("<d", self._mm, 0)[0]

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class _SeriesItem:
    """Append-only float64 series backed by an mmap file."""

    __slots__ = ("_mm", "_capacity", "_count", "_path", "_closed")

    def __init__(self, path: Path, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._path = path
        self._capacity = capacity
        self._count = 0
        self._closed = False
        total = _HEADER_BYTES + capacity * _VALUE_BYTES
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        struct.pack_into("<Q", self._mm, 0, 0)

    def append(self, value: float) -> None:
        if self._closed:
            logger.warning("append() called on closed series: %s", self._path)
            return
        if self._count >= self._capacity:
            self._grow()
        offset = _HEADER_BYTES + self._count * _VALUE_BYTES
        struct.pack_into("<d", self._mm, offset, value)
        # Flush to ensure the value is visible before updating the count header.
        # Without this, ARM (weak memory model) readers could see the new count
        # but read stale/uninitialized data at the corresponding offset.
        self._mm.flush()
        self._count += 1
        struct.pack_into("<Q", self._mm, 0, self._count)

    def get(self) -> SeriesStats:
        """Read all values from the mmap and return as SeriesStats."""
        if self._count == 0:
            return SeriesStats()
        raw = self._mm[_HEADER_BYTES : _HEADER_BYTES + self._count * _VALUE_BYTES]
        values = list(struct.unpack(f"<{self._count}d", raw))
        return SeriesStats(values)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._mm.close()

    def _grow(self) -> None:
        old_mm = self._mm
        new_capacity = self._capacity * 2
        total = _HEADER_BYTES + new_capacity * _VALUE_BYTES
        fd = os.open(str(self._path), os.O_RDWR)
        try:
            os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total)
            self._capacity = new_capacity
        except Exception:
            self._mm = old_mm
            raise
        finally:
            os.close(fd)
        old_mm.close()


class _SeriesReader:
    """Reader for a series item with lazy rollup."""

    __slots__ = ("_fd", "_mm", "_path", "_stats")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._stats = SeriesStats()
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        if path.exists():
            self._open()

    def _open(self) -> None:
        fd = os.open(str(self._path), os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size > 0:
                self._mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
                self._fd = fd
            else:
                os.close(fd)
        except Exception:
            os.close(fd)
            raise

    def get(self) -> SeriesStats:
        if self._mm is None:
            if self._path.exists():
                self._open()
            if self._mm is None:
                return self._stats

        # Re-map if file grew
        file_size = os.fstat(self._fd).st_size  # type: ignore[arg-type]
        if file_size > self._mm.size():
            self._mm.close()
            self._mm = mmap.mmap(self._fd, 0, prot=mmap.PROT_READ)  # type: ignore[arg-type]

        count = struct.unpack_from("<Q", self._mm, 0)[0]
        if count == 0:
            return self._stats

        old_count = len(self._stats.values)
        if count > old_count:
            start_offset = _HEADER_BYTES + old_count * _VALUE_BYTES
            n_new = count - old_count
            raw = self._mm[start_offset : start_offset + n_new * _VALUE_BYTES]
            new_vals = list(struct.unpack(f"<{n_new}d", raw))
            self._stats.values.extend(new_vals)
            self._stats._update_rollup()

        return self._stats

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ---------------------------------------------------------------------------
# BasicKVStore (mmap-backed)
# ---------------------------------------------------------------------------


class BasicKVStore(KVStore):
    """KVStore backed by per-key mmap files on /dev/shm (or any directory).

    Each key gets its own file: counters are 8 bytes, series are append-only
    with a count header. Suitable for single-writer, multi-reader access.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, _CounterItem | _SeriesItem] = {}

    def create_key(self, key: str, type: Literal["series", "counter"]) -> None:
        if key in self._items:
            return
        path = self._dir / f"{key}.kv"
        if type == "counter":
            self._items[key] = _CounterItem(path)
        elif type == "series":
            self._items[key] = _SeriesItem(path)
        else:
            raise ValueError(f"Unknown key type: {type}")

    def update(self, key: str, value: float) -> None:
        item = self._items.get(key)
        if item is None:
            raise KeyError(f"Key not created: {key}")
        if isinstance(item, _CounterItem):
            item.set(value)
        else:
            item.append(value)

    def get(self, key: str) -> float | SeriesStats:
        # Writer-side get (direct access to items)
        item = self._items.get(key)
        if item is None:
            raise KeyError(f"Key not created: {key}")
        return item.get()

    def snapshot(self) -> dict[str, float | SeriesStats]:
        return {key: item.get() for key, item in self._items.items()}

    def close(self) -> None:
        for item in self._items.values():
            item.close()

    def unlink(self) -> None:
        """Close all items and remove the store directory."""
        self.close()
        shutil.rmtree(self._dir, ignore_errors=True)


class BasicKVStoreReader:
    """Read-only view of a BasicKVStore from another process.

    Lazily opens files and reads values. Each call to get() or snapshot()
    picks up new values appended by the writer.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._readers: dict[str, _CounterReader | _SeriesReader] = {}

    def register_key(self, key: str, type: Literal["series", "counter"]) -> None:
        """Register a key to read. Call before get()/snapshot()."""
        if key in self._readers:
            return
        path = self._dir / f"{key}.kv"
        if type == "counter":
            self._readers[key] = _CounterReader(path)
        elif type == "series":
            self._readers[key] = _SeriesReader(path)

    def get(self, key: str) -> float | SeriesStats:
        reader = self._readers.get(key)
        if reader is None:
            raise KeyError(f"Key not registered: {key}")
        return reader.get()

    def snapshot(self) -> dict[str, float | SeriesStats]:
        return {key: reader.get() for key, reader in self._readers.items()}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
