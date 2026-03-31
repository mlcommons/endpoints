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

"""Tests for the KVStore (BasicKVStore + BasicKVStoreReader)."""

import multiprocessing
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.kv_store import (
    BasicKVStore,
    BasicKVStoreReader,
    SeriesStats,
)

# ---------------------------------------------------------------------------
# SeriesStats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeriesStats:
    def test_from_values(self):
        stats = SeriesStats([10.0, 20.0, 5.0])
        assert stats.count == 3
        assert stats.total == 35.0
        assert stats.min_val == 5.0
        assert stats.max_val == 20.0

    def test_sum_sq(self):
        stats = SeriesStats([3.0, 4.0])
        assert stats.sum_sq == pytest.approx(3.0**2 + 4.0**2)

    def test_empty(self):
        stats = SeriesStats()
        assert stats.count == 0
        assert stats.total == 0.0

    def test_incremental_rollup(self):
        stats = SeriesStats([1.0, 2.0])
        assert stats._last_rollup_idx == 2
        stats.values.extend([3.0, 4.0])
        stats._update_rollup()
        assert stats.count == 4
        assert stats.total == 10.0
        assert stats._last_rollup_idx == 4


# ---------------------------------------------------------------------------
# BasicKVStore (writer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicKVStore:
    def test_counter(self, tmp_path: Path):
        store = BasicKVStore(tmp_path / "kv")
        store.create_key("error_count", "counter")
        store.update("error_count", 5.0)
        assert store.get("error_count") == 5.0
        store.update("error_count", 10.0)
        assert store.get("error_count") == 10.0
        store.close()

    def test_series(self, tmp_path: Path):
        store = BasicKVStore(tmp_path / "kv")
        store.create_key("ttft_ns", "series")
        store.update("ttft_ns", 100.0)
        store.update("ttft_ns", 200.0)
        result = store.get("ttft_ns")
        assert isinstance(result, SeriesStats)
        assert result.count == 2
        store.close()

    def test_snapshot(self, tmp_path: Path):
        store = BasicKVStore(tmp_path / "kv")
        store.create_key("n_issued", "counter")
        store.create_key("latency", "series")
        store.update("n_issued", 42.0)
        store.update("latency", 1.5)
        store.update("latency", 2.5)

        snap = store.snapshot()
        assert snap["n_issued"] == 42.0
        assert isinstance(snap["latency"], SeriesStats)
        assert snap["latency"].count == 2
        store.close()

    def test_update_unknown_key_raises(self, tmp_path: Path):
        store = BasicKVStore(tmp_path / "kv")
        with pytest.raises(KeyError, match="Key not created"):
            store.update("missing", 1.0)
        store.close()

    def test_create_key_idempotent(self, tmp_path: Path):
        store = BasicKVStore(tmp_path / "kv")
        store.create_key("x", "counter")
        store.update("x", 5.0)
        store.create_key("x", "counter")  # should not reset
        assert store.get("x") == 5.0
        store.close()

    def test_unlink(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        store = BasicKVStore(store_dir)
        store.create_key("a", "counter")
        assert store_dir.exists()
        store.unlink()
        assert not store_dir.exists()


# ---------------------------------------------------------------------------
# BasicKVStoreReader
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicKVStoreReader:
    def test_read_counter(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        writer = BasicKVStore(store_dir)
        writer.create_key("count", "counter")
        writer.update("count", 7.0)

        reader = BasicKVStoreReader(store_dir)
        reader.register_key("count", "counter")
        assert reader.get("count") == 7.0

        reader.close()
        writer.close()

    def test_read_series(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        writer = BasicKVStore(store_dir)
        writer.create_key("ttft", "series")
        writer.update("ttft", 100.0)
        writer.update("ttft", 200.0)

        reader = BasicKVStoreReader(store_dir)
        reader.register_key("ttft", "series")
        stats = reader.get("ttft")
        assert isinstance(stats, SeriesStats)
        assert stats.count == 2
        assert stats.values == [100.0, 200.0]

        reader.close()
        writer.close()

    def test_incremental_read(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        writer = BasicKVStore(store_dir)
        writer.create_key("lat", "series")
        writer.update("lat", 1.0)

        reader = BasicKVStoreReader(store_dir)
        reader.register_key("lat", "series")
        s1 = reader.get("lat")
        assert isinstance(s1, SeriesStats)
        assert s1.count == 1

        writer.update("lat", 2.0)
        writer.update("lat", 3.0)
        s2 = reader.get("lat")
        assert isinstance(s2, SeriesStats)
        assert s2.count == 3
        assert s2.total == 6.0

        reader.close()
        writer.close()

    def test_snapshot(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        writer = BasicKVStore(store_dir)
        writer.create_key("n", "counter")
        writer.create_key("s", "series")
        writer.update("n", 5.0)
        writer.update("s", 10.0)

        reader = BasicKVStoreReader(store_dir)
        reader.register_key("n", "counter")
        reader.register_key("s", "series")
        snap = reader.snapshot()
        assert snap["n"] == 5.0
        assert isinstance(snap["s"], SeriesStats)
        assert snap["s"].count == 1

        reader.close()
        writer.close()

    def test_reader_lazy_open(self, tmp_path: Path):
        """Reader for a key whose file doesn't exist yet opens lazily."""
        store_dir = tmp_path / "kv"
        store_dir.mkdir()
        reader = BasicKVStoreReader(store_dir)
        reader.register_key("lat", "series")
        s = reader.get("lat")
        assert isinstance(s, SeriesStats)
        assert s.count == 0

        # Now create the writer and write
        writer = BasicKVStore(store_dir)
        writer.create_key("lat", "series")
        writer.update("lat", 42.0)

        s = reader.get("lat")
        assert isinstance(s, SeriesStats)
        assert s.count == 1
        assert s.values == [42.0]

        reader.close()
        writer.close()


# ---------------------------------------------------------------------------
# Cross-process
# ---------------------------------------------------------------------------


def _child_read(store_dir_str: str, queue: multiprocessing.Queue) -> None:
    store_dir = Path(store_dir_str)
    reader = BasicKVStoreReader(store_dir)
    reader.register_key("n", "counter")
    reader.register_key("ttft", "series")
    snap = reader.snapshot()
    ttft = snap["ttft"]
    assert isinstance(ttft, SeriesStats)
    queue.put((snap["n"], ttft.count, ttft.values))
    reader.close()


@pytest.mark.unit
class TestCrossProcess:
    def test_cross_process_read(self, tmp_path: Path):
        store_dir = tmp_path / "kv"
        writer = BasicKVStore(store_dir)
        writer.create_key("n", "counter")
        writer.create_key("ttft", "series")
        writer.update("n", 2.0)
        writer.update("ttft", 42.0)
        writer.update("ttft", 99.0)

        q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_child_read, args=(str(store_dir), q))
        proc.start()
        proc.join(timeout=10)

        assert not q.empty()
        n, count, values = q.get()
        assert n == 2.0
        assert count == 2
        assert values == [42.0, 99.0]

        writer.close()
