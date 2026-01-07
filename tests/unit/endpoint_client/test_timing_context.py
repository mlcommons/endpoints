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

"""Tests for timing_context module."""

import orjson
import pytest
from inference_endpoint.endpoint_client.timing_context import (
    MemoryBufferPrinter,
    RequestTimingContext,
)


class TestRequestTimingContext:
    """Tests for RequestTimingContext dict subclass."""

    def test_compute_pre_overheads(self):
        """Test compute_pre_overheads returns metrics dict."""
        ctx = RequestTimingContext(id="req-1")
        ctx["t_recv"] = 0
        ctx["t_encode"] = 500_000  # 0.5ms
        ctx["t_prepare"] = 1_000_000  # 1ms
        ctx["t_conn_start"] = 1_200_000  # 1.2ms
        ctx["t_conn_end"] = 1_500_000  # 1.5ms
        ctx["t_http"] = 2_000_000  # 2ms

        metrics = ctx.compute_pre_overheads()

        assert metrics["recv_to_bytes"] == 0.5
        assert metrics["bytes_to_http_payload"] == 0.5
        assert metrics["tcp_conn_pool"] == pytest.approx(0.3)
        assert metrics["http_payload_send"] == 0.5
        assert metrics["pre_overhead"] == 2.0

    def test_compute_post_overheads(self):
        """Test compute_post_overheads returns metrics dict."""
        ctx = RequestTimingContext(id="req-1")
        ctx["t_recv"] = 0
        ctx["t_http"] = 1_000_000  # 1ms
        ctx["t_task_awake"] = 1_100_000  # 1.1ms
        ctx["t_headers"] = 2_000_000  # 2ms
        ctx["t_first_chunk"] = 3_000_000  # 3ms
        ctx["t_response"] = 10_000_000  # 10ms
        ctx["t_zmq_sent"] = 10_500_000  # 10.5ms

        metrics = ctx.compute_post_overheads()

        assert metrics["task_overhead"] == pytest.approx(0.1)
        assert metrics["http_to_headers"] == 1.0
        assert metrics["headers_to_first_chunk"] == 1.0
        assert metrics["first_to_last_chunk"] == 7.0
        assert metrics["in_flight_time"] == 9.0
        assert metrics["post_overhead"] == 0.5
        assert metrics["end_to_end"] == 10.5
        assert metrics["t_recv"] == 0.0
        assert metrics["t_zmq_sent"] == 10_500_000.0


class TestMemoryBufferPrinter:
    """Tests for MemoryBufferPrinter."""

    def test_buffer_entries(self, tmp_path):
        """Test entries are buffered in memory."""
        path = tmp_path / "timing.jsonl"
        printer = MemoryBufferPrinter(path, worker_id=0)

        printer("q1", "pre", {"metric1": 1.0})
        printer("q2", "post", {"metric2": 2.0})

        assert len(printer) == 2
        assert not path.exists()  # Not written yet

    def test_flush_writes_to_file(self, tmp_path):
        """Test flush() writes entries to file."""
        path = tmp_path / "timing.jsonl"
        printer = MemoryBufferPrinter(path, worker_id=42)

        printer("query-1", "pre", {"pre_overhead": 1.5})
        printer("query-2", "post", {"end_to_end": 10.0})

        printer.flush()

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        entry1 = orjson.loads(lines[0])
        assert entry1["query_id"] == "query-1"
        assert entry1["worker_id"] == 42
        assert entry1["phase"] == "pre"
        assert entry1["metrics"]["pre_overhead"] == 1.5

        entry2 = orjson.loads(lines[1])
        assert entry2["query_id"] == "query-2"
        assert entry2["phase"] == "post"

        # flush() should clear the buffer as well
        assert len(printer) == 0


class TestMemoryBufferPrinterObjectSize:
    """Tests documenting object CPU memory pressure from using MemoryBufferPrinter."""

    # Empirically measured: orjson serializes to 108 bytes for this exact entry format
    # JSON: {"query_id":"query-00000000","worker_id":0,"phase":"pre","metrics":{"pre_overhead":1.5,"tcp_conn_pool":0.3}}
    ENTRY_BYTES = 108

    @pytest.mark.parametrize("num_entries", [1, 10, 100])
    def test_buffer_size_linear_scaling(self, tmp_path, num_entries):
        """Test buffer size scales linearly: total = n * ENTRY_BYTES."""
        path = tmp_path / "timing.jsonl"
        printer = MemoryBufferPrinter(path, worker_id=0)

        for i in range(num_entries):
            printer(
                f"query-{i:08d}",  # Fixed 16-char query_id
                "pre",
                {"pre_overhead": 1.5, "tcp_conn_pool": 0.3},
            )

        expected_size = num_entries * self.ENTRY_BYTES
        assert printer.buffer_size_bytes == expected_size
