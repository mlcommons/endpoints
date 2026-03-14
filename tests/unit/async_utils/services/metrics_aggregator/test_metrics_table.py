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

import msgspec
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.metrics_table import (
    MetricsTable,
    SampleRow,
)


@pytest.mark.unit
class TestSampleRow:
    def test_initial_timestamps_are_none(self):
        row = SampleRow("s1")
        assert row.issued_ns is None
        assert row.complete_ns is None
        assert row.recv_first_ns is None
        assert row.last_recv_ns is None
        assert row.client_send_ns is None
        assert row.client_resp_done_ns is None
        assert row.prompt_text is None
        assert row.first_chunk_text is None
        assert row.output_chunks == []

    def test_is_msgspec_struct(self):
        row = SampleRow("s1")
        assert isinstance(row, msgspec.Struct)

    def test_ttft(self):
        row = SampleRow("s1")
        row.issued_ns = 1000
        row.recv_first_ns = 2500
        assert row.ttft_ns() == 1500

    def test_ttft_returns_none_without_issued(self):
        row = SampleRow("s1")
        row.recv_first_ns = 2500
        assert row.ttft_ns() is None

    def test_ttft_returns_none_without_recv_first(self):
        row = SampleRow("s1")
        row.issued_ns = 1000
        assert row.ttft_ns() is None

    def test_sample_latency(self):
        row = SampleRow("s1")
        row.issued_ns = 1000
        row.complete_ns = 5000
        assert row.sample_latency_ns() == 4000

    def test_sample_latency_returns_none_without_issued(self):
        row = SampleRow("s1")
        row.complete_ns = 5000
        assert row.sample_latency_ns() is None

    def test_sample_latency_returns_none_without_complete(self):
        row = SampleRow("s1")
        row.issued_ns = 1000
        assert row.sample_latency_ns() is None

    def test_request_duration(self):
        row = SampleRow("s1")
        row.client_send_ns = 100
        row.client_resp_done_ns = 600
        assert row.request_duration_ns() == 500

    def test_request_duration_returns_none_without_send(self):
        row = SampleRow("s1")
        row.client_resp_done_ns = 600
        assert row.request_duration_ns() is None

    def test_request_duration_returns_none_without_resp_done(self):
        row = SampleRow("s1")
        row.client_send_ns = 100
        assert row.request_duration_ns() is None

    def test_output_text_empty(self):
        row = SampleRow("s1")
        assert row.output_text() == ""

    def test_output_text_from_chunks(self):
        row = SampleRow("s1")
        row.output_chunks.append("Hello")
        row.output_chunks.append(" World")
        assert row.output_text() == "Hello World"

    def test_first_chunk_text_stored(self):
        row = SampleRow("s1")
        row.first_chunk_text = "First chunk"
        assert row.first_chunk_text == "First chunk"

    def test_fields_are_mutable(self):
        row = SampleRow("s1")
        row.issued_ns = 100
        row.issued_ns = 200
        assert row.issued_ns == 200

    def test_last_recv_ns_tracks_latest(self):
        row = SampleRow("s1")
        row.last_recv_ns = 1000
        row.last_recv_ns = 2000
        row.last_recv_ns = 3000
        assert row.last_recv_ns == 3000


@pytest.mark.unit
class TestMetricsTable:
    def test_create_and_get_row(self):
        table = MetricsTable()
        row = table.create_row("s1")
        assert table.get_row("s1") is row
        assert len(table) == 1

    def test_remove_row(self):
        table = MetricsTable()
        table.create_row("s1")
        removed = table.remove_row("s1")
        assert removed is not None
        assert table.get_row("s1") is None
        assert len(table) == 0

    def test_remove_nonexistent_returns_none(self):
        table = MetricsTable()
        assert table.remove_row("nope") is None

    def test_duplicate_create_returns_existing(self):
        table = MetricsTable()
        row1 = table.create_row("s1")
        row1.issued_ns = 12345
        row2 = table.create_row("s1")
        # Should return same row (retry semantics)
        assert row1 is row2
        assert row2.issued_ns == 12345
        assert len(table) == 1

    def test_multiple_rows(self):
        table = MetricsTable()
        table.create_row("s1")
        table.create_row("s2")
        table.create_row("s3")
        assert len(table) == 3
        table.remove_row("s2")
        assert len(table) == 2
        assert table.get_row("s2") is None
        assert table.get_row("s1") is not None
        assert table.get_row("s3") is not None
