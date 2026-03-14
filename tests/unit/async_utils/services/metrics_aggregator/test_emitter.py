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

import json

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.emitter import (
    JsonlMetricEmitter,
)


@pytest.mark.unit
class TestJsonlMetricEmitter:
    def test_emit_writes_jsonl_line(self, tmp_path):
        emitter = JsonlMetricEmitter(tmp_path / "metrics", flush_interval=1)
        emitter.emit("sample1", "ttft_ns", 1500)
        emitter.close()

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["sample_uuid"] == "sample1"
        assert record["metric_name"] == "ttft_ns"
        assert record["value"] == 1500
        assert "timestamp_ns" in record

    def test_emit_multiple_metrics(self, tmp_path):
        emitter = JsonlMetricEmitter(tmp_path / "metrics", flush_interval=10)
        emitter.emit("s1", "ttft_ns", 100)
        emitter.emit("s1", "sample_latency_ns", 500)
        emitter.emit("s2", "ttft_ns", 200)
        emitter.close()

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_flush_interval(self, tmp_path):
        emitter = JsonlMetricEmitter(tmp_path / "metrics", flush_interval=2)
        emitter.emit("s1", "m1", 1)
        # After 1 emit, file may not be flushed yet (OS buffering)
        emitter.emit("s1", "m2", 2)
        # After 2 emits, flush_interval triggers flush
        emitter.flush()  # explicit flush to verify no error
        emitter.close()

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_close_is_idempotent(self, tmp_path):
        emitter = JsonlMetricEmitter(tmp_path / "metrics")
        emitter.close()
        emitter.close()  # Should not raise

    def test_float_value(self, tmp_path):
        emitter = JsonlMetricEmitter(tmp_path / "metrics", flush_interval=1)
        emitter.emit("s1", "tpot_ns", 1234.5)
        emitter.close()

        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["value"] == 1234.5
