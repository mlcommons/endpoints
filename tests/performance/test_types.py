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
Performance benchmarks for core types encode/decode using pytest-benchmark.

Measures ns/op for msgspec serialization of Query, QueryResult, StreamChunk
with varying payload sizes (0, 100, 1k, 8k, 32k). Run with:

    pytest tests/performance/test_types.py --benchmark-only --benchmark-columns=mean,stddev,ops

To save results for comparison:
    pytest tests/performance/test_types.py --benchmark-only --benchmark-save=baseline

To compare against saved results:
    pytest tests/performance/test_types.py --benchmark-only --benchmark-compare=baseline
"""

import msgspec
import pytest
from inference_endpoint.core.types import (
    Query,
    QueryResult,
    StreamChunk,
    TextModelOutput,
)

TEXT_SIZES = {
    "empty": "",
    "100": "x" * 100,
    "1k": "x" * 1_000,
    "8k": "x" * 8_000,
    "32k": "x" * 32_000,
}

ENCODER = msgspec.msgpack.Encoder()
DECODERS = {
    "Query": msgspec.msgpack.Decoder(Query),
    "QueryResult": msgspec.msgpack.Decoder(QueryResult),
    "StreamChunk": msgspec.msgpack.Decoder(StreamChunk),
}


def make_instance(type_name: str, text: str):
    """Create a test instance of the given type with the given text payload."""
    if type_name == "Query":
        return Query(
            id="test-id",
            data={"prompt": text, "model": "test-model", "max_tokens": 100},
            headers={"Authorization": "Bearer token"},
        )
    elif type_name == "QueryResult":
        return QueryResult(
            id="test-id",
            response_output=TextModelOutput(output=text),
            metadata={"tokens": 0},
        )
    else:
        return StreamChunk(id="test-id", response_chunk=text, metadata={})


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
@pytest.mark.parametrize("type_name", ["Query", "QueryResult", "StreamChunk"])
def test_encode(benchmark, type_name, size_name, text):
    """Benchmark encoding for each type and payload size."""
    instance = make_instance(type_name, text)
    benchmark.group = f"{type_name.lower()}_encode"
    benchmark(ENCODER.encode, instance)


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("size_name,text", TEXT_SIZES.items(), ids=TEXT_SIZES.keys())
@pytest.mark.parametrize("type_name", ["Query", "QueryResult", "StreamChunk"])
def test_decode(benchmark, type_name, size_name, text):
    """Benchmark decoding for each type and payload size."""
    instance = make_instance(type_name, text)
    encoded = ENCODER.encode(instance)
    benchmark.group = f"{type_name.lower()}_decode"
    benchmark(DECODERS[type_name].decode, encoded)
