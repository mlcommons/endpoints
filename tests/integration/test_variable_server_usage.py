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

"""Integration test: the variable-response stub streams usage only when asked."""

import json
import urllib.request

import pytest
from inference_endpoint.testing.variable_throughput_server import (
    VariableResponseServer,
)

from tests.test_helpers import parse_sse_frames


@pytest.mark.integration
@pytest.mark.parametrize("include_usage", [True, False])
def test_variable_server_stream_usage_follows_request(include_usage):
    payload: dict = {"stream": True}
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    with VariableResponseServer(
        host="127.0.0.1",
        port=0,
        output_len_mean=40,
        output_len_spread=0.0,
        stream=True,
        stream_interval=10,
        num_workers=1,
        quiet=True,
    ) as srv:
        req = urllib.request.Request(
            f"{srv.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            frames = parse_sse_frames(resp.read())
    # Content frames, then the finish frame, then usage (only when asked), [DONE].
    tail_len = 3 if include_usage else 2
    n = sum(len(f["choices"][0]["delta"]["content"]) for f in frames[:-tail_len])
    assert n > 0
    assert frames[-tail_len]["choices"][0]["finish_reason"] == "stop"
    if include_usage:
        assert frames[-2]["choices"] == []
        assert frames[-2]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": n,
            "total_tokens": n + 10,
        }
    assert frames[-1] == b"[DONE]"
