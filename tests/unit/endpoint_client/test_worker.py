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

"""Worker request encoding and error handling, without sockets."""

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from inference_endpoint.core.types import Query
from inference_endpoint.endpoint_client import worker as worker_mod
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http import InFlightRequest
from inference_endpoint.endpoint_client.worker import (
    Worker,
    _is_stream_options_rejection,
)
from inference_endpoint.openai.openai_msgspec_adapter import OpenAIMsgspecAdapter

_REJECTION = (
    b'{"object":"error","message":"Unrecognized request argument supplied: '
    b'stream_options","code":400}'
)


def _worker(**config: Any) -> Worker:
    http_config = HTTPClientConfig(
        endpoint_urls=["http://localhost:8000"],
        num_workers=1,
        warmup_connections=0,
        max_connections=1,
        **config,
    )
    worker = Worker(worker_id=0, connector=MagicMock(), http_config=http_config)
    worker._pool = MagicMock()
    worker._responses = MagicMock()
    worker._http_template = MagicMock()
    return worker


@pytest.mark.unit
@pytest.mark.parametrize("stream_include_usage", [True, False])
def test_prepare_request_passes_the_run_usage_setting(
    monkeypatch, stream_include_usage
):
    encode = MagicMock(return_value=b"{}")
    monkeypatch.setattr(OpenAIMsgspecAdapter, "encode_query", encode)
    worker = _worker(stream_include_usage=stream_include_usage)
    query = Query(id="q", data={"prompt": "hi", "stream": True})

    worker._prepare_request(query)

    encode.assert_called_once_with(query, include_usage=stream_include_usage)


@pytest.mark.unit
@pytest.mark.parametrize(
    "status_code, body, streaming, expected",
    [
        (400, _REJECTION, True, True),
        (422, b"extra field stream_options not permitted", True, True),
        (400, _REJECTION, False, False),
        (400, b'{"message":"max_tokens too large"}', True, False),
        (500, _REJECTION, True, False),
        (399, _REJECTION, True, False),
    ],
)
def test_is_stream_options_rejection(status_code, body, streaming, expected):
    assert _is_stream_options_rejection(status_code, body, streaming) is expected


def _failing_request(query_id: str, status_code: int, body: bytes) -> InFlightRequest:
    conn = MagicMock()
    conn.protocol.read_headers = AsyncMock(return_value=(status_code, []))
    conn.protocol.read_body = AsyncMock(return_value=body)
    req = InFlightRequest(query_id=query_id, http_bytes=b"", is_streaming=True)
    req.connection = conn
    return req


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_options_rejection_logs_opt_out_hint_once(monkeypatch, caplog):
    monkeypatch.setattr(worker_mod, "_stream_options_hint_logged", False)
    caplog.set_level(logging.WARNING, logger=worker_mod.__name__)
    worker = _worker()

    for query_id in ("q0", "q1"):
        await worker._process_response(_failing_request(query_id, 400, _REJECTION))

    hints = [
        r.getMessage()
        for r in caplog.records
        if "--no-stream-include-usage" in r.getMessage()
    ]
    assert len(hints) == 1
    assert "HTTP 400" in hints[0]
    assert "settings.client.stream_include_usage" in hints[0]
    # The hint is advice only: both requests still fail.
    errors = [c.args[0] for c in worker._responses.send.call_args_list]
    assert [e.id for e in errors] == ["q0", "q1"]
    assert all(e.error is not None for e in errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_other_errors_log_no_opt_out_hint(monkeypatch, caplog):
    monkeypatch.setattr(worker_mod, "_stream_options_hint_logged", False)
    caplog.set_level(logging.WARNING, logger=worker_mod.__name__)
    worker = _worker()

    await worker._process_response(_failing_request("q0", 400, b"bad max_tokens"))

    assert not any(
        "--no-stream-include-usage" in r.getMessage() for r in caplog.records
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_opted_out_worker_never_logs_opt_out_hint(monkeypatch, caplog):
    monkeypatch.setattr(worker_mod, "_stream_options_hint_logged", False)
    caplog.set_level(logging.WARNING, logger=worker_mod.__name__)
    worker = _worker(stream_include_usage=False)

    await worker._process_response(_failing_request("q0", 400, _REJECTION))

    assert not any(
        "--no-stream-include-usage" in r.getMessage() for r in caplog.records
    )
    (error,) = (c.args[0] for c in worker._responses.send.call_args_list)
    assert error.id == "q0" and error.error is not None
