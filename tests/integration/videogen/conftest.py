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

"""Integration test fixtures for the videogen adapter.

The two mocks subclass `EchoServer` to reuse its background-thread
aiohttp lifecycle (port discovery, start/stop, ready event). Only the
route registration differs — videogen serves `/v1/videos/generations`
instead of OpenAI's `/v1/chat/completions`.
"""

import hashlib
from collections.abc import Generator

import pytest
from aiohttp import web
from inference_endpoint.testing.echo_server import EchoServer

# Minimal mp4 bytes the mock returns for response_format="file".
DUMMY_VIDEO_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 24


@pytest.fixture(scope="module")
def mock_video_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Stable per-module path string the mock server returns for video_path mode.

    The mock never writes a real file — the path only needs to be a unique
    string the adapter and tests can assert against. Using tmp_path_factory
    avoids hardcoding shared-storage locations like /lustre/....
    """
    return str(tmp_path_factory.mktemp("videogen") / "mock_video_001.mp4")


class MockTrtllmServe(EchoServer):
    """trtllm-serve-shaped mock for `/v1/videos/generations`.

    Branches on the request body's `response_format` field (server wire values):
    - "file": returns DUMMY_VIDEO_BYTES as the response body.
    - anything else: returns {id, output_path} with the configured `video_path`.
    """

    def __init__(self, video_path: str) -> None:
        super().__init__(port=0)
        self.video_path = video_path

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/videos/generations", self._handle_videogen)

    async def _handle_videogen(self, request: web.Request) -> web.Response:
        body = await request.json()
        # sha1 not hash() — Python's hash() is salted per-interpreter via
        # PYTHONHASHSEED, which makes the mock id non-deterministic across runs.
        prompt = body.get("prompt", "")
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:4]
        video_id = f"mock_video_{digest}"
        if body.get("response_format") == "file":
            return web.Response(body=DUMMY_VIDEO_BYTES, content_type="video/mp4")
        return web.json_response({"id": video_id, "output_path": self.video_path})


@pytest.fixture(scope="module")
def mock_trtllm_serve(mock_video_path: str) -> Generator[MockTrtllmServe, None, None]:
    server = MockTrtllmServe(video_path=mock_video_path)
    server.start()
    yield server
    server.stop()


class MockTrtllmServeError(EchoServer):
    """Mock trtllm-serve that returns HTTP 500 for `/v1/videos/generations`."""

    def __init__(self) -> None:
        super().__init__(port=0)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/videos/generations", self._handle_error)

    async def _handle_error(self, request: web.Request) -> web.Response:
        return web.Response(status=500, text="Internal Server Error")


@pytest.fixture(scope="module")
def mock_trtllm_serve_error() -> Generator[MockTrtllmServeError, None, None]:
    server = MockTrtllmServeError()
    server.start()
    yield server
    server.stop()
