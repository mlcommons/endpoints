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

"""Integration test fixtures for WAN2.2 adapter tests."""

import asyncio
import base64
import threading
from collections.abc import Generator

import pytest
from aiohttp import web

DUMMY_VIDEO_PATH = "/lustre/videos/mock_video_001.mp4"
# Minimal dummy video bytes returned in accuracy mode (base64-encoded in responses).
DUMMY_VIDEO_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 24


class MockTrtllmServe:
    """Lightweight aiohttp server mimicking trtllm-serve's video generation API.

    Supports both response formats:
    - response_format='video_path': returns VideoPathResponse JSON.
    - response_format='video_bytes': returns VideoPayloadResponse JSON with
      base64-encoded DUMMY_VIDEO_BYTES.
    """

    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port = 0
        self._actual_port: int | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._actual_port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(
                timeout=5
            )

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/videos/generations", self._handle_sync)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._actual_port = self._runner.addresses[0][1]
        self._ready.set()
        await asyncio.Event().wait()

    async def _shutdown(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle_sync(self, request: web.Request) -> web.Response:
        body = await request.json()
        video_id = f"mock_video_{hash(body.get('prompt', '')) & 0xFFFF:04x}"
        return web.json_response(
            {
                "video_id": video_id,
                "video_bytes": base64.b64encode(DUMMY_VIDEO_BYTES).decode(),
            }
        )


@pytest.fixture(scope="module")
def mock_trtllm_serve() -> Generator[MockTrtllmServe, None, None]:
    server = MockTrtllmServe()
    server.start()
    yield server
    server.stop()


class MockTrtllmServeError:
    """Mock trtllm-serve that returns HTTP 500 for all requests."""

    def __init__(self) -> None:
        self.host = "127.0.0.1"
        self.port = 0
        self._actual_port: int | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._actual_port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(
                timeout=5
            )

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/videos/generations", self._handle_error)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._actual_port = self._runner.addresses[0][1]
        self._ready.set()
        await asyncio.Event().wait()

    async def _shutdown(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle_error(self, request: web.Request) -> web.Response:
        return web.Response(status=500, text="Internal Server Error")


@pytest.fixture(scope="module")
def mock_trtllm_serve_error() -> Generator[MockTrtllmServeError, None, None]:
    server = MockTrtllmServeError()
    server.start()
    yield server
    server.stop()
