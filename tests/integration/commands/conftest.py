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

"""Fixtures shared by the CLI command integration tests."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from aiohttp import web
from inference_endpoint.testing.echo_server import EchoServer


class RecordingEchoServer(EchoServer):
    """Echo server that keeps every chat and text-completion request body."""

    def __init__(self, *, report_usage: bool = True) -> None:
        super().__init__(port=0, report_usage=report_usage)
        self.bodies: list[dict[str, Any]] = []

    async def _handle_echo_chat_completions_request(
        self, request: web.Request
    ) -> web.Response:
        # aiohttp caches the body, so the parent handler can read it again.
        self.bodies.append(json.loads(await request.read()))
        return await super()._handle_echo_chat_completions_request(request)

    async def _handle_echo_completions_request(
        self, request: web.Request
    ) -> web.StreamResponse:
        self.bodies.append(json.loads(await request.read()))
        return await super()._handle_echo_completions_request(request)


@pytest.fixture
def recording_echo_server_factory() -> Iterator[Callable[..., RecordingEchoServer]]:
    """Yields ``factory(*, report_usage=True) -> RecordingEchoServer``.

    Each server is started on a free port and stopped at test teardown.
    """
    servers: list[RecordingEchoServer] = []

    def factory(*, report_usage: bool = True) -> RecordingEchoServer:
        server = RecordingEchoServer(report_usage=report_usage)
        server.start()
        servers.append(server)
        return server

    yield factory

    for server in servers:
        server.stop()
