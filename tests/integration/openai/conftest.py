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

"""Integration fixtures for the OpenAI transcriptions adapter.

`MockTranscriptionServer` subclasses `EchoServer` to reuse its background-thread
aiohttp lifecycle (port discovery, start/stop, ready event); only the route
differs. It parses the upload with aiohttp's real multipart reader, so a
malformed body fails here rather than silently round-tripping.
"""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from aiohttp import web
from inference_endpoint.testing.echo_server import EchoServer

#: 16-bit PCM WAV header plus a short run of samples. Only the bytes matter -
#: nothing in the client or the mock decodes the audio.
WAV_HEADER = (
    b"RIFF" + (36 + 512).to_bytes(4, "little") + b"WAVEfmt "
    b"\x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00"
    b"data" + (512).to_bytes(4, "little")
)


def write_wav(path: Path, payload: bytes | None = None) -> bytes:
    """Write a dummy WAV at `path` and return its bytes."""
    data = WAV_HEADER + (payload if payload is not None else bytes(range(256)) * 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


class MockTranscriptionServer(EchoServer):
    """vLLM-shaped mock for `POST /v1/audio/transcriptions`.

    Returns the transcript registered for the uploaded file's basename, or the
    basename itself when none is registered. Records each request's form fields
    and uploaded bytes on `self.requests` so tests can assert on the wire
    format rather than only on the decoded result.
    """

    def __init__(self, transcripts: dict[str, str] | None = None) -> None:
        super().__init__(port=0)
        self.transcripts = transcripts or {}
        self.requests: list[dict] = []

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/audio/transcriptions", self._handle_transcription)

    async def _handle_transcription(self, request: web.Request) -> web.Response:
        if not request.content_type.startswith("multipart/form-data"):
            return web.Response(
                status=415, text=f"expected multipart, got {request.content_type}"
            )

        fields: dict[str, str] = {}
        filename = ""
        audio = b""
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "file":
                filename = part.filename or ""
                audio = bytes(await part.read(decode=False))
            else:
                fields[part.name] = (await part.read(decode=False)).decode()

        self.requests.append({"fields": fields, "filename": filename, "audio": audio})
        if not audio:
            return web.Response(status=400, text="empty file part")

        text = self.transcripts.get(filename, filename)
        if fields.get("response_format", "json") != "json":
            return web.Response(
                status=400, text="mock only serves response_format=json"
            )
        return web.Response(
            body=json.dumps({"text": text, "duration": 29.5}).encode(),
            content_type="application/json",
        )


@pytest.fixture
def mock_transcription_server(
    request: pytest.FixtureRequest,
) -> Generator[MockTranscriptionServer, None, None]:
    transcripts = getattr(request, "param", None)
    server = MockTranscriptionServer(transcripts=transcripts)
    server.start()
    yield server
    server.stop()
