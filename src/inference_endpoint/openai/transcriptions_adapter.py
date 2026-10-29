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

"""OpenAI /v1/audio/transcriptions adapter for Whisper-class ASR endpoints."""

from pathlib import Path
from typing import ClassVar

import msgspec
from inference_endpoint.config.schema import ModelParams
from inference_endpoint.core.types import Query, QueryResult, TextModelOutput
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnFilter,
    Transform,
)
from inference_endpoint.endpoint_client.adapter_protocol import HttpRequestAdapter

from .types import TranscriptionResponse

#: Dataset column holding the path to each sample's audio file.
AUDIO_FILE_COLUMN = "audio_file"

#: The worker folds Content-Type into its pre-encoded request prefix once at
#: setup, so the boundary cannot vary per request and must be a constant. It is
#: long and non-ASCII-audio-like enough that a collision with WAV payload bytes
#: is not a practical concern.
_BOUNDARY = "----mlperf-inference-endpoints-audio-boundary-Q3nV7xTt"

_DASHES = f"--{_BOUNDARY}\r\n".encode("ascii")
_CLOSING = f"--{_BOUNDARY}--\r\n".encode("ascii")


def encode_multipart(
    fields: dict[str, str], filename: str, audio: bytes, mime_type: str
) -> bytes:
    """Encode `fields` plus one `file` part as an RFC 7578 multipart body.

    Args:
        fields: Text form fields, in the order they should be written.
        filename: Value of the file part's `filename` parameter.
        audio: Raw audio bytes.
        mime_type: Media type of the file part.
    """
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(_DASHES)
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(_DASHES)
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n".encode()
    )
    parts.append(audio)
    parts.append(b"\r\n")
    parts.append(_CLOSING)
    return b"".join(parts)


class OpenAITranscriptionsAdapter(HttpRequestAdapter):
    """Adapter for OpenAI /v1/audio/transcriptions (e.g. vLLM serving Whisper).

    The request is a `multipart/form-data` audio upload, so the body is not JSON
    and `CONTENT_TYPE` carries the boundary. The dataset row holds a *path*; the
    bytes are read here, inside the worker, so a full audio corpus never
    traverses the issue thread or the ZMQ frame. Keep the audio on fast local
    storage: the read happens inside the measured request.

    `language` is what makes the server build Whisper's
    `<|startoftranscript|><|LANG|><|transcribe|><|notimestamps|>` prompt, which
    is how this matches the MLPerf speech2text reference.

    The endpoint is non-streaming: OpenAI does not support `stream` for
    Whisper-class models, so `decode_sse_message` raises and the registered
    accumulator is `NonStreamingAccumulator`.
    """

    CONTENT_TYPE: ClassVar[str] = f"multipart/form-data; boundary={_BOUNDARY}"

    #: Form fields forwarded from the dataset row, in request order. `model`
    #: and `file` are required by the API; the rest are optional and omitted
    #: when absent or None so server-side defaults apply.
    FORM_FIELDS: ClassVar[tuple[str, ...]] = (
        "model",
        "language",
        "prompt",
        "response_format",
        "temperature",
    )

    #: WAV is the only format the MLPerf LibriSpeech corpus ships. Servers
    #: sniff the payload rather than trusting this part header (vLLM hands the
    #: bytes to librosa/soundfile), so a per-row MIME column buys nothing.
    MIME_TYPE: ClassVar[str] = "audio/wav"

    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        TranscriptionResponse
    )

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        metadata = {
            "model": model_params.name,
            "response_format": "json",
            "temperature": model_params.temperature,
        }
        return [
            ColumnFilter(
                required_columns=[AUDIO_FILE_COLUMN],
                optional_columns=list(cls.FORM_FIELDS),
            ),
            # overwrite=False so a dataset row that carries its own `language`
            # or `prompt` wins over the model_params defaults.
            AddStaticColumns(metadata, overwrite=False),
        ]

    @classmethod
    def encode_query(cls, query: Query) -> bytes:
        data = query.data
        if AUDIO_FILE_COLUMN not in data:
            raise KeyError(
                f"{AUDIO_FILE_COLUMN!r} not found in query.data keys: "
                f"{list(data.keys())}"
            )
        if data.get("stream"):
            raise ValueError(
                "OpenAITranscriptionsAdapter is non-streaming; remove `stream` "
                "from query.data and set model_params.streaming to 'off'."
            )
        audio_path = Path(data[AUDIO_FILE_COLUMN])
        fields = {
            name: str(data[name])
            for name in cls.FORM_FIELDS
            if data.get(name) is not None
        }
        return encode_multipart(
            fields, audio_path.name, audio_path.read_bytes(), cls.MIME_TYPE
        )

    @classmethod
    def decode_response(cls, response_bytes: bytes, query_id: str) -> QueryResult:
        resp = cls._response_decoder.decode(response_bytes)
        return QueryResult(
            id=query_id,
            response_output=TextModelOutput(output=resp.text),
        )

    @classmethod
    def decode_sse_message(cls, json_bytes: bytes) -> str:
        raise NotImplementedError(
            "OpenAITranscriptionsAdapter does not use SSE streaming"
        )
