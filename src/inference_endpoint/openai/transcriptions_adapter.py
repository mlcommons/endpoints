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

_DELIMITER = f"--{_BOUNDARY}"
_DASHES = f"{_DELIMITER}\r\n".encode("ascii")
_CLOSING = f"{_DELIMITER}--\r\n".encode("ascii")

#: Characters that would terminate a `Content-Disposition` parameter early, with
#: the percent escapes browsers substitute for them (WHATWG HTML, "multipart/
#: form-data encoding algorithm"). Servers decode these, so escaping keeps the
#: original value recoverable where rejecting the row would not.
_PARAM_ESCAPES = str.maketrans({'"': "%22", "\r": "%0D", "\n": "%0A"})


def _escape_param(value: str) -> str:
    """Escape a quoted-string header parameter (a `name=` or `filename=` value)."""
    return value.translate(_PARAM_ESCAPES)


def encode_multipart(
    fields: dict[str, str], filename: str, audio: bytes, mime_type: str
) -> bytes:
    """Encode `fields` plus one `file` part as an RFC 7578 multipart body.

    Both the parameter names and the filename are dataset-controlled, so they are
    escaped: an unescaped `"` or CRLF would close the parameter early and let a
    row inject its own headers or parts. Field *values* sit in a part's body
    where CRLF is legal, so they are checked only for the one string that would
    forge a part boundary.

    Args:
        fields: Text form fields, in the order they should be written.
        filename: Value of the file part's `filename` parameter.
        audio: Raw audio bytes.
        mime_type: Media type of the file part.

    Raises:
        ValueError: A field value contains the multipart delimiter.
    """
    parts: list[bytes] = []
    for name, value in fields.items():
        if _DELIMITER in value:
            raise ValueError(
                f"Form field {name!r} contains the multipart delimiter "
                f"{_DELIMITER!r}, which would forge a part boundary."
            )
        parts.append(_DASHES)
        parts.append(
            f'Content-Disposition: form-data; name="{_escape_param(name)}"'
            f"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(_DASHES)
    parts.append(
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{_escape_param(filename)}"\r\n'
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
    traverses the issue thread or the ZMQ frame.

    The read is synchronous on the worker's event loop, so it stalls every
    in-flight response in that worker process, not just this request. Keep the
    audio on fast local storage: a cold-cache or network-mounted read shows up
    as added latency across the whole worker.

    `language` is what makes the server build Whisper's
    `<|startoftranscript|><|LANG|><|transcribe|><|notimestamps|>` prompt instead
    of auto-detecting per clip, which is how this matches the MLPerf speech2text
    reference. `DEFAULT_LANGUAGE` supplies it; a dataset `language` column wins.

    The endpoint is non-streaming: OpenAI does not support `stream` for
    Whisper-class models, so `decode_sse_message` raises and the registered
    accumulator is `NonStreamingAccumulator`. Config validation rejects
    `streaming: on` for this api_type, and `ColumnFilter` admits no `stream`
    column, so `query.data` never carries one.
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

    #: Transcription language, sent on every request. The MLPerf speech2text
    #: corpus is English LibriSpeech and the reference decodes with the language
    #: pinned, so leaving the server to auto-detect per clip would be a different
    #: workload. A dataset `language` column overrides this per row.
    DEFAULT_LANGUAGE: ClassVar[str] = "en"

    _response_decoder: msgspec.json.Decoder = msgspec.json.Decoder(
        TranscriptionResponse
    )

    @classmethod
    def dataset_transforms(cls, model_params: ModelParams) -> list[Transform]:
        metadata = {
            "model": model_params.name,
            "language": cls.DEFAULT_LANGUAGE,
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
