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

"""Unit tests for OpenAITranscriptionsAdapter."""

from pathlib import Path

import msgspec
import pandas as pd
import pytest
from inference_endpoint.config.schema import ModelParams
from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.transforms import apply_transforms
from inference_endpoint.openai.transcriptions_adapter import (
    OpenAITranscriptionsAdapter as Adapter,
)

WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(range(256))


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip-0001.wav"
    path.write_bytes(WAV_BYTES)
    return path


def _query(audio_file: Path, **extra) -> Query:
    data = {
        "audio_file": str(audio_file),
        "model": "openai/whisper-large-v3",
        "language": "en",
        "response_format": "json",
        "temperature": 0.0,
    }
    data.update(extra)
    return Query(id="q1", data=data)


@pytest.mark.unit
class TestContentType:
    def test_content_type_is_multipart_with_boundary(self):
        assert Adapter.CONTENT_TYPE.startswith("multipart/form-data; boundary=")

    def test_declared_boundary_matches_every_encoded_body(self, audio_file: Path):
        # The worker folds Content-Type into a prefix built once at setup, so a
        # body whose boundary differed from the declared one would be unparseable.
        boundary = Adapter.CONTENT_TYPE.split("boundary=", 1)[1].encode()
        for _ in range(2):
            body = Adapter.encode_query(_query(audio_file))
            assert body.startswith(b"--" + boundary + b"\r\n")


@pytest.mark.unit
class TestEncodeQuery:
    def test_body_contains_audio_bytes_verbatim(self, audio_file: Path):
        body = Adapter.encode_query(_query(audio_file))
        assert WAV_BYTES in body

    def test_body_carries_filename_and_part_content_type(self, audio_file: Path):
        body = Adapter.encode_query(_query(audio_file))
        assert b'name="file"; filename="clip-0001.wav"' in body
        assert b"Content-Type: audio/wav" in body

    def test_body_carries_form_fields(self, audio_file: Path):
        body = Adapter.encode_query(_query(audio_file))
        for name, value in (
            ("model", b"openai/whisper-large-v3"),
            ("language", b"en"),
            ("response_format", b"json"),
            ("temperature", b"0.0"),
        ):
            assert f'name="{name}"'.encode() in body
            assert value in body

    def test_body_is_closed_with_terminating_boundary(self, audio_file: Path):
        boundary = Adapter.CONTENT_TYPE.split("boundary=", 1)[1]
        body = Adapter.encode_query(_query(audio_file))
        assert body.endswith(f"--{boundary}--\r\n".encode())

    def test_absent_optional_fields_are_omitted(self, audio_file: Path):
        query = Query(id="q", data={"audio_file": str(audio_file), "model": "m"})
        body = Adapter.encode_query(query)
        assert b'name="language"' not in body
        assert b'name="temperature"' not in body

    def test_none_valued_fields_are_omitted(self, audio_file: Path):
        # AddStaticColumns materialises model_params defaults that may be None;
        # sending "None" as a form value would be rejected by the server.
        query = _query(audio_file, temperature=None)
        assert b'name="temperature"' not in Adapter.encode_query(query)

    def test_missing_audio_file_raises_keyerror(self):
        with pytest.raises(KeyError, match="audio_file"):
            Adapter.encode_query(Query(id="q", data={"model": "m"}))

    def test_streaming_request_raises(self, audio_file: Path):
        with pytest.raises(ValueError, match="non-streaming"):
            Adapter.encode_query(_query(audio_file, stream=True))

    def test_missing_audio_on_disk_raises(self, tmp_path: Path):
        query = Query(
            id="q", data={"audio_file": str(tmp_path / "gone.wav"), "model": "m"}
        )
        with pytest.raises(FileNotFoundError):
            Adapter.encode_query(query)


@pytest.mark.unit
class TestDecodeResponse:
    def test_decodes_text_field(self):
        result = Adapter.decode_response(b'{"text": "hello world"}', "q1")
        assert result.id == "q1"
        assert result.error is None
        assert str(result.response_output) == "hello world"

    def test_ignores_server_specific_extras(self):
        body = b'{"text": "hi", "duration": 29.9, "language": "english"}'
        assert str(Adapter.decode_response(body, "q").response_output) == "hi"

    def test_missing_text_raises(self):
        with pytest.raises(msgspec.ValidationError):
            Adapter.decode_response(b'{"duration": 1.0}', "q")

    def test_malformed_json_raises(self):
        with pytest.raises(msgspec.DecodeError):
            Adapter.decode_response(b"not json", "q")


@pytest.mark.unit
class TestSse:
    def test_decode_sse_message_raises(self):
        with pytest.raises(NotImplementedError):
            Adapter.decode_sse_message(b'{"text": "x"}')


@pytest.mark.unit
class TestDatasetTransforms:
    def _apply(self, df: pd.DataFrame, **model_kwargs) -> pd.DataFrame:
        params = ModelParams(name="openai/whisper-large-v3", **model_kwargs)
        return apply_transforms(df, Adapter.dataset_transforms(params))

    def test_drops_ground_truth_columns_from_the_request_row(self):
        df = pd.DataFrame(
            [{"audio_file": "/a.wav", "transcript": "hello", "duration": 29.5}]
        )
        out = self._apply(df)
        assert "transcript" not in out.columns
        assert "duration" not in out.columns
        assert out.iloc[0]["audio_file"] == "/a.wav"

    def test_adds_model_and_response_format(self):
        df = pd.DataFrame([{"audio_file": "/a.wav", "transcript": "hello"}])
        row = self._apply(df, temperature=0.0).iloc[0]
        assert row["model"] == "openai/whisper-large-v3"
        assert row["response_format"] == "json"
        assert row["temperature"] == 0.0

    def test_dataset_language_overrides_defaults(self):
        df = pd.DataFrame(
            [{"audio_file": "/a.wav", "transcript": "hola", "language": "es"}]
        )
        assert self._apply(df).iloc[0]["language"] == "es"

    def test_missing_audio_file_column_raises(self):
        df = pd.DataFrame([{"transcript": "hello"}])
        with pytest.raises(KeyError, match="audio_file"):
            self._apply(df)
