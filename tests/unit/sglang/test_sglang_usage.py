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

"""Unit tests for SGLang meta_info decoding and server-reported token counts."""

import json

import msgspec
import pytest
from inference_endpoint.core.types import (
    PROMPT_TOKENS_KEY,
    TOKEN_HDR_HIGH,
    server_usage_count,
)
from inference_endpoint.sglang import accumulator as sglang_accumulator
from inference_endpoint.sglang.accumulator import SGLangSSEAccumulator
from inference_endpoint.sglang.adapter import SGLangGenerateAdapter as A
from inference_endpoint.sglang.types import SGLangSSEDelta

_STOP = {"type": "stop", "matched": 200002}


def _frame(text, ids, ct, *, fr=None, e2e=None, retr=("num_retractions", 0)) -> bytes:
    meta = {
        "id": "a1b2",
        "finish_reason": fr,
        "prompt_tokens": 7,
        "weight_version": "default",
        retr[0]: retr[1],
        "completion_tokens": ct,
        "cached_tokens": 0,
    }
    if e2e is not None:
        meta["e2e_latency"] = e2e
    body = {"text": text, "meta_info": meta} | (
        {} if ids is None else {"output_ids": ids}
    )
    return json.dumps(body).encode()


# Current SGLang: finish_reason null until the last frame; e2e_latency only on it;
# the stop token counts but adds no text.
_STREAM = [
    _frame("Hello", [9906], 1),
    _frame("Hello world", [1917], 2),
    _frame("Hello world", [], 3, fr=_STOP, e2e=0.042),
]


@pytest.mark.unit
class TestSGLangMetaInfo:
    def test_unfinished_frame_with_null_finish_reason_decodes(self):
        assert A.decode_sse_message(_STREAM[0]) == SGLangSSEDelta(
            text="Hello",
            token_delta=[9906],
            total_completion_tokens=1,
            has_retractions=False,
            prompt_tokens=7,
        )

    def test_minimal_meta_info_without_output_ids_decodes(self):
        d = A.decode_sse_message(
            b'{"text":"Hi","meta_info":{"prompt_tokens":7,"completion_tokens":1}}'
        )
        assert (d.token_delta, d.prompt_tokens, d.has_retractions) == ([], 7, False)

    @pytest.mark.parametrize(
        "retr,expected",
        [
            (("num_retractions", 2), True),
            (("num_retractions", 0), False),
            (("total_retractions", 1), True),
            (("total_retractions", 0), False),
        ],
    )
    def test_either_retraction_name_is_accepted(self, retr, expected):
        frame = _frame("x", [1], 1, fr=_STOP, e2e=0.1, retr=retr)
        assert A.decode_sse_message(frame).has_retractions is expected

    def test_completion_tokens_stays_required(self):
        body = json.loads(_STREAM[0])
        del body["meta_info"]["completion_tokens"]
        with pytest.raises(msgspec.ValidationError, match="completion_tokens"):
            A.decode_sse_message(json.dumps(body).encode())

    @pytest.mark.parametrize(
        "prompt_tokens", ["absent", None, 7.0, "7", True, -1], ids=repr
    )
    def test_unusable_prompt_tokens_decode_as_absent(self, prompt_tokens):
        body = json.loads(_STREAM[0])
        if prompt_tokens == "absent":
            del body["meta_info"]["prompt_tokens"]
        else:
            body["meta_info"]["prompt_tokens"] = prompt_tokens
        frame = json.dumps(body).encode()
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        acc.add_chunk(A.decode_sse_message(frame))
        for md in (
            acc.get_final_output().metadata,
            A.decode_response(frame, "q").metadata,
        ):
            assert "prompt_tokens" not in md and md["completion_tokens"] == 1


@pytest.mark.unit
class TestSGLangServerTokenCounts:
    def test_stream_final_output_carries_server_usage(self):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        chunks = [c for f in _STREAM if (c := acc.add_chunk(A.decode_sse_message(f)))]
        r = acc.get_final_output()
        assert r.response_output.output == "Hello world"
        assert (r.metadata["prompt_tokens"], r.metadata["completion_tokens"]) == (7, 3)
        assert r.metadata["token_ids"] == [9906, 1917] and "n_tokens" not in r.metadata
        assert len(chunks) == 2 and all("n_tokens" not in c.metadata for c in chunks)

    def test_prompt_tokens_captured_when_completion_count_never_changes(self):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=False)
        frame = _frame("", [], 0, fr={"type": "abort"}, e2e=0.01)
        assert acc.add_chunk(A.decode_sse_message(frame)) is None
        meta = acc.get_final_output().metadata
        assert (meta["prompt_tokens"], meta["completion_tokens"]) == (7, 0)

    def test_sglang_frame_without_prompt_tokens_keeps_earlier_value(self):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        later = json.loads(_STREAM[2])
        del later["meta_info"]["prompt_tokens"]
        for frame in (_STREAM[0], _STREAM[1], json.dumps(later).encode()):
            acc.add_chunk(A.decode_sse_message(frame))
        meta = acc.get_final_output().metadata
        assert (meta["prompt_tokens"], meta["completion_tokens"]) == (7, 3)

    @pytest.mark.parametrize("later", [7.0, "8", -1, None], ids=repr)
    def test_later_unusable_prompt_tokens_keep_the_earlier_count(self, later):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        acc.add_chunk(A.decode_sse_message(_STREAM[0]))
        for raw in _STREAM[1:]:
            body = json.loads(raw)
            body["meta_info"]["prompt_tokens"] = later
            acc.add_chunk(A.decode_sse_message(json.dumps(body).encode()))
        meta = acc.get_final_output().metadata
        assert (meta["prompt_tokens"], meta["completion_tokens"]) == (7, 3)

    @pytest.mark.parametrize(
        "sent, expected", [((7, 7, 8), 8), (("7", 7, 7), 7)], ids=["changed", "fixed"]
    )
    def test_a_changed_usable_prompt_tokens_is_taken(self, sent, expected):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        for raw, prompt_tokens in zip(_STREAM, sent, strict=True):
            body = json.loads(raw)
            body["meta_info"]["prompt_tokens"] = prompt_tokens
            acc.add_chunk(A.decode_sse_message(json.dumps(body).encode()))
        assert acc.get_final_output().metadata["prompt_tokens"] == expected

    def test_prompt_tokens_validated_once_per_distinct_value(self, monkeypatch):
        seen: list[object] = []

        def counting(value: object, key: str) -> int | None:
            if key == PROMPT_TOKENS_KEY:
                seen.append(value)
            return server_usage_count(value, key)

        monkeypatch.setattr(sglang_accumulator, "server_usage_count", counting)
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        for frame in (*_STREAM, _frame("Hello world!", [0], 4)):
            acc.add_chunk(A.decode_sse_message(frame))
        assert acc.get_final_output().metadata["prompt_tokens"] == 7
        assert seen == [7]

    def test_stream_without_prompt_tokens_still_reports_completion(self):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        for raw in _STREAM:
            body = json.loads(raw)
            del body["meta_info"]["prompt_tokens"]
            acc.add_chunk(A.decode_sse_message(json.dumps(body).encode()))
        meta = acc.get_final_output().metadata
        assert "prompt_tokens" not in meta and meta["completion_tokens"] == 3

    def test_no_decoded_frame_reports_no_counts(self):
        meta = (
            SGLangSSEAccumulator("q", stream_all_chunks=True)
            .get_final_output()
            .metadata
        )
        assert "prompt_tokens" not in meta and "completion_tokens" not in meta

    def test_non_streaming_response_carries_server_usage(self):
        r = A.decode_response(
            _frame("Hello world", [9906, 1917], 3, fr=_STOP, e2e=0.042), "q"
        )
        assert r.metadata == {
            "token_ids": [9906, 1917],
            "prompt_tokens": 7,
            "completion_tokens": 3,
        }


@pytest.mark.unit
class TestSGLangMalformedCountWarning:
    def test_streaming_malformed_prompt_tokens_warns_once(
        self, usage_warning_unlatched
    ):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        for raw, prompt_tokens in zip(_STREAM, (7.0, 7.0, "7"), strict=True):
            body = json.loads(raw)
            body["meta_info"]["prompt_tokens"] = prompt_tokens
            acc.add_chunk(A.decode_sse_message(json.dumps(body).encode()))
        assert "prompt_tokens" not in acc.get_final_output().metadata
        (record,) = usage_warning_unlatched.records
        assert "'prompt_tokens': 7.0 (type 'float')" in record.getMessage()

    def test_non_streaming_malformed_completion_tokens_warns(
        self, usage_warning_unlatched
    ):
        md = A.decode_response(_frame("Hi", [1], -1, fr=_STOP, e2e=0.1), "q").metadata
        assert md["prompt_tokens"] == 7 and "completion_tokens" not in md
        (record,) = usage_warning_unlatched.records
        assert "'completion_tokens': -1 (type 'int')" in record.getMessage()

    def test_stream_with_out_of_range_completion_tokens_warns(
        self, usage_warning_unlatched
    ):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        acc.add_chunk(A.decode_sse_message(_frame("Hi", [1], TOKEN_HDR_HIGH + 1)))
        meta = acc.get_final_output().metadata
        assert meta["prompt_tokens"] == 7 and "completion_tokens" not in meta
        (record,) = usage_warning_unlatched.records
        assert f"'completion_tokens': {TOKEN_HDR_HIGH + 1}" in record.getMessage()

    def test_well_formed_stream_never_warns(self, usage_warning_unlatched):
        acc = SGLangSSEAccumulator("q", stream_all_chunks=True)
        for frame in _STREAM:
            acc.add_chunk(A.decode_sse_message(frame))
        acc.get_final_output()
        assert usage_warning_unlatched.records == []
