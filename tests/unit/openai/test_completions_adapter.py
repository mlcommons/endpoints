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

"""Unit tests for OpenAITextCompletionsAdapter."""

import json

import msgspec
import pandas as pd
import pytest
from inference_endpoint.config.schema import (
    BenchmarkConfig,
    ModelParams,
    StreamingMode,
    TestType,
)
from inference_endpoint.core.types import Query, TextModelOutput
from inference_endpoint.dataset_manager.predefined.aime25 import (
    presets as aime25_presets,
)
from inference_endpoint.dataset_manager.predefined.gpqa import presets as gpqa_presets
from inference_endpoint.dataset_manager.predefined.livecodebench import (
    presets as livecodebench_presets,
)
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    Harmonize,
)
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.completions_adapter import OpenAITextCompletionsAdapter
from inference_endpoint.openai.types import (
    SSEChoice,
    SSEDelta,
    SSEMessage,
    TextCompletionSSEChoice,
    TextCompletionSSEMessage,
)

_COMPLETION = {
    "model": "m",
    "input_tokens": [1, 2, 3],
    "max_tokens": 8,
    "min_tokens": 1,
}
# The exact body sent when usage is not requested.
_MAIN_COMPLETION_BODY = {
    True: b'{"model":"m","prompt":[1,2,3],"max_tokens":8,"min_tokens":1,"stream":true}',
    False: b'{"model":"m","prompt":[1,2,3],"max_tokens":8,"min_tokens":1,'
    b'"stream":false}',
}


class TestOpenAITextCompletionsAdapterDatasetTransforms:
    @pytest.mark.unit
    def test_transforms_include_harmonize(self):
        params = ModelParams(name="test-model")
        transforms = OpenAITextCompletionsAdapter.dataset_transforms(params)
        assert any(isinstance(t, Harmonize) for t in transforms)

    @pytest.mark.unit
    def test_streaming_flag_set_correctly(self):
        params = ModelParams(name="m", streaming=StreamingMode.ON)
        transforms = OpenAITextCompletionsAdapter.dataset_transforms(params)
        static = next(t for t in transforms if isinstance(t, AddStaticColumns))
        assert static.data["stream"] is True

    @pytest.mark.unit
    def test_streaming_off_by_default(self):
        params = ModelParams(name="m")
        transforms = OpenAITextCompletionsAdapter.dataset_transforms(params)
        static = next(t for t in transforms if isinstance(t, AddStaticColumns))
        assert static.data["stream"] is False

    @pytest.mark.unit
    def test_generation_controls_flow_from_model_params_to_request(self):
        params = ModelParams(name="m", min_new_tokens=0, skip_special_tokens=False)
        data = pd.DataFrame({"input_tokens": [[10, 20]]})
        for transform in OpenAITextCompletionsAdapter.dataset_transforms(params):
            data = transform(data)

        payload = json.loads(
            OpenAITextCompletionsAdapter.encode_query(
                Query(data=data.to_dict(orient="records")[0]), include_usage=False
            )
        )
        assert payload["min_tokens"] == 0
        assert payload["skip_special_tokens"] is False

    @pytest.mark.unit
    def test_generation_controls_use_defaults_in_request(self):
        params = ModelParams(name="m")
        data = pd.DataFrame({"input_tokens": [[10, 20]]})
        for transform in OpenAITextCompletionsAdapter.dataset_transforms(params):
            data = transform(data)

        payload = json.loads(
            OpenAITextCompletionsAdapter.encode_query(
                Query(data=data.to_dict(orient="records")[0]), include_usage=False
            )
        )
        assert payload["min_tokens"] == 1
        assert payload["skip_special_tokens"] is True


class TestOpenAITextCompletionsAdapterEncodeQuery:
    @pytest.mark.unit
    def test_encode_with_token_ids(self):
        query = Query(data={"input_tokens": [1, 2, 3], "model": "m", "max_tokens": 100})
        payload = json.loads(
            OpenAITextCompletionsAdapter.encode_query(query, include_usage=False)
        )
        assert payload["prompt"] == [1, 2, 3]
        assert payload["model"] == "m"
        assert payload["max_tokens"] == 100

    @pytest.mark.unit
    def test_encode_missing_input_tokens_raises(self):
        query = Query(data={"model": "m"})
        with pytest.raises(KeyError, match="input_tokens"):
            OpenAITextCompletionsAdapter.encode_query(query, include_usage=False)

    @pytest.mark.unit
    def test_encode_omits_none_fields(self):
        query = Query(data={"input_tokens": [10, 20], "model": "m"})
        payload = json.loads(
            OpenAITextCompletionsAdapter.encode_query(query, include_usage=False)
        )
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "min_tokens" not in payload
        assert "skip_special_tokens" not in payload

    @staticmethod
    def _body(include_usage: bool, **data) -> dict:
        query = Query(data={**_COMPLETION, **data})
        return json.loads(
            OpenAITextCompletionsAdapter.encode_query(
                query, include_usage=include_usage
            )
        )

    @pytest.mark.unit
    def test_streaming_body_requests_usage_when_enabled(self):
        body = self._body(True, stream=True)
        assert body["stream_options"] == {"include_usage": True}

    @pytest.mark.unit
    @pytest.mark.parametrize("stream", [True, False])
    def test_body_matches_main_when_usage_disabled(self, stream):
        query = Query(data={**_COMPLETION, "stream": stream})
        assert (
            OpenAITextCompletionsAdapter.encode_query(query, include_usage=False)
            == _MAIN_COMPLETION_BODY[stream]
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "data", [{"stream": False}, {}], ids=["stream-false", "unset"]
    )
    def test_non_streaming_body_never_requests_usage(self, data):
        assert "stream_options" not in self._body(True, **data)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "n, expected",
        [(2, "absent"), (2.0, "absent"), (1, {"include_usage": True})],
    )
    def test_n_gt_1_row_omits_stream_options(self, n, expected):
        body = self._body(True, stream=True, n=n)
        assert body["n"] == n
        assert body.get("stream_options", "absent") == expected


class TestOpenAITextCompletionsAdapterDecodeResponse:
    @pytest.mark.unit
    def test_decode_non_streaming(self):
        response_bytes = json.dumps(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1234567890,
                "model": "gpt-oss-120b",
                "choices": [
                    {"index": 0, "text": "hello world", "finish_reason": "stop"}
                ],
                "usage": None,
            }
        ).encode()
        result = OpenAITextCompletionsAdapter.decode_response(response_bytes, "qid-1")
        assert result.id == "qid-1"
        assert isinstance(result.response_output, TextModelOutput)
        assert result.response_output.output == "hello world"
        assert result.metadata["finish_reason"] == "stop"

    @pytest.mark.unit
    def test_decode_empty_choices_raises(self):
        response_bytes = json.dumps(
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1234567890,
                "model": "m",
                "choices": [],
                "usage": None,
            }
        ).encode()
        with pytest.raises(ValueError, match="at least one choice"):
            OpenAITextCompletionsAdapter.decode_response(response_bytes, "qid-1")

    @pytest.mark.unit
    def test_decode_non_streaming_copies_usage(self):
        body = {
            "id": "c",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
        result = OpenAITextCompletionsAdapter.decode_response(
            json.dumps(body).encode(), "q"
        )
        assert result.metadata == {
            "finish_reason": "stop",
            "prompt_tokens": 4,
            "completion_tokens": 2,
        }

    @pytest.mark.unit
    def test_completions_body_without_usage_key_decodes(self):
        body = {
            "id": "c",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
        }
        result = OpenAITextCompletionsAdapter.decode_response(
            json.dumps(body).encode(), "q"
        )
        assert result.response_output.output == "hi"
        assert result.metadata == {"finish_reason": "stop"}

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "usage", [{}, "x", [], {"prompt_tokens": 4.0, "completion_tokens": -2}]
    )
    def test_non_streaming_malformed_usage_succeeds(self, usage):
        body = {
            "id": "c",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "text": "hi", "finish_reason": "stop"}],
            "usage": usage,
        }
        result = OpenAITextCompletionsAdapter.decode_response(
            json.dumps(body).encode(), "q"
        )
        assert result.error is None
        assert result.response_output.output == "hi"
        assert result.metadata == {"finish_reason": "stop"}

    @pytest.mark.unit
    def test_non_streaming_n_gt_1_has_no_counts(self):
        body = {
            "id": "c",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "text": "hi", "finish_reason": "stop"},
                {"index": 1, "text": "yo", "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
        }
        result = OpenAITextCompletionsAdapter.decode_response(
            json.dumps(body).encode(), "q"
        )
        assert result.response_output.output == "hi"
        assert result.metadata == {"finish_reason": "stop"}


class TestOpenAITextCompletionsAdapterDecodeSSE:
    @pytest.mark.unit
    def test_decode_sse_message_returns_sse_choice(self):
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text="tok"),))
        json_bytes = msgspec.json.encode(msg)
        choice = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        assert isinstance(choice, SSEChoice)
        assert choice.delta is not None
        assert choice.delta.content == "tok"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "frame, expected",
        [
            pytest.param({"choices": []}, None, id="empty-choices-no-usage"),
            pytest.param(
                {"choices": [{"index": 0, "text": "tok"}], "usage": None},
                SSEChoice(delta=SSEDelta(content="tok")),
                id="null-usage",
            ),
            pytest.param(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 9,
                        "total_tokens": 14,
                    },
                },
                SSEMessage(
                    usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 9,
                        "total_tokens": 14,
                    }
                ),
                id="usage-trailer",
            ),
            pytest.param(
                {
                    "choices": [{"index": 0, "text": "tok", "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
                SSEMessage(
                    choices=(
                        SSEChoice(
                            delta=SSEDelta(content="tok"), finish_reason="length"
                        ),
                    ),
                    usage={"prompt_tokens": 5, "completion_tokens": 9},
                ),
                id="choice-and-usage",
            ),
            pytest.param(
                {"choices": [{"index": 0, "text": "tok"}], "usage": "x"},
                SSEMessage(
                    choices=(SSEChoice(delta=SSEDelta(content="tok")),), usage="x"
                ),
                id="malformed-usage",
            ),
            pytest.param(
                {"choices": [{"index": 1, "text": "tok"}]},
                SSEChoice(delta=SSEDelta(content="tok")),
                id="second-choice-index-no-usage",
            ),
            pytest.param(
                {"choices": [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}]},
                SSEChoice(delta=SSEDelta(content="a")),
                id="two-choices-no-usage",
            ),
            pytest.param(
                {
                    "choices": [{"index": 0, "text": "a"}, {"index": 1, "text": "b"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9},
                },
                SSEMessage(
                    choices=(
                        SSEChoice(delta=SSEDelta(content="a")),
                        SSEChoice(delta=SSEDelta(content="b")),
                    ),
                    usage={"prompt_tokens": 5, "completion_tokens": 9},
                ),
                id="two-choices-with-usage",
            ),
        ],
    )
    def test_decode_sse_usage_carrier(self, frame, expected):
        assert (
            OpenAITextCompletionsAdapter.decode_sse_message(msgspec.json.encode(frame))
            == expected
        )

    @pytest.mark.unit
    def test_decode_sse_empty_text_returns_choice_with_empty_content(self):
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text=""),))
        json_bytes = msgspec.json.encode(msg)
        choice = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        assert isinstance(choice, SSEChoice)
        assert choice.delta is not None
        assert choice.delta.content == ""

    @pytest.mark.unit
    def test_sse_choice_compatible_with_openai_accumulator(self):
        acc = OpenAISSEAccumulator(query_id="q1", stream_all_chunks=True)
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text="hello"),))
        json_bytes = msgspec.json.encode(msg)
        choice = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        chunk = acc.add_chunk(choice)
        assert chunk is not None
        assert chunk.response_chunk == "hello"


class TestAPITypeIntegration:
    @pytest.mark.unit
    def test_completions_adapter_registered(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={
                "endpoints": ["http://x:8000"],
                "api_type": "openai_completions",
            },
            datasets=[{"path": "D"}],
        )
        assert config.settings.client.adapter is OpenAITextCompletionsAdapter

    @pytest.mark.unit
    def test_completions_accumulator_is_openai_accumulator(self):
        config = BenchmarkConfig(
            type=TestType.OFFLINE,
            model_params={"name": "M"},
            endpoint_config={
                "endpoints": ["http://x:8000"],
                "api_type": "openai_completions",
            },
            datasets=[{"path": "D"}],
        )
        assert config.settings.client.accumulator is OpenAISSEAccumulator


class TestPresetExistence:
    @pytest.mark.unit
    def test_aime25_gptoss_preset_exists(self):
        transforms = aime25_presets.gptoss()
        assert len(transforms) == 1

    @pytest.mark.unit
    def test_gpqa_gptoss_preset_exists(self):
        transforms = gpqa_presets.gptoss()
        assert len(transforms) == 1

    @pytest.mark.unit
    def test_livecodebench_gptoss_preset_exists(self):
        transforms = livecodebench_presets.gptoss()
        assert len(transforms) == 1
