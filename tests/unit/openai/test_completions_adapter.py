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
from inference_endpoint.dataset_manager.transforms import AddStaticColumns, Harmonize
from inference_endpoint.openai.accumulator import OpenAISSEAccumulator
from inference_endpoint.openai.completions_adapter import OpenAITextCompletionsAdapter
from inference_endpoint.openai.types import (
    SSEDelta,
    TextCompletionSSEChoice,
    TextCompletionSSEMessage,
)


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


class TestOpenAITextCompletionsAdapterEncodeQuery:
    @pytest.mark.unit
    def test_encode_with_token_ids(self):
        query = Query(data={"input_tokens": [1, 2, 3], "model": "m", "max_tokens": 100})
        payload = json.loads(OpenAITextCompletionsAdapter.encode_query(query))
        assert payload["prompt"] == [1, 2, 3]
        assert payload["model"] == "m"
        assert payload["max_tokens"] == 100

    @pytest.mark.unit
    def test_encode_missing_input_tokens_raises(self):
        query = Query(data={"model": "m"})
        with pytest.raises(KeyError, match="input_tokens"):
            OpenAITextCompletionsAdapter.encode_query(query)

    @pytest.mark.unit
    def test_encode_omits_none_fields(self):
        query = Query(data={"input_tokens": [10, 20], "model": "m"})
        payload = json.loads(OpenAITextCompletionsAdapter.encode_query(query))
        assert "temperature" not in payload
        assert "top_p" not in payload


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


class TestOpenAITextCompletionsAdapterDecodeSSE:
    @pytest.mark.unit
    def test_decode_sse_message_returns_sse_delta(self):
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text="tok"),))
        json_bytes = msgspec.json.encode(msg)
        delta = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        assert isinstance(delta, SSEDelta)
        assert delta.content == "tok"

    @pytest.mark.unit
    def test_decode_sse_empty_choices_returns_empty_delta(self):
        msg = TextCompletionSSEMessage(choices=())
        json_bytes = msgspec.json.encode(msg)
        delta = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        assert isinstance(delta, SSEDelta)
        assert delta.content == ""

    @pytest.mark.unit
    def test_decode_sse_empty_text_returns_empty_delta(self):
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text=""),))
        json_bytes = msgspec.json.encode(msg)
        delta = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        assert isinstance(delta, SSEDelta)
        assert delta.content == ""

    @pytest.mark.unit
    def test_sse_delta_compatible_with_openai_accumulator(self):
        acc = OpenAISSEAccumulator(query_id="q1", stream_all_chunks=True)
        msg = TextCompletionSSEMessage(choices=(TextCompletionSSEChoice(text="hello"),))
        json_bytes = msgspec.json.encode(msg)
        delta = OpenAITextCompletionsAdapter.decode_sse_message(json_bytes)
        chunk = acc.add_chunk(delta)
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
