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

"""Tests that APIType.OPENAI_TRANSCRIPTIONS is registered and wired up correctly."""

from importlib import import_module

import pytest
from inference_endpoint.core.types import APIType
from inference_endpoint.endpoint_client.config import (
    ACCUMULATOR_MAP,
    ADAPTER_MAP,
    HTTPClientConfig,
)


def _resolve(path: str) -> type:
    module_path, class_name = path.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)


@pytest.mark.unit
def test_api_type_exists():
    assert APIType.OPENAI_TRANSCRIPTIONS == "openai_transcriptions"


@pytest.mark.unit
def test_default_route():
    assert APIType.OPENAI_TRANSCRIPTIONS.default_route() == "v1/audio/transcriptions"


@pytest.mark.unit
def test_in_adapter_map():
    path = ADAPTER_MAP[APIType.OPENAI_TRANSCRIPTIONS]
    assert _resolve(path).__name__ == "OpenAITranscriptionsAdapter"


@pytest.mark.unit
def test_in_accumulator_map():
    # An omitted entry silently falls back to the OpenAI SSE accumulator, which
    # would surface zero-output queries as successful on a non-streaming route.
    path = ACCUMULATOR_MAP[APIType.OPENAI_TRANSCRIPTIONS]
    assert _resolve(path).__name__ == "NonStreamingAccumulator"


@pytest.mark.unit
def test_client_config_resolves_the_pair():
    config = HTTPClientConfig(api_type=APIType.OPENAI_TRANSCRIPTIONS)
    assert config.adapter.__name__ == "OpenAITranscriptionsAdapter"
    assert config.accumulator.__name__ == "NonStreamingAccumulator"


@pytest.mark.unit
def test_adapter_declares_multipart_content_type():
    adapter = _resolve(ADAPTER_MAP[APIType.OPENAI_TRANSCRIPTIONS])
    assert adapter.CONTENT_TYPE.startswith("multipart/form-data; boundary=")


@pytest.mark.unit
def test_other_adapters_still_declare_json():
    for api_type in (APIType.OPENAI, APIType.OPENAI_COMPLETIONS, APIType.SGLANG):
        adapter = _resolve(ADAPTER_MAP[api_type])
        assert adapter.CONTENT_TYPE == "application/json"
