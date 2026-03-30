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

"""Unit tests for multi-turn load generator parameter forwarding."""

import pandas as pd
import pytest
from inference_endpoint.config.schema import APIType, ModelParams, StreamingMode
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset


@pytest.mark.unit
def test_multi_turn_forwards_generation_params():
    """Test that generation parameters are forwarded from dataset to request."""
    # Create dataset with specific generation params
    df = pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "Hi",
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "max_completion_tokens": 256,
                "stream": True,
                "model": "test-model",
            }
        ]
    )

    dataset = MultiTurnDataset(df)
    dataset.load()

    sample = dataset.load_sample(0)

    # Verify all params are present
    assert sample["temperature"] == 0.8
    assert sample["top_p"] == 0.9
    assert sample["top_k"] == 50
    assert sample["repetition_penalty"] == 1.1
    assert sample["max_completion_tokens"] == 256
    assert sample["stream"] is True
    assert sample["model"] == "test-model"


@pytest.mark.unit
def test_multi_turn_forwards_partial_params():
    """Test that dataset forwards only params that are present."""
    # Create dataset with subset of params
    df = pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "Hi",
                "temperature": 0.7,
                # No top_p, top_k, etc.
                "model": "test-model",
            }
        ]
    )

    dataset = MultiTurnDataset(df)
    dataset.load()

    sample = dataset.load_sample(0)

    # Verify present params
    assert sample["temperature"] == 0.7
    assert sample["model"] == "test-model"

    # Verify absent params are not in sample
    assert "top_p" not in sample
    assert "top_k" not in sample
    assert "repetition_penalty" not in sample

    # Defaults should be set
    assert "stream" in sample  # Default: False
    assert sample["stream"] is False


@pytest.mark.unit
def test_multi_turn_skips_nan_values():
    """Test that NaN values from pandas are skipped."""
    import numpy as np

    # Create dataset with NaN values
    df = pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "Hi",
                "temperature": np.nan,
                "top_p": 0.9,
                "model": "test-model",
            }
        ]
    )

    dataset = MultiTurnDataset(df)
    dataset.load()

    sample = dataset.load_sample(0)

    # NaN temperature should be skipped
    assert "temperature" not in sample
    # Non-NaN top_p should be present
    assert sample["top_p"] == 0.9


@pytest.mark.unit
def test_multi_turn_load_applies_model_params_without_prompt_column():
    """Benchmark loading should not require single-turn prompt columns."""
    df = pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "Hi",
            },
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": "Hello",
            },
        ]
    )

    dataset = MultiTurnDataset(df)
    dataset.load(
        api_type=APIType.OPENAI,
        model_params=ModelParams(
            name="test-model",
            streaming=StreamingMode.ON,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.8,
            top_k=42,
            repetition_penalty=1.05,
        ),
    )

    sample = dataset.load_sample(0)

    assert sample["model"] == "test-model"
    assert sample["stream"] is True
    assert sample["max_completion_tokens"] == 512
    assert sample["temperature"] == 0.3
    assert sample["top_p"] == 0.8
    assert sample["top_k"] == 42
    assert sample["repetition_penalty"] == 1.05
