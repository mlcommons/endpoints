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

"""Tests for multi-turn dataset turn numbering validation.

Validates that the dataset loader catches invalid turn sequences
before they cause runtime failures.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset


@pytest.mark.unit
def test_validation_rejects_non_contiguous_turns_with_gap():
    """Test validation rejects turn sequence with gaps (1,2,5,6)."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        {"conversation_id": "c1", "turn": 5, "role": "user", "content": "msg2"},  # GAP!
        {"conversation_id": "c1", "turn": 6, "role": "assistant", "content": "resp2"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Non-contiguous turn numbering"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_turns_not_starting_at_one():
    """Test validation rejects conversations starting at turn 3."""
    data = [
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg"},
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "resp"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="First user turn must be turn 1"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_accepts_valid_contiguous_turns():
    """Test validation passes for valid turn sequence (1,2,3,4)."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg2"},
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "resp2"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(
            temp_path, format=DatasetFormat.JSONL
        )
        dataset.load()
        assert dataset.num_samples() == 2  # 2 user turns
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_turn_starting_at_zero():
    """Test validation rejects conversations starting at turn 0."""
    data = [
        {"conversation_id": "c1", "turn": 0, "role": "user", "content": "msg"},
        {"conversation_id": "c1", "turn": 1, "role": "assistant", "content": "resp"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="First user turn must be turn 1"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_multiple_conversations_mixed_validity():
    """Test validation catches invalid conversation among valid ones."""
    data = [
        # Valid conversation
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        # Invalid conversation - gap
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "msg2"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "resp2"},
        {"conversation_id": "c2", "turn": 5, "role": "user", "content": "msg3"},  # GAP
        {"conversation_id": "c2", "turn": 6, "role": "assistant", "content": "resp3"},
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="c2.*Non-contiguous"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_error_message_includes_conversation_id():
    """Test that error messages clearly identify which conversation is invalid."""
    data = [
        {
            "conversation_id": "customer_support_001",
            "turn": 1,
            "role": "user",
            "content": "msg",
        },
        {
            "conversation_id": "customer_support_001",
            "turn": 2,
            "role": "assistant",
            "content": "resp",
        },
        {
            "conversation_id": "customer_support_001",
            "turn": 10,
            "role": "user",
            "content": "msg",
        },  # Jump to 10
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)

        error_msg = str(exc_info.value)
        assert "customer_support_001" in error_msg
        assert "Expected turn 3" in error_msg or "found turn 10" in error_msg
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_with_dataframe_direct():
    """Test validation when creating dataset directly from DataFrame."""
    df = pd.DataFrame(
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": "resp1",
            },
            {"conversation_id": "c1", "turn": 5, "role": "user", "content": "msg2"},
        ]
    )

    with pytest.raises(ValueError, match="Non-contiguous"):
        MultiTurnDataset(df)
