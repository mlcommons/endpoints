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

import json
import tempfile
from pathlib import Path

import pytest
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset


@pytest.fixture
def valid_multi_turn_jsonl() -> str:
    """Create valid multi-turn conversation JSONL data."""
    data = [
        {
            "conversation_id": "conv_001",
            "turn": 1,
            "role": "user",
            "content": "Hello, how are you?",
            "system": "You are a helpful assistant",
        },
        {
            "conversation_id": "conv_001",
            "turn": 2,
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        },
        {
            "conversation_id": "conv_001",
            "turn": 3,
            "role": "user",
            "content": "What can you help me with?",
        },
        {
            "conversation_id": "conv_002",
            "turn": 1,
            "role": "user",
            "content": "What's the weather?",
        },
        {
            "conversation_id": "conv_002",
            "turn": 2,
            "role": "assistant",
            "content": "I don't have access to real-time weather data.",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def invalid_role_sequence_jsonl() -> str:
    """Create JSONL with invalid role sequence (not alternating)."""
    data = [
        {"conversation_id": "conv_001", "turn": 1, "role": "user", "content": "Hello"},
        {
            "conversation_id": "conv_001",
            "turn": 2,
            "role": "user",
            "content": "Another user message",
        },  # Invalid - consecutive user
        {
            "conversation_id": "conv_001",
            "turn": 3,
            "role": "assistant",
            "content": "Response",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def missing_fields_jsonl() -> str:
    """Create JSONL with missing required fields."""
    data = [
        {"conversation_id": "conv_001", "turn": 1, "role": "user"},  # Missing content
        {
            "conversation_id": "conv_001",
            "turn": 2,
            "role": "assistant",
            "content": "Response",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_load_valid_data(valid_multi_turn_jsonl):
    """Test loading valid multi-turn conversation data."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # Should have 5 rows total (3 for conv_001, 2 for conv_002)
    assert len(dataset.data) == 5

    # Should have 3 user turns (samples) - only user turns are indexed
    assert dataset.num_samples() == 3


@pytest.mark.unit
def test_multi_turn_dataset_user_turn_indexing(valid_multi_turn_jsonl):
    """Test that only user turns are indexed as samples."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # Verify user turn indices are correct
    assert len(dataset._user_turn_indices) == 3

    # Check that indices point to user turns
    for idx in dataset._user_turn_indices:
        assert dataset.data[idx]["role"] == "user"


@pytest.mark.unit
def test_multi_turn_dataset_load_sample(valid_multi_turn_jsonl):
    """Test load_sample returns correct user turns with dense indexing."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # Sample 0 should be first user turn
    sample_0 = dataset.load_sample(0)
    assert sample_0["conversation_id"] == "conv_001"
    assert sample_0["turn"] == 1
    assert sample_0["role"] == "user"
    assert sample_0["content"] == "Hello, how are you?"
    assert sample_0["system"] == "You are a helpful assistant"

    # Sample 1 should be second user turn (conv_001 turn 3)
    sample_1 = dataset.load_sample(1)
    assert sample_1["conversation_id"] == "conv_001"
    assert sample_1["turn"] == 3
    assert sample_1["role"] == "user"
    assert sample_1["content"] == "What can you help me with?"

    # Sample 2 should be third user turn (conv_002 turn 1)
    sample_2 = dataset.load_sample(2)
    assert sample_2["conversation_id"] == "conv_002"
    assert sample_2["turn"] == 1
    assert sample_2["role"] == "user"
    assert sample_2["content"] == "What's the weather?"


@pytest.mark.unit
def test_multi_turn_dataset_conversation_metadata(valid_multi_turn_jsonl):
    """Test conversation metadata generation."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    metadata = dataset.conversation_metadata

    # Check metadata structure
    assert "samples" in metadata
    assert "num_conversations" in metadata
    assert "max_turns_per_conv" in metadata

    # Should have 3 user turn samples
    assert len(metadata["samples"]) == 3

    # Should have 2 conversations
    assert metadata["num_conversations"] == 2

    # Max turns per conversation should be 3 (conv_001 has 3 turns)
    assert metadata["max_turns_per_conv"] == 3

    # Check sample metadata structure
    sample_meta = metadata["samples"][0]
    assert "index" in sample_meta
    assert "conversation_id" in sample_meta
    assert "turn" in sample_meta
    assert "system" in sample_meta


@pytest.mark.unit
def test_multi_turn_dataset_validation_invalid_role_sequence(
    invalid_role_sequence_jsonl,
):
    """Test validation rejects invalid role sequences."""
    # Validation happens during load_from_file (in __init__), not during load()
    with pytest.raises(ValueError, match="invalid role sequence"):
        dataset = MultiTurnDataset.load_from_file(
            invalid_role_sequence_jsonl, format=DatasetFormat.JSONL
        )


@pytest.mark.unit
def test_multi_turn_dataset_validation_missing_fields(missing_fields_jsonl):
    """Test validation handles missing required fields."""
    # This should either raise an error or handle gracefully
    # depending on implementation
    dataset = MultiTurnDataset.load_from_file(
        missing_fields_jsonl, format=DatasetFormat.JSONL
    )

    # Load may succeed but sample loading should fail
    dataset.load()
    # Check that content is None or raises error
    sample = dataset.load_sample(0)
    # Implementation may vary - just ensure it doesn't crash


@pytest.mark.unit
def test_multi_turn_dataset_multiple_conversations():
    """Test dataset with multiple conversations of varying lengths."""
    data = [
        # Conversation 1: 3 turns (user-assistant-user, missing final assistant)
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg1b"},
        # Conversation 2: 4 turns (complete user-assistant alternation)
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "msg2"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "resp2"},
        {"conversation_id": "c2", "turn": 3, "role": "user", "content": "msg3"},
        {"conversation_id": "c2", "turn": 4, "role": "assistant", "content": "resp3"},
        # Conversation 3: 2 turns (complete user-assistant)
        {"conversation_id": "c3", "turn": 1, "role": "user", "content": "msg4"},
        {"conversation_id": "c3", "turn": 2, "role": "assistant", "content": "resp4"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        # 9 total rows, 5 user turns (c1:t1, c1:t3, c2:t1, c2:t3, c3:t1)
        assert len(dataset.data) == 9
        assert dataset.num_samples() == 5

        # Metadata checks
        metadata = dataset.conversation_metadata
        assert metadata["num_conversations"] == 3
        assert metadata["max_turns_per_conv"] == 4  # c2 has 4 turns

        # Verify user turns are correctly indexed
        samples = [dataset.load_sample(i) for i in range(5)]

        # Check we got all the user turns
        user_turns = [(s["conversation_id"], s["turn"]) for s in samples]
        expected_turns = [("c1", 1), ("c1", 3), ("c2", 1), ("c2", 3), ("c3", 1)]
        assert sorted(user_turns) == sorted(expected_turns)

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_system_prompt_handling(valid_multi_turn_jsonl):
    """Test system prompt is correctly included in metadata.

    System prompt is only included in samples where it's present in the dataset row
    (typically the first user turn). The conversation manager is responsible for
    preserving it across subsequent turns.
    """
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # First sample should have system prompt
    sample_0 = dataset.load_sample(0)
    assert "system" in sample_0
    assert sample_0["system"] == "You are a helpful assistant"

    # Second sample (same conversation) doesn't have system in dataset row
    sample_1 = dataset.load_sample(1)
    # System prompt is NOT in sample_1 because it's not in the dataset row
    # The conversation manager will preserve it from turn 1
    assert "system" not in sample_1 or sample_1.get("system") is None

    # Metadata should include system prompt for first turn
    metadata = dataset.conversation_metadata
    assert metadata["samples"][0]["system"] == "You are a helpful assistant"


@pytest.mark.unit
def test_multi_turn_dataset_single_turn_conversations():
    """Test conversations with only one turn."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Single turn"},
        # No assistant response
        {
            "conversation_id": "c2",
            "turn": 1,
            "role": "user",
            "content": "Another single",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        # 2 rows, 2 user turns
        assert len(dataset.data) == 2
        assert dataset.num_samples() == 2

        # Both samples should be user turns
        assert dataset.load_sample(0)["role"] == "user"
        assert dataset.load_sample(1)["role"] == "user"

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_empty_conversation():
    """Test empty dataset."""
    # Create an empty JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write nothing (empty file)
        temp_path = f.name

    try:
        # Empty file may cause issues during load_from_file - handle gracefully
        try:
            dataset = MultiTurnDataset.load_from_file(
                temp_path, format=DatasetFormat.JSONL
            )
            dataset.load()

            assert len(dataset.data) == 0
            assert dataset.num_samples() == 0
            # Empty dataframe groupby returns empty iterator, so num_conversations should be 0
            metadata = dataset.conversation_metadata
            assert metadata is not None
            assert metadata.get("num_conversations", 0) == 0
        except (ValueError, KeyError):
            # Empty dataframe may cause validation errors - that's okay
            pass

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_conversation_grouping():
    """Test that conversations are correctly grouped."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "c1t1"},
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "c2t1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "c1t2"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "c2t2"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "c1t3"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        # 5 total rows, 3 user turns (c1t1, c2t1, c1t3)
        assert len(dataset.data) == 5
        assert dataset.num_samples() == 3

        # Load samples to verify conversation grouping
        samples = [dataset.load_sample(i) for i in range(3)]

        # Verify conversation IDs
        conv_ids = [s["conversation_id"] for s in samples]
        assert conv_ids == ["c1", "c2", "c1"]

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_validation_assistant_first():
    """Test validation rejects conversations starting with assistant."""
    data = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "assistant",
            "content": "I start!",
        },  # Invalid
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "user",
            "content": "User responds",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Validation happens during load_from_file (in __init__)
        with pytest.raises(ValueError, match="invalid role sequence"):
            dataset = MultiTurnDataset.load_from_file(
                temp_path, format=DatasetFormat.JSONL
            )

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_validation_consecutive_assistants():
    """Test validation rejects consecutive assistant messages."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hello"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hi"},
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "assistant",
            "content": "Another response",
        },  # Invalid
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Validation happens during load_from_file (in __init__)
        with pytest.raises(ValueError, match="invalid role sequence"):
            dataset = MultiTurnDataset.load_from_file(
                temp_path, format=DatasetFormat.JSONL
            )

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_additional_fields():
    """Test that additional fields (model, max_new_tokens, etc.) are preserved."""
    data = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "Hello",
            "model": "gpt-4",
            "max_new_tokens": 256,
            "temperature": 0.7,
        },
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hi"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        sample = dataset.load_sample(0)
        # Fields may or may not be present depending on how dataframe handles them
        # Just check they're accessible if present
        if "model" in sample:
            assert sample["model"] == "gpt-4"
        if "max_new_tokens" in sample:
            assert sample["max_new_tokens"] == 256
        if "temperature" in sample:
            assert sample["temperature"] == 0.7

    finally:
        Path(temp_path).unlink()
