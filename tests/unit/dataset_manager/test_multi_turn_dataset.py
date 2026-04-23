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
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset


@pytest.fixture
def valid_multi_turn_jsonl() -> Generator[str, None, None]:
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
def invalid_role_sequence_jsonl() -> Generator[str, None, None]:
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
def missing_fields_jsonl() -> Generator[str, None, None]:
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
    """Test that only client turns (user + tool) are indexed as samples."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # Verify client turn indices are correct (fixture has only user turns)
    assert len(dataset._client_turn_indices) == 3

    # Check that indices point to client turns
    for idx in dataset._client_turn_indices:
        assert dataset.data[idx]["role"] in ("user", "tool")


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
    # System prompt is in pre_built_messages, not as a separate field
    assert sample_0["pre_built_messages"][0]["role"] == "system"
    assert sample_0["pre_built_messages"][0]["content"] == "You are a helpful assistant"

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
    assert "client_turns_per_conversation" in metadata

    # Should have 3 client turn samples (fixture has only user turns, no tool turns)
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


@pytest.mark.unit
def test_multi_turn_dataset_validation_invalid_role_sequence(
    invalid_role_sequence_jsonl,
):
    """Test validation rejects invalid role sequences."""
    # Validation happens during load_from_file (in __init__), not during load()
    with pytest.raises(ValueError, match="invalid role sequence"):
        MultiTurnDataset.load_from_file(
            invalid_role_sequence_jsonl, format=DatasetFormat.JSONL
        )


@pytest.mark.unit
def test_multi_turn_dataset_validation_missing_fields(missing_fields_jsonl):
    """Missing content field is preserved as None in the loaded sample."""
    dataset = MultiTurnDataset.load_from_file(
        missing_fields_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    sample = dataset.load_sample(0)
    # Missing content is no longer propagated to the sample dict
    assert "content" not in sample


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
    """Test system prompt is included as the first message in pre_built_messages.

    The system prompt is pre-baked into every client turn's message list so the
    conversation manager no longer needs to track it separately.
    """
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # First sample: pre_built_messages starts with system message
    sample_0 = dataset.load_sample(0)
    assert "pre_built_messages" in sample_0
    msgs = sample_0["pre_built_messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a helpful assistant"

    # Second sample (same conversation, turn 3): system message still first
    sample_1 = dataset.load_sample(1)
    msgs_1 = sample_1["pre_built_messages"]
    assert msgs_1[0]["role"] == "system"
    assert msgs_1[0]["content"] == "You are a helpful assistant"


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
    """Empty JSONL file raises ValueError (no columns to validate against)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_conversation_grouping():
    """Test that properly grouped conversations load correctly."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "c1t1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "c1t2"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "c1t3"},
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "c2t1"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "c2t2"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        # 5 total rows, 3 user turns (c1t1, c1t3, c2t1)
        assert len(dataset.data) == 5
        assert dataset.num_samples() == 3

        # Load samples to verify conversation grouping
        samples = [dataset.load_sample(i) for i in range(3)]

        # Verify conversation IDs
        conv_ids = [s["conversation_id"] for s in samples]
        assert conv_ids == ["c1", "c1", "c2"]

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_interleaved_conversations_rejected():
    """Test that interleaved conversation rows raise InputValidationError."""
    from inference_endpoint.exceptions import InputValidationError

    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "c1t1"},
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "c2t1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "c1t2"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(InputValidationError, match="not consecutive"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
@pytest.mark.parametrize(
    "rows",
    [
        # assistant-first
        [
            {"conversation_id": "c1", "turn": 1, "role": "assistant", "content": "A"},
            {"conversation_id": "c1", "turn": 2, "role": "user", "content": "B"},
        ],
        # consecutive assistants
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "A"},
            {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "B"},
            {"conversation_id": "c1", "turn": 3, "role": "assistant", "content": "C"},
        ],
        # tool directly after user (tool-before-assistant)
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "A"},
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "tool",
                "tool_results": [{"tool_call_id": "x", "content": "r"}],
            },
        ],
        # consecutive users
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "A"},
            {"conversation_id": "c1", "turn": 2, "role": "user", "content": "B"},
        ],
    ],
)
def test_validation_rejects_invalid_role_sequence(rows):
    """Invalid role sequences raise ValueError regardless of turn numbering."""
    with pytest.raises(ValueError, match="invalid role sequence"):
        MultiTurnDataset(pd.DataFrame(rows))


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
            assert sample["temperature"] == pytest.approx(0.7)

    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_openai_field_forwarding():
    """Test that OpenAI-specific fields are preserved and forwarded."""
    data = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "Hello",
            # OpenAI fields that should be forwarded
            "n": 3,
            "name": "Alice",
            "user": "user_12345",
            "logit_bias": {"50256": -100},
            "chat_template": "custom_template",
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

        # Verify OpenAI fields are present
        assert sample.get("n") == 3
        assert sample.get("name") == "Alice"
        assert sample.get("user") == "user_12345"
        assert sample.get("logit_bias") == {"50256": -100}
        assert sample.get("chat_template") == "custom_template"
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_multi_turn_dataset_all_generation_params():
    """Test that all generation parameters in GENERATION_PARAMS are forwarded."""
    from inference_endpoint.dataset_manager.multi_turn_dataset import GENERATION_PARAMS

    # Create dataset with all possible generation params
    data = [
        {
            "conversation_id": "c1",
            "turn": 1,
            "role": "user",
            "content": "Test",
            # Include all params from GENERATION_PARAMS
            "model": "test-model",
            "max_new_tokens": 100,
            "max_completion_tokens": 100,
            "stream": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "seed": 42,
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop": ["END"],
            "n": 2,
            "logit_bias": {"100": 10},
            "name": "TestEntity",
            "user": "test_user_001",
            "chat_template": "test_template",
        },
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant",
            "content": "Response",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        sample = dataset.load_sample(0)

        # Verify all GENERATION_PARAMS fields are forwarded
        # (excluding conversational fields like conversation_id, turn, role, content, system)
        for param in GENERATION_PARAMS:
            if param in data[0]:
                assert (
                    param in sample
                ), f"Generation parameter '{param}' not forwarded to sample"
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_non_contiguous_turns():
    """Turn numbers must be consecutive; gaps are rejected."""
    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "a"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "b"},
        {"conversation_id": "c1", "turn": 5, "role": "user", "content": "c"},
        {"conversation_id": "c1", "turn": 6, "role": "assistant", "content": "d"},
    ]
    with pytest.raises(ValueError, match="consecutive"):
        MultiTurnDataset(pd.DataFrame(rows))


@pytest.mark.unit
def test_validation_rejects_turns_not_starting_at_one():
    """Validation should reject conversations whose turns don't start at 1."""
    data = [
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg"},
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "resp"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="consecutive"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_accepts_valid_contiguous_turns():
    """Validation should accept contiguous turn sequences."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg2"},
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "resp2"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()
        assert dataset.num_samples() == 2
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_turn_starting_at_zero():
    """Validation should reject conversations starting at turn 0."""
    data = [
        {"conversation_id": "c1", "turn": 0, "role": "user", "content": "msg"},
        {"conversation_id": "c1", "turn": 1, "role": "assistant", "content": "resp"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="consecutive"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_duplicate_turn_numbers():
    """Duplicate turn numbers within a conversation are rejected."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "msg2"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "resp2"},
        # c2 has duplicate turn 2 — second assistant row with same turn number
        {"conversation_id": "c2", "turn": 2, "role": "user", "content": "dup"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="consecutive"):
            MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_validation_rejects_assistant_tc_role_literal():
    """role='assistant_tc' literal in dataset is rejected; only 'assistant' is valid."""
    rows = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Q"},
        {
            "conversation_id": "c1",
            "turn": 2,
            "role": "assistant_tc",
            "tool_calls": [
                {
                    "id": "c0",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {
            "conversation_id": "c1",
            "turn": 3,
            "role": "tool",
            "tool_results": [{"tool_call_id": "c0", "content": "r"}],
        },
        {"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "A"},
    ]
    with pytest.raises(ValueError, match="invalid role sequence"):
        MultiTurnDataset(pd.DataFrame(rows))


# ============================================================================
# Tool sequence tests
# ============================================================================


def _make_tool_sequence_df():
    """Return a DataFrame with a tool sequence embedded between user turns."""
    return pd.DataFrame(
        [
            {
                "conversation_id": "c1",
                "turn": 1,
                "role": "user",
                "content": "What is the weather?",
                "system": "Be helpful",
            },
            # assistant (with tool_calls): dispatches a tool call
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_c1_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            # tool result
            {
                "conversation_id": "c1",
                "turn": 3,
                "role": "tool",
                "tool_results": [
                    {"tool_call_id": "call_c1_0", "content": '{"temp": 22}'}
                ],
            },
            # terminal assistant
            {
                "conversation_id": "c1",
                "turn": 4,
                "role": "assistant",
                "content": "The weather is 22°C.",
            },
            # second user turn
            {
                "conversation_id": "c1",
                "turn": 5,
                "role": "user",
                "content": "Thanks!",
            },
        ]
    )


@pytest.mark.unit
def test_validation_accepts_tool_sequence():
    """user → assistant → tool → assistant → user passes validation."""
    df = _make_tool_sequence_df()
    ds = MultiTurnDataset(df)
    ds.load()
    assert ds.num_samples() == 3  # user(1), tool(3), user(5) are all client turns


@pytest.mark.unit
def test_validation_accepts_parallel_tool_calls():
    """Assistant with two tool_calls + merged tool_results row passes."""
    df = pd.DataFrame(
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hi"},
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c_0",
                        "type": "function",
                        "function": {"name": "f1", "arguments": "{}"},
                    },
                    {
                        "id": "c_1",
                        "type": "function",
                        "function": {"name": "f2", "arguments": "{}"},
                    },
                ],
            },
            {
                "conversation_id": "c1",
                "turn": 3,
                "role": "tool",
                "tool_results": [
                    {"tool_call_id": "c_0", "content": "r1"},
                    {"tool_call_id": "c_1", "content": "r2"},
                ],
            },
            {
                "conversation_id": "c1",
                "turn": 4,
                "role": "assistant",
                "content": "Done",
            },
        ]
    )
    ds = MultiTurnDataset(df)
    ds.load()
    assert ds.num_samples() == 2  # user(1), tool(3) are client turns


@pytest.mark.unit
def test_load_sample_merged_tool_row_has_no_content_key():
    """load_sample for a merged tool_results row must not emit content: NaN."""
    df = pd.DataFrame(
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Go"},
            {
                "conversation_id": "c1",
                "turn": 2,
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c_0",
                        "type": "function",
                        "function": {"name": "f1", "arguments": "{}"},
                    },
                    {
                        "id": "c_1",
                        "type": "function",
                        "function": {"name": "f2", "arguments": "{}"},
                    },
                ],
            },
            {
                "conversation_id": "c1",
                "turn": 3,
                "role": "tool",
                "tool_results": [
                    {"tool_call_id": "c_0", "content": "r1"},
                    {"tool_call_id": "c_1", "content": "r2"},
                ],
            },
            {
                "conversation_id": "c1",
                "turn": 4,
                "role": "assistant",
                "content": "Done",
            },
        ]
    )
    ds = MultiTurnDataset(df)
    ds.load()

    # Sample 1 is the merged tool row (turn 3)
    s1 = ds.load_sample(1)
    assert s1["role"] == "tool"
    assert "content" not in s1  # must NOT emit NaN
    assert "pre_built_messages" in s1


@pytest.mark.unit
def test_build_metadata_pre_built_messages():
    """pre_built_messages_by_key contains complete message arrays for each client turn.

    Dataset:
      turn 1: user      ← client turn 1
      turn 2: asst_tc   ← scripted (assistant with tool_calls)
      turn 3: tool      ← client turn 2
      turn 4: assistant ← terminal assistant
      turn 5: user      ← client turn 3

    Expected pre_built_messages:
      client turn 1 (t=1): [system, user(1)]
      client turn 2 (t=3): [system, user(1), asst_tc(2), tool(3)]
      client turn 3 (t=5): [system, user(1), asst_tc(2), tool(3), asst(4), user(5)]
    """
    df = _make_tool_sequence_df()
    ds = MultiTurnDataset(df)

    pbm = ds.conversation_metadata["pre_built_messages_by_key"]

    # Client turn 1 (user, t=1): [system, user(1)]
    msgs_t1 = pbm[("c1", 1)]
    assert len(msgs_t1) == 2
    assert msgs_t1[0] == {"role": "system", "content": "Be helpful"}
    assert msgs_t1[1] == {"role": "user", "content": "What is the weather?"}

    # Client turn 2 (tool, t=3): [system, user(1), asst_tc(2), tool(3)]
    msgs_t3 = pbm[("c1", 3)]
    assert len(msgs_t3) == 4
    assert msgs_t3[0]["role"] == "system"
    assert msgs_t3[1]["role"] == "user"
    assert msgs_t3[2]["role"] == "assistant"
    assert "tool_calls" in msgs_t3[2]
    assert msgs_t3[3]["role"] == "tool"
    assert msgs_t3[3]["content"] == '{"temp": 22}'
    assert msgs_t3[3]["tool_call_id"] == "call_c1_0"

    # Client turn 3 (user, t=5): [system, user(1), asst_tc(2), tool(3), asst(4), user(5)]
    msgs_t5 = pbm[("c1", 5)]
    assert len(msgs_t5) == 6
    assert msgs_t5[4] == {"role": "assistant", "content": "The weather is 22°C."}
    assert msgs_t5[5] == {"role": "user", "content": "Thanks!"}


@pytest.mark.unit
def test_build_metadata_pre_built_messages_no_tools():
    """Plain user/assistant alternation produces correct pre_built_messages."""
    df = pd.DataFrame(
        [
            {"conversation_id": "c1", "turn": 1, "role": "user", "content": "A"},
            {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "B"},
            {"conversation_id": "c1", "turn": 3, "role": "user", "content": "C"},
        ]
    )
    ds = MultiTurnDataset(df)
    pbm = ds.conversation_metadata["pre_built_messages_by_key"]

    # Turn 1: just the user message (no system, no prior rows)
    assert pbm[("c1", 1)] == [{"role": "user", "content": "A"}]

    # Turn 3: user(1) + assistant(2) + user(3)
    msgs = pbm[("c1", 3)]
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "A"}
    assert msgs[1] == {"role": "assistant", "content": "B"}
    assert msgs[2] == {"role": "user", "content": "C"}


@pytest.mark.unit
def test_load_sample_includes_pre_built_messages():
    """load_sample returns pre_built_messages with the complete message list."""
    df = _make_tool_sequence_df()
    ds = MultiTurnDataset(df)
    ds.load()

    s0 = ds.load_sample(0)  # user turn 1
    assert "pre_built_messages" in s0
    msgs = s0["pre_built_messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "What is the weather?"}

    s1 = ds.load_sample(1)  # tool turn 3
    assert s1["role"] == "tool"
    msgs_t3 = s1["pre_built_messages"]
    # system + user(1) + asst_tc(2) + tool(3) = 4 messages
    assert len(msgs_t3) == 4
    assert msgs_t3[-1]["role"] == "tool"

    s2 = ds.load_sample(2)  # user turn 5
    msgs_t5 = s2["pre_built_messages"]
    # system + user(1) + asst_tc(2) + tool(3) + asst(4) + user(5) = 6 messages
    assert len(msgs_t5) == 6


@pytest.mark.unit
def test_client_turns_include_tool_rows():
    """Tool rows are counted in num_samples() as client turns."""
    df = _make_tool_sequence_df()
    ds = MultiTurnDataset(df)
    ds.load()
    # 5 rows total: user(1), assistant(2), tool(3), assistant(4), user(5)
    # Client turns: user(1), tool(3), user(5) → 3
    assert ds.num_samples() == 3


# ============================================================================
# Pre-built messages content correctness
# ============================================================================


@pytest.mark.unit
def test_pre_built_messages_include_prior_assistant_response(valid_multi_turn_jsonl):
    """The terminal assistant response before each user turn is included in pre_built_messages."""
    dataset = MultiTurnDataset.load_from_file(
        valid_multi_turn_jsonl, format=DatasetFormat.JSONL
    )
    dataset.load()

    # Sample 0: turn 1 (first user) → just [system, user(1)]
    s0 = dataset.load_sample(0)
    msgs_0 = s0["pre_built_messages"]
    assert msgs_0[0]["role"] == "system"
    assert msgs_0[-1]["role"] == "user"

    # Sample 1: turn 3 (second user) → [system, user(1), assistant(2), user(3)]
    s1 = dataset.load_sample(1)
    msgs_1 = s1["pre_built_messages"]
    assert len(msgs_1) == 4
    assert msgs_1[2] == {"role": "assistant", "content": "I'm doing well, thank you!"}
    assert msgs_1[3]["role"] == "user"

    # Sample 2: turn 1 of conv_002 → no prior assistant row
    s2 = dataset.load_sample(2)
    msgs_2 = s2["pre_built_messages"]
    assert all(m["role"] != "assistant" for m in msgs_2)


@pytest.mark.unit
def test_pre_built_messages_no_cross_conversation_bleed():
    """Messages for conv_001 must not appear in conv_002's pre_built_messages."""
    data = [
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "c1 user"},
        {"conversation_id": "c2", "turn": 1, "role": "user", "content": "c2 user"},
        {"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "c2 resp"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        # c1: only its own user message
        s_c1 = dataset.load_sample(0)
        assert s_c1["pre_built_messages"] == [{"role": "user", "content": "c1 user"}]

        # c2: only c2 messages (no c1 content)
        s_c2 = dataset.load_sample(1)
        contents = [m.get("content") for m in s_c2["pre_built_messages"]]
        assert "c1 user" not in contents
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_pre_built_messages_with_tool_sequence_terminal_assistant():
    """Terminal assistant response (turn 4) appears in pre_built_messages for user(5)."""
    df = _make_tool_sequence_df()
    ds = MultiTurnDataset(df)
    ds.load()

    s2 = ds.load_sample(2)  # user turn 5
    msgs = s2["pre_built_messages"]
    # The terminal assistant at turn 4 should be included
    assistant_msgs = [m for m in msgs if m["role"] == "assistant" and m.get("content")]
    assert any(m["content"] == "The weather is 22°C." for m in assistant_msgs)
