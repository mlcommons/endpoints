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

"""Multi-turn conversation dataset for conversational AI benchmarking."""

from typing import Any

import pandas as pd

from ..config.schema import APIType, ModelParams, StreamingMode
from .dataset import Dataset
from .transforms import apply_transforms

# Known generation parameter fields to forward from dataset to API requests
# Known generation parameter fields to forward from dataset to API requests.
# Aligned with OpenAI API specification and openai_msgspec_adapter.py implementation.
# These parameters work in both single-turn and multi-turn modes.
GENERATION_PARAMS = {
    # Model selection
    "model",
    # Token/length limits
    "max_new_tokens",  # Internal parameter
    "max_completion_tokens",  # OpenAI standard
    # Streaming configuration
    "stream",
    # Sampling parameters
    "temperature",
    "top_p",
    "top_k",
    "seed",
    # Repetition/diversity control
    "repetition_penalty",  # Internal parameter
    "frequency_penalty",  # OpenAI standard
    "presence_penalty",  # OpenAI standard
    # Output control
    "stop",  # Stop sequences
    "n",  # Number of completions to generate
    # Advanced OpenAI features
    "logit_bias",  # Token probability adjustments
    "name",  # Entity name for role (NOT model name, e.g., 'Bob' for tracking)
    "user",  # End-user identifier for monitoring/abuse detection
    "chat_template",  # Custom chat formatting template
}


def _model_param_defaults(model_params: ModelParams | None) -> dict[str, Any]:
    """Build per-request defaults for multi-turn rows from model params.

    Multi-turn datasets use `content` and conversation metadata rather than the
    single-turn `prompt` field expected by adapter dataset transforms. Applying
    those transforms would drop the conversation schema before load_sample() can
    construct the messages array. Instead, we inject the request defaults here.
    """
    if model_params is None:
        return {}

    return {
        "model": model_params.name,
        "stream": model_params.streaming == StreamingMode.ON,
        "max_completion_tokens": model_params.max_new_tokens,
        "temperature": model_params.temperature,
        "top_p": model_params.top_p,
        "top_k": model_params.top_k,
        "repetition_penalty": model_params.repetition_penalty,
    }


class MultiTurnDataset(Dataset, dataset_id="multi_turn_conversations"):
    """Dataset for multi-turn conversations.

    Supports conversational AI benchmarking with turn sequencing and conversation history.
    Validates that conversations have proper structure (alternating user/assistant roles)
    and builds metadata for the scheduler to enforce turn ordering.

    Dataset format (JSONL):
        {"conversation_id": "c1", "turn": 1, "role": "user", "content": "...", "system": "..."}
        {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "..."}
        {"conversation_id": "c1", "turn": 3, "role": "user", "content": "..."}

    Required columns:
        - conversation_id: Unique identifier for each conversation
        - turn: Turn number within conversation (1-indexed)
        - role: Speaker role ("user" or "assistant")
        - content: Message content

    Optional columns:
        - system: System prompt associated with the conversation (typically set on the first user turn)
        - model: Model name override
        - max_new_tokens: Max tokens for this turn

    Attributes:
        conversation_metadata: Metadata dict containing:
            - samples: List of user turn metadata (index, conversation_id, turn, system)
            - num_conversations: Total number of unique conversations
            - max_turns_per_conv: Maximum turns in any conversation
    """

    COLUMN_NAMES = ["conversation_id", "turn", "role", "content"]

    def __init__(self, dataframe: pd.DataFrame, **kwargs):
        """Initialize multi-turn dataset.

        Args:
            dataframe: DataFrame with conversation data.
            **kwargs: Additional arguments passed to Dataset.__init__.

        Raises:
            ValueError: If conversation structure is invalid.
        """
        super().__init__(dataframe, **kwargs)
        self._validate_conversation_structure()
        self._validate_turn_numbering()
        self.conversation_metadata = self._build_metadata()
        self._user_turn_indices: list[int] | None = None

    def _validate_conversation_structure(self):
        """Validate conversations are well-formed (alternating user/assistant).

        Raises:
            ValueError: If any conversation has invalid role sequence.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        for conv_id, group in self.dataframe.groupby("conversation_id"):
            roles = group.sort_values("turn")["role"].tolist()

            # Check alternation: user, assistant, user, assistant...
            for i in range(len(roles)):
                expected_role = "user" if i % 2 == 0 else "assistant"
                if roles[i] != expected_role:
                    raise ValueError(
                        f"Conversation {conv_id} has invalid role sequence at position {i}: "
                        f"expected {expected_role}, got {roles[i]}"
                    )

    def _validate_turn_numbering(self):
        """Validate turn numbers are contiguous starting from 1.

        Ensures that:
        - User turns start at turn 1
        - Turn numbers are sequential with no gaps (1, 2, 3, ...)
        - Runtime assumptions about turn sequencing are met

        Raises:
            ValueError: If turn numbering is non-contiguous or doesn't start at 1.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"

        for conv_id, group in self.dataframe.groupby("conversation_id"):
            sorted_group = group.sort_values("turn")
            turns = sorted_group["turn"].tolist()
            roles = sorted_group["role"].tolist()

            # Extract user turns for start validation
            user_turns = [
                turn for turn, role in zip(turns, roles, strict=False) if role == "user"
            ]
            if user_turns and user_turns[0] != 1:
                raise ValueError(
                    f"Conversation {conv_id}: First user turn must be turn 1, got {user_turns[0]}. "
                    f"Multi-turn conversations must start with user turn 1."
                )

            # Validate contiguous numbering across all turns
            for i, (expected_turn, actual_turn) in enumerate(
                zip(range(1, len(turns) + 1), turns, strict=False), start=1
            ):
                if actual_turn != expected_turn:
                    raise ValueError(
                        f"Conversation {conv_id}: Non-contiguous turn numbering detected. "
                        f"Expected turn {expected_turn} at position {i}, found turn {actual_turn}. "
                        f"Turn sequence: {turns}. "
                        f"Turns must be sequential with no gaps (1, 2, 3, 4, ...)."
                    )

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata for scheduler (maps sample index to conversation context).

        Returns:
            Metadata dict with samples list, num_conversations, max_turns_per_conv,
            and user_turns_per_conversation.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        samples = []
        user_turns = self.dataframe[self.dataframe["role"] == "user"]

        # Count user turns per conversation for completion tracking
        user_turns_per_conv = user_turns.groupby("conversation_id").size().to_dict()

        for idx, row in user_turns.iterrows():
            sample_meta = {
                "index": idx,
                "conversation_id": row["conversation_id"],
                "turn": row["turn"],
            }
            # Only include system if it's a valid string (pandas returns nan for missing values)
            system_prompt = row.get("system")
            if system_prompt and isinstance(system_prompt, str):
                sample_meta["system"] = system_prompt
            samples.append(sample_meta)

        return {
            "samples": samples,
            "num_conversations": self.dataframe["conversation_id"].nunique(),
            "max_turns_per_conv": self.dataframe.groupby("conversation_id")["turn"]
            .max()
            .max(),
            "user_turns_per_conversation": user_turns_per_conv,
        }

    def load(
        self,
        adapter=None,
        api_type: APIType | None = None,
        model_params: ModelParams | None = None,
        force: bool = False,
    ):
        """Load dataset and build a dense user-turn index.

        Multi-turn benchmarks only issue user turns. Assistant turns remain in the
        backing data so the conversation structure can still be validated.

        Unlike single-turn datasets, multi-turn rows do not have a `prompt`
        column, so adapter dataset transforms are intentionally skipped here.
        They would apply a single-turn ColumnFilter and strip the conversation
        fields required by load_sample(). Request defaults from model_params are
        merged directly into the conversation rows instead.
        """
        if not force and self.data is not None:
            self._user_turn_indices = [
                index for index, row in enumerate(self.data) if row["role"] == "user"
            ]
            return

        df = self.dataframe
        if df is None:
            raise ValueError(
                f"Cannot load dataset {self.__class__.__name__}: dataframe is None"
            )

        transforms = []
        if self.transforms is not None:
            transforms.extend(self.transforms)

        if transforms:
            df = apply_transforms(df, transforms)

        defaults = _model_param_defaults(model_params)
        for key, value in defaults.items():
            if value is None:
                continue
            if key in df.columns:
                df[key] = df[key].where(pd.notna(df[key]), value)
            else:
                df[key] = value

        self.data = df.to_dict(orient="records")
        assert self.data is not None, "Failed to convert DataFrame to records"

        self._user_turn_indices = [
            index for index, row in enumerate(self.data) if row["role"] == "user"
        ]

    def load_sample(self, index: int) -> dict[str, Any]:
        """Load the Nth user turn as a benchmark sample."""
        assert self.data is not None, "Dataset not loaded. Call load() first."
        assert (
            self._user_turn_indices is not None
        ), "Dataset not loaded. Call load() first."
        row = self.data[self._user_turn_indices[index]]

        # Build base sample with required fields
        sample = {
            "conversation_id": row["conversation_id"],
            "turn": row["turn"],
            "role": row["role"],
            "content": row["content"],
        }

        # Forward all generation parameters that exist in row
        for param in GENERATION_PARAMS:
            if param in row:
                value = row[param]
                # Skip pandas NaN/None values
                if value is not None and (
                    not isinstance(value, float) or not pd.isna(value)
                ):
                    sample[param] = value

        # Set defaults for critical params if not present
        if "max_new_tokens" not in sample and "max_completion_tokens" not in sample:
            sample["max_new_tokens"] = 128
        if "stream" not in sample:
            sample["stream"] = False

        # Only include system if it's a valid string (pandas returns nan for missing values)
        system_prompt = row.get("system")
        if system_prompt and isinstance(system_prompt, str):
            sample["system"] = system_prompt

        # Include the dataset's reference assistant response (the turn immediately following
        # this user turn in the same conversation), so history uses ground-truth context.
        user_data_index = self._user_turn_indices[index]
        next_index = user_data_index + 1
        if next_index < len(self.data):
            next_row = self.data[next_index]
            if (
                next_row.get("role") == "assistant"
                and next_row.get("conversation_id") == row["conversation_id"]
            ):
                sample["dataset_assistant_response"] = next_row["content"]

        return sample

    def num_samples(self) -> int:
        assert (
            self._user_turn_indices is not None
        ), "Dataset not loaded. Call load() first."
        return len(self._user_turn_indices)
