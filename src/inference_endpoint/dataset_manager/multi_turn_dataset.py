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

from ..config.schema import APIType, ModelParams
from .dataset import Dataset


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
        - role: Speaker role ("user", "assistant", or "system")
        - content: Message content

    Optional columns:
        - system: System prompt (only for first user turn)
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

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata for scheduler (maps sample index to conversation context).

        Returns:
            Metadata dict with samples list, num_conversations, and max_turns_per_conv.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        samples = []
        user_turns = self.dataframe[self.dataframe["role"] == "user"]

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
        """
        super().load(
            adapter=adapter,
            api_type=None,
            model_params=None,
            force=force,
        )

        assert self.data is not None
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

        # Build sample dict with required fields
        sample = {
            "conversation_id": row["conversation_id"],
            "turn": row["turn"],
            "role": row["role"],
            "content": row["content"],
            "model": row.get("model"),
            "max_new_tokens": row.get("max_new_tokens", 128),
            "stream": row.get("stream", False),
        }

        # Only include system if it's a valid string (pandas returns nan for missing values)
        system_prompt = row.get("system")
        if system_prompt and isinstance(system_prompt, str):
            sample["system"] = system_prompt

        return sample

    def num_samples(self) -> int:
        assert (
            self._user_turn_indices is not None
        ), "Dataset not loaded. Call load() first."
        return len(self._user_turn_indices)
