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
from ..exceptions import InputValidationError
from .dataset import Dataset
from .transforms import apply_transforms

# Known generation parameter fields to forward from dataset to API requests.
# Aligned with OpenAI API specification and openai_msgspec_adapter.py implementation.
# These parameters work in both single-turn and multi-turn modes.
GENERATION_PARAMS = {
    "model",
    "max_new_tokens",
    "max_completion_tokens",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "logit_bias",  # Token probability adjustments
    "name",  # Entity name for role (NOT model name, e.g., 'Bob' for tracking)
    "user",  # End-user identifier for monitoring/abuse detection
    "chat_template",  # Custom chat formatting template
    "tools",  # OpenAI tool definitions (list[dict]) for tool-calling models
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


def _expand_tool_results(row: dict) -> list[dict]:
    """Expand a tool row into one OpenAI tool message per result.

    All ``role: "tool"`` rows carry a ``tool_results`` array. Each entry expands to
    one OpenAI tool message with ``tool_call_id`` and ``content``.

    Returns an empty list if ``tool_results`` is absent or not a list (non-tool rows).
    """
    tool_results = row.get("tool_results")
    if not isinstance(tool_results, list):
        return []
    return [
        {
            "role": "tool",
            "tool_call_id": result.get("tool_call_id"),
            "content": result.get("content"),
        }
        for result in tool_results
    ]


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
        self._validate_conversation_grouping()
        self._validate_conversation_structure()
        self._validate_turn_numbering()
        self.conversation_metadata = self._build_metadata()
        self._client_turn_indices: list[int] | None = None

    def _validate_conversation_grouping(self) -> None:
        """Validate that all rows for each conversation_id appear consecutively in file order.

        Raises:
            InputValidationError: If rows for a conversation_id are interleaved with other conversations.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        seen: set[str] = set()
        last_conv: str | None = None
        for row in self.dataframe.to_dict(orient="records"):
            conv_id = str(row["conversation_id"])
            if conv_id != last_conv:
                if conv_id in seen:
                    raise InputValidationError(
                        f"Rows for conversation '{conv_id}' are not consecutive. "
                        "All rows for a conversation must appear together in the file."
                    )
                seen.add(conv_id)
                last_conv = conv_id

    def _validate_conversation_structure(self):
        """Validate conversations are well-formed.

        Accepts plain user/assistant alternation as well as tool sequences:
            user → assistant → tool → [assistant → tool]* → assistant → user

        Raises:
            ValueError: If any conversation has invalid role sequence.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"

        # Valid state transitions (flat 4-state machine — no assistant_tc node,
        # no tool→tool; converter always merges consecutive tool rows into tool_results)
        VALID_NEXT: dict[str, set[str]] = {
            "start": {"user"},
            "user": {"assistant"},
            "assistant": {"tool", "user"},
            "tool": {"assistant", "user"},
        }

        for conv_id, group in self.dataframe.groupby("conversation_id"):
            sorted_group = group.sort_values("turn")
            state = "start"

            for _, row in sorted_group.iterrows():
                role = row["role"]

                if role not in VALID_NEXT.get(state, set()):
                    raise ValueError(
                        f"Conversation {conv_id} has invalid role sequence at turn "
                        f"{row['turn']}: got '{role}' after state '{state}'"
                    )
                state = role

    def _validate_turn_numbering(self):
        """Validate turn numbers are consecutive starting at 1.

        Raises:
            ValueError: If turn numbers are not exactly 1, 2, 3, …, N.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"

        for conv_id, group in self.dataframe.groupby("conversation_id"):
            turns = sorted(group["turn"].tolist())
            expected = list(range(1, len(turns) + 1))
            if turns != expected:
                raise ValueError(
                    f"Conversation {conv_id}: Turn numbers must be consecutive starting at 1, "
                    f"got {turns}"
                )

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata for scheduler (maps sample index to conversation context).

        Pre-computes the complete message list for each client turn so that
        conversation history does not need to be accumulated at runtime.

        Returns:
            Metadata dict with samples list, num_conversations, max_turns_per_conv,
            client_turns_per_conversation, and pre_built_messages_by_key.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        samples = []
        client_turns_df = self.dataframe[self.dataframe["role"].isin(["user", "tool"])]

        # Count client turns (user + tool) per conversation for completion tracking
        client_turns_per_conv = (
            client_turns_df.groupby("conversation_id").size().to_dict()
        )

        # Map (conversation_id, turn) → complete message list ready to send to endpoint.
        # Each entry is: [system (optional)] + all prior rows formatted as messages
        #                + the current client turn message.
        # This includes assistant rows (tool dispatches or terminal responses)
        # so no runtime injection is required.
        pre_built_messages_by_key: dict[tuple, list[dict]] = {}

        for conv_id, group in self.dataframe.groupby("conversation_id"):
            sorted_group = group.sort_values("turn")
            client_rows = sorted_group[sorted_group["role"].isin(["user", "tool"])]

            # Extract system prompt from the first row that has it (typically turn 1)
            system_content: str | None = None
            for _, srow in sorted_group.iterrows():
                val = srow.get("system")
                if val and isinstance(val, str):
                    system_content = val
                    break

            for idx, row in client_rows.iterrows():
                t_n = int(row["turn"])

                messages: list[dict] = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})

                # All dataset rows strictly before this client turn (includes
                # assistant rows and prior tool results).
                prior_rows = sorted_group[sorted_group["turn"] < t_n]
                for _, prior_row in prior_rows.iterrows():
                    msg: dict[str, Any] = {}
                    for key in ("role", "content", "tool_calls"):
                        val = prior_row.get(key)
                        if val is not None and not (
                            isinstance(val, float) and pd.isna(val)
                        ):
                            msg[key] = val
                    if msg.get("role"):
                        # Expand merged parallel tool results: a single row with
                        # tool_results: [{tool_call_id, content}, ...] expands into
                        # one OpenAI tool message per result entry.
                        expanded = _expand_tool_results(msg)
                        if expanded:
                            messages.extend(expanded)
                        else:
                            messages.append(msg)

                # Append the current client turn message.
                # A merged parallel-tool row carries tool_results instead of a
                # single tool_call_id/content pair; expand to one message per result.
                expanded = _expand_tool_results(row)
                if expanded:
                    messages.extend(expanded)
                else:
                    cur: dict[str, Any] = {}
                    for key in ("role", "content"):
                        val = row.get(key)
                        if val is not None and not (
                            isinstance(val, float) and pd.isna(val)
                        ):
                            cur[key] = val
                    messages.append(cur)

                pre_built_messages_by_key[(conv_id, t_n)] = messages

                samples.append(
                    {
                        "index": idx,
                        "conversation_id": conv_id,
                        "turn": t_n,
                    }
                )

        return {
            "samples": samples,
            "num_conversations": self.dataframe["conversation_id"].nunique(),
            "max_turns_per_conv": self.dataframe.groupby("conversation_id")["turn"]
            .max()
            .max(),
            "client_turns_per_conversation": client_turns_per_conv,
            "pre_built_messages_by_key": pre_built_messages_by_key,
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
            self._client_turn_indices = [
                index
                for index, row in enumerate(self.data)
                if row["role"] in ("user", "tool")
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

        self._client_turn_indices = [
            index
            for index, row in enumerate(self.data)
            if row["role"] in ("user", "tool")
        ]

    def load_sample(self, index: int) -> dict[str, Any]:
        """Load the Nth client turn (user or tool) as a benchmark sample."""
        assert self.data is not None, "Dataset not loaded. Call load() first."
        assert (
            self._client_turn_indices is not None
        ), "Dataset not loaded. Call load() first."
        row = self.data[self._client_turn_indices[index]]

        content_val = row.get("content")
        sample: dict[str, Any] = {
            "conversation_id": row["conversation_id"],
            "turn": row["turn"],
            "role": row["role"],
        }
        if content_val is not None and not (
            isinstance(content_val, float) and pd.isna(content_val)
        ):
            sample["content"] = content_val

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

        # Attach pre-built message list (system + history + current turn).
        key = (row["conversation_id"], int(row["turn"]))
        pre_built = self.conversation_metadata.get("pre_built_messages_by_key", {}).get(
            key, []
        )
        sample["pre_built_messages"] = pre_built

        # Fields for use_dataset_history=False path (live history accumulation).
        sample["current_turn_message"] = pre_built[-1] if pre_built else {}
        first = pre_built[0] if pre_built else {}
        sample["system_content"] = (
            first.get("content") if first.get("role") == "system" else None
        )

        return sample

    def num_samples(self) -> int:
        assert (
            self._client_turn_indices is not None
        ), "Dataset not loaded. Call load() first."
        return len(self._client_turn_indices)
