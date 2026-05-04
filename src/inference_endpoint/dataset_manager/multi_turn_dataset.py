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

import logging
from typing import Any

import pandas as pd

from ..config.schema import APIType, ModelParams
from ..exceptions import InputValidationError
from .dataset import Dataset
from .transforms import (
    AddDefaultColumns,
    AddStaticColumns,
    apply_transforms,
    get_transforms_for_api_type,
)

logger = logging.getLogger(__name__)


def _expand_tool_results(row: dict) -> list[dict]:
    """Expand a tool row into one OpenAI tool message per result.

    All ``role: "tool"`` rows carry a ``tool_results`` array. Each entry expands to
    one OpenAI tool message with ``tool_call_id`` and ``content``.

    Returns an empty list if ``tool_results`` is absent or not a list (non-tool rows).
    """
    tool_results = row.get("tool_results")
    if not isinstance(tool_results, list):
        return []
    if not tool_results:
        logger.warning(
            "Row has empty tool_results list (conversation_id=%s, turn=%s)",
            row.get("conversation_id"),
            row.get("turn"),
        )
        return []
    messages = []
    for i, result in enumerate(tool_results):
        tool_call_id = result.get("tool_call_id")
        content = result.get("content")
        if tool_call_id is None:
            raise InputValidationError(
                f"tool_results[{i}] in conversation {row.get('conversation_id')!r} "
                f"turn {row.get('turn')} is missing required field 'tool_call_id'"
            )
        if content is None:
            raise InputValidationError(
                f"tool_results[{i}] in conversation {row.get('conversation_id')!r} "
                f"turn {row.get('turn')} is missing required field 'content'"
            )
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )
    return messages


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
        - max_new_tokens / max_completion_tokens: Max tokens for this turn (alias; mapped to max_completion_tokens)

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
        assert self.dataframe is not None, "Dataframe must be initialized"
        self._conv_groups = dict(
            list(self.dataframe.groupby("conversation_id", sort=False))
        )
        self._validate_conversation_grouping()
        self._validate_conversation_structure()
        self._validate_turn_numbering()
        self.conversation_metadata = self._build_metadata()

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
        VALID_NEXT: dict[str, set[str]] = {
            "start": {"user"},
            "user": {"assistant"},
            "assistant": {"tool", "user"},
            "tool": {"assistant", "user"},
        }

        for conv_id, group in self._conv_groups.items():
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
        for conv_id, group in self._conv_groups.items():
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
        samples = []

        # Count client turns (user + tool) per conversation for completion tracking
        client_turns_per_conv = {
            str(conv_id): int(group["role"].isin(["user", "tool"]).sum())
            for conv_id, group in self._conv_groups.items()
        }

        # Map (conversation_id, turn) → complete message list ready to send to endpoint.
        # Each entry is: [system (optional)] + all prior rows formatted as messages
        #                + the current client turn message.
        # This includes assistant rows (tool dispatches or terminal responses)
        # so no runtime injection is required.
        pre_built_messages_by_key: dict[tuple, list[dict]] = {}
        current_turn_messages_by_key: dict[tuple, list[dict]] = {}
        system_prompts_by_conv: dict[str, str | None] = {}

        assert self.dataframe is not None, "Dataframe must be initialized"
        for conv_id, group in self._conv_groups.items():
            sorted_group = group.sort_values("turn")
            client_rows = sorted_group[sorted_group["role"].isin(["user", "tool"])]

            # Extract system prompt from the first row that has it (typically turn 1)
            system_content: str | None = None
            for _, srow in sorted_group.iterrows():
                val = srow.get("system")
                if val and isinstance(val, str):
                    system_content = val
                    break
            system_prompts_by_conv[str(conv_id)] = system_content

            for _, row in client_rows.iterrows():
                t_n = int(row["turn"])

                messages: list[dict] = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})

                # All dataset rows strictly before this client turn (includes
                # assistant rows and prior tool results).
                prior_rows = sorted_group[sorted_group["turn"] < t_n]
                for _, prior_row in prior_rows.iterrows():
                    msg: dict[str, Any] = {}
                    for key in ("role", "content", "tool_calls", "tool_results"):
                        val = prior_row.get(key)
                        if val is not None and not (
                            isinstance(val, float) and pd.isna(val)
                        ):
                            msg[key] = val
                    if (
                        msg.get("role") == "assistant"
                        and "tool_calls" in msg
                        and "content" not in msg
                    ):
                        msg["content"] = None
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
                current_turn_msgs: list[dict] = []
                expanded = _expand_tool_results(row)
                if expanded:
                    current_turn_msgs = expanded
                else:
                    cur: dict[str, Any] = {}
                    for key in ("role", "content"):
                        val = row.get(key)
                        if val is not None and not (
                            isinstance(val, float) and pd.isna(val)
                        ):
                            cur[key] = val
                    current_turn_msgs = [cur]
                messages.extend(current_turn_msgs)

                str_conv_id = str(conv_id)
                pre_built_messages_by_key[(str_conv_id, t_n)] = messages
                current_turn_messages_by_key[(str_conv_id, t_n)] = current_turn_msgs

                samples.append(
                    {
                        "conversation_id": str_conv_id,
                        "turn": t_n,
                    }
                )

        return {
            "samples": samples,
            "num_conversations": len(self._conv_groups),
            "max_turns_per_conv": max(
                g["turn"].max() for g in self._conv_groups.values()
            ),
            "client_turns_per_conversation": client_turns_per_conv,
            "pre_built_messages_by_key": pre_built_messages_by_key,
            "current_turn_messages_by_key": current_turn_messages_by_key,
            "system_prompts_by_conv": system_prompts_by_conv,
        }

    def load(
        self,
        adapter=None,
        api_type: APIType | None = None,
        model_params: ModelParams | None = None,
        force: bool = False,
    ):
        """Load dataset, apply adapter defaults, and pre-bake client-turn samples.

        Unlike single-turn datasets, multi-turn rows do not have a `prompt` column,
        so ColumnFilter (which requires prompt) is skipped. AddStaticColumns entries
        from the adapter are applied via AddDefaultColumns (fill-missing-only) so that
        per-row dataset overrides are preserved.

        After transforms, only client turns (user + tool) are stored in self.data as
        fully assembled sample dicts (with messages attached).
        load_sample() and num_samples() are inherited from the base class.
        """
        if not force and self.data is not None:
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

        # Extract AddStaticColumns defaults from adapter transforms and apply as
        # fill-missing-only (preserves per-row dataset values).
        if api_type is not None and model_params is not None:
            adapter_transforms = get_transforms_for_api_type(api_type, model_params)
            defaults: dict[str, Any] = {}
            for t in adapter_transforms:
                if isinstance(t, AddStaticColumns):
                    defaults.update(t.data)
            if defaults:
                df = AddDefaultColumns(defaults)(df)

        all_rows = df.to_dict(orient="records")

        # Pre-bake: assemble one complete sample dict per client turn.
        # NaN filtering replaces the GENERATION_PARAMS allowlist — any key whose
        # value is float NaN was absent in the original dataset row.
        pre_built = self.conversation_metadata.get("pre_built_messages_by_key", {})
        client_turn_samples: list[dict[str, Any]] = []

        # Collect per-conversation defaults from the first user row so that
        # fields like model/max_completion_tokens propagate to tool rows.
        _PROPAGATED_KEYS = {
            "model",
            "max_completion_tokens",
            "max_new_tokens",
            "stream",
        }
        conv_defaults: dict[str, dict[str, Any]] = {}
        for row in all_rows:
            cid = row.get("conversation_id")
            if cid not in conv_defaults and row.get("role") == "user":
                conv_defaults[cid] = {
                    k: row[k]
                    for k in _PROPAGATED_KEYS
                    if k in row
                    and row[k] is not None
                    and not (isinstance(row[k], float) and pd.isna(row[k]))
                }

        for row in all_rows:
            if row.get("role") not in ("user", "tool"):
                continue

            # Filter NaN values; keep all meaningful fields (extra keys are harmless
            # since adapters consume only what they recognize).
            sample: dict[str, Any] = {
                k: v
                for k, v in row.items()
                if v is not None and not (isinstance(v, float) and pd.isna(v))
            }
            # Strip dataset-internal fields that must not reach the endpoint.
            sample.pop("tool_results", None)
            sample.pop("tool_calls", None)

            # Fill missing propagated fields from the first user row of this conversation.
            for k, v in conv_defaults.get(row.get("conversation_id"), {}).items():
                if k not in sample:
                    sample[k] = v

            # max_new_tokens → max_completion_tokens alias
            if "max_completion_tokens" not in sample and "max_new_tokens" in sample:
                sample["max_completion_tokens"] = sample.pop("max_new_tokens")
            if "max_completion_tokens" not in sample:
                sample["max_completion_tokens"] = 128
            if "stream" not in sample:
                sample["stream"] = False

            # Attach pre-built message list (system + history + current turn).
            key = (str(row["conversation_id"]), int(row["turn"]))
            messages = pre_built.get(key, [])
            sample["messages"] = messages

            client_turn_samples.append(sample)

        self.data = client_turn_samples
