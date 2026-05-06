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
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from ..config.schema import APIType, ModelParams
from ..exceptions import InputValidationError
from .cache_salt import apply_salt, compute_salt
from .dataset import Dataset
from .transforms import (
    AddStaticColumns,
    apply_transforms,
    get_transforms_for_api_type,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationSampleEntry:
    """One client-turn entry in ConversationMetadata.samples.

    sample_index is populated after transforms in MultiTurnDataset.load();
    None before load() is called.
    """

    conversation_id: str
    turn: int
    sample_index: int | None = None


@dataclass
class ConversationMetadata:
    """Bundle of maps/lists consumed by MultiTurnStrategy.

    Produced by MultiTurnDataset._build_metadata() from the post-transform dataframe.
    Keys in the *_by_key dicts are (str(conversation_id), int(turn)).
    Populated by load(); None before load() is called.
    """

    samples: list[ConversationSampleEntry]
    num_conversations: int
    max_turns_per_conv: int
    client_turns_per_conversation: dict[str, int]
    pre_built_messages_by_key: dict[tuple[str, int], list[dict]] = field(
        default_factory=dict
    )
    current_turn_messages_by_key: dict[tuple[str, int], list[dict]] = field(
        default_factory=dict
    )
    system_prompts_by_conv: dict[str, str | None] = field(default_factory=dict)


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
        if not isinstance(result, dict):
            raise InputValidationError(
                f"tool_results[{i}] in conversation {row.get('conversation_id')!r} "
                f"turn {row.get('turn')} must be a dict, got {type(result).__name__}"
            )
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
        conversation_metadata: ConversationMetadata populated by load() (None before).
            Validators run at construction; metadata is built once in load() against
            the post-transform dataframe so pre_built_messages_by_key is always in sync.
    """

    COLUMN_NAMES = ["conversation_id", "turn", "role", "content"]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        enable_salt: bool = False,
        **kwargs,
    ):
        """Initialize multi-turn dataset.

        Args:
            dataframe: DataFrame with conversation data.
            enable_salt: If True, append a per-trajectory hash to the end of
                each trajectory's system prompt (cache-bursting salt; see
                ``examples/09_MultiTurn/docs/EVALUATION.md``).
            **kwargs: Additional arguments passed to Dataset.__init__.

        Raises:
            ValueError: If conversation structure is invalid.
        """
        super().__init__(dataframe, **kwargs)
        assert self.dataframe is not None, "Dataframe must be initialized"
        # Some upstream snapshots (e.g. Workato) ship a single ``dataset_metadata``
        # sentinel record on line 1 carrying license/source attribution. It has
        # no conversation_id and is not a real conversation; drop it before
        # grouping. Real flat rows always have a conversation_id.
        if self.dataframe["conversation_id"].isna().any():
            self.dataframe = self.dataframe.dropna(
                subset=["conversation_id"]
            ).reset_index(drop=True)
        self._enable_salt = enable_salt
        self._conv_groups = dict(
            list(self.dataframe.groupby("conversation_id", sort=False, dropna=False))
        )
        self._validate_conversation_grouping()
        self._validate_conversation_structure()
        self._validate_turn_numbering()
        # Populated by load() after transforms; None until then.
        self.conversation_metadata: ConversationMetadata | None = None
        if enable_salt:
            logger.info(
                "MultiTurnDataset cache-bursting salt enabled "
                "(salt = blake2b(conversation_id) truncated to 16 hex chars)"
            )

    def _validate_conversation_grouping(self) -> None:
        """Validate that all rows for each conversation_id appear consecutively in file order.

        Raises:
            InputValidationError: If rows for a conversation_id are interleaved with other conversations.
        """
        assert self.dataframe is not None, "Dataframe must be initialized"
        seen: set[str] = set()
        last_conv: str | None = None
        for row_idx, row in enumerate(self.dataframe.to_dict(orient="records")):
            raw_id = row["conversation_id"]
            if (
                raw_id is None
                or (isinstance(raw_id, float) and pd.isna(raw_id))
                or str(raw_id) in ("", "nan")
            ):
                raise InputValidationError(
                    f"Row {row_idx}: 'conversation_id' must be a non-empty string"
                )
            conv_id = str(raw_id)
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
            prev_assistant_had_tool_calls = False

            for _, row in sorted_group.iterrows():
                role = row["role"]

                if role not in VALID_NEXT.get(state, set()):
                    raise ValueError(
                        f"Conversation {conv_id} has invalid role sequence at turn "
                        f"{row['turn']}: got '{role}' after state '{state}'"
                    )

                if role == "tool":
                    if state == "assistant" and not prev_assistant_had_tool_calls:
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "'tool' row must follow an 'assistant' row that has non-empty 'tool_calls'"
                        )
                    tool_results = row.get("tool_results")
                    if not isinstance(tool_results, list) or len(tool_results) == 0:
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "tool rows must have a non-empty 'tool_results' list"
                        )
                    for res_idx, res in enumerate(tool_results):
                        if not isinstance(res, dict):
                            raise InputValidationError(
                                f"Conversation {conv_id} turn {row['turn']} "
                                f"tool_results[{res_idx}]: must be a dict"
                            )
                        if (
                            not isinstance(res.get("tool_call_id"), str)
                            or not res["tool_call_id"]
                        ):
                            raise InputValidationError(
                                f"Conversation {conv_id} turn {row['turn']} "
                                f"tool_results[{res_idx}]: "
                                "'tool_call_id' must be a non-empty string"
                            )
                        if "content" not in res:
                            raise InputValidationError(
                                f"Conversation {conv_id} turn {row['turn']} "
                                f"tool_results[{res_idx}]: "
                                "'content' field is required"
                            )
                elif role == "assistant":
                    content = row.get("content")
                    is_empty_content = (
                        content is None
                        or (isinstance(content, float) and pd.isna(content))
                        or content == ""
                    )
                    tool_calls = row.get("tool_calls")
                    has_tool_calls = (
                        isinstance(tool_calls, list) and len(tool_calls) > 0
                    )
                    if has_tool_calls:
                        for call_idx, call in enumerate(tool_calls):
                            if not isinstance(call, dict):
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: must be a dict"
                                )
                            if not isinstance(call.get("id"), str) or not call["id"]:
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: "
                                    "missing or empty 'id' (string)"
                                )
                            if call.get("type") != "function":
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: "
                                    "'type' must be 'function'"
                                )
                            fn = call.get("function")
                            if not isinstance(fn, dict):
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: "
                                    "'function' must be a dict"
                                )
                            if not isinstance(fn.get("name"), str) or not fn["name"]:
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: "
                                    "'function.name' must be a non-empty string"
                                )
                            if not isinstance(fn.get("arguments"), str | dict):
                                raise InputValidationError(
                                    f"Conversation {conv_id} turn {row['turn']} "
                                    f"tool_calls[{call_idx}]: "
                                    "'function.arguments' must be a JSON string or dict"
                                )
                    if is_empty_content and not has_tool_calls:
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "assistant rows must have non-empty 'content' or non-empty 'tool_calls'"
                        )
                    if (
                        tool_calls is not None
                        and not (isinstance(tool_calls, float) and pd.isna(tool_calls))
                        and not has_tool_calls
                    ):
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "'tool_calls' field is present but is not a non-empty list; "
                            "omit the field or provide a valid non-empty list"
                        )
                    prev_assistant_had_tool_calls = has_tool_calls
                elif role == "user":
                    content = row.get("content")
                    is_empty_content = (
                        content is None
                        or (isinstance(content, float) and pd.isna(content))
                        or content == ""
                    )
                    if is_empty_content:
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "user rows must have non-empty 'content'"
                        )
                    if state == "assistant" and prev_assistant_had_tool_calls:
                        raise InputValidationError(
                            f"Conversation {conv_id} turn {row['turn']}: "
                            "'user' row cannot follow an assistant row with 'tool_calls'; "
                            "a 'tool' result row is required first"
                        )

                if role != "assistant":
                    prev_assistant_had_tool_calls = False

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

    def _build_metadata(self) -> ConversationMetadata:
        """Build metadata for scheduler (maps sample index to conversation context).

        Pre-computes the complete message list for each client turn so that
        conversation history does not need to be accumulated at runtime.

        Returns:
            ConversationMetadata with samples, counts, and pre-built message maps.
        """
        samples: list[ConversationSampleEntry] = []

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
            # Cache-bursting salt: append a per-trajectory hash to the end of
            # the system prompt so the engine's prefix cache cannot extend
            # past the system boundary across trajectories. Salt is computed
            # once per trajectory and reused on every turn of that trajectory,
            # so within-trajectory prefix caching is preserved.
            if self._enable_salt and system_content:
                system_content = apply_salt(system_content, compute_salt(str(conv_id)))
            system_prompts_by_conv[str(conv_id)] = system_content

            for _, row in client_rows.iterrows():
                t_n = int(row["turn"])

                messages: list[dict] = []
                if system_content:
                    messages.append({"role": "system", "content": system_content})

                # All dataset rows strictly before this client turn (includes
                # assistant rows and prior tool results).
                # ``reasoning_content`` MUST be propagated so prior assistant
                # turns send their thinking back into context — without it,
                # the chat-template-rendered prompt diverges from what the
                # original capture sent, and replay outputs differ from the
                # captured trajectory even at temperature=0.
                prior_rows = sorted_group[sorted_group["turn"] < t_n]
                for _, prior_row in prior_rows.iterrows():
                    msg: dict[str, Any] = {}
                    for key in (
                        "role",
                        "content",
                        "name",
                        "tool_calls",
                        "tool_results",
                        "reasoning_content",
                    ):
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
                    for key in ("role", "content", "name"):
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
                    ConversationSampleEntry(
                        conversation_id=str_conv_id,
                        turn=t_n,
                    )
                )

        return ConversationMetadata(
            samples=samples,
            num_conversations=len(self._conv_groups),
            max_turns_per_conv=max(g["turn"].max() for g in self._conv_groups.values()),
            client_turns_per_conversation=client_turns_per_conv,
            pre_built_messages_by_key=pre_built_messages_by_key,
            current_turn_messages_by_key=current_turn_messages_by_key,
            system_prompts_by_conv=system_prompts_by_conv,
        )

    def load(
        self,
        adapter=None,
        api_type: APIType | None = None,
        model_params: ModelParams | None = None,
        force: bool = False,
    ):
        """Load dataset, apply adapter defaults, and pre-bake client-turn samples.

        Passing ``adapter=`` without ``api_type`` and ``model_params`` raises
        ``NotImplementedError``; use ``load(api_type=..., model_params=...)`` instead.

        After transforms, only client turns (user + tool) are stored in self.data as
        fully assembled sample dicts (with messages attached).
        load_sample() and num_samples() are inherited from the base class.
        """
        if not force and self.data is not None:
            return

        if adapter is not None and (api_type is None or model_params is None):
            raise NotImplementedError(
                "MultiTurnDataset.load(adapter=...) is not supported; "
                "pass api_type=... and model_params=... instead. "
                "Multi-turn datasets cherry-pick AddStaticColumns defaults from "
                "the api_type's transforms because rows lack a 'prompt' column "
                "and the full adapter pipeline (ColumnFilter) does not apply."
            )

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
                df = AddStaticColumns(defaults, overwrite=False)(df)

        # Rebuild conv_groups + metadata from the final post-transform df so
        # pre_built_messages_by_key reflects any transforms applied above.
        self.dataframe = df
        self._conv_groups = dict(
            list(df.groupby("conversation_id", sort=False, dropna=False))
        )
        self.conversation_metadata = self._build_metadata()

        all_rows = df.to_dict(orient="records")

        # Pre-bake: assemble one complete sample dict per client turn.
        # NaN filtering replaces the GENERATION_PARAMS allowlist — any key whose
        # value is float NaN was absent in the original dataset row.
        pre_built = self.conversation_metadata.pre_built_messages_by_key
        client_turn_samples: list[dict[str, Any]] = []
        # Maps (conv_id, turn) → dense sample_index for metadata backfill.
        key_to_sample_index: dict[tuple[str, int], int] = {}

        # Collect per-conversation defaults from the first user row so that
        # fields like model/max_completion_tokens propagate to tool rows.
        # ``tools`` MUST be propagated so every turn sends the tool definitions —
        # SGLang's tool-call parser is gated on the request having a non-empty
        # ``tools`` array, and without it the parser silently doesn't fire and
        # the model's literal tool-call markup leaks into the ``content`` channel.
        _PROPAGATED_KEYS = {
            "model",
            "max_completion_tokens",
            "max_new_tokens",
            "stream",
            "tools",
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

            # Normalize max-tokens across all adapter aliases.
            max_tokens_val = (
                sample.pop("max_new_tokens", None)
                or sample.get("max_completion_tokens")
                or sample.get("max_tokens")
                or 128
            )
            sample["max_new_tokens"] = max_tokens_val
            sample["max_completion_tokens"] = max_tokens_val
            sample["max_tokens"] = max_tokens_val
            if "stream" not in sample:
                sample["stream"] = False

            # Attach pre-built message list (system + history + current turn).
            key = (str(row["conversation_id"]), int(row["turn"]))
            if key not in pre_built:
                logger.warning(
                    "dropping sample missing pre-built messages: key=%s", key
                )
                continue
            sample["messages"] = pre_built[key]

            # Record dense 0-based index before appending (matches load_sample() position).
            key_to_sample_index[key] = len(client_turn_samples)
            client_turn_samples.append(sample)

        # Backfill explicit sample_index into conversation_metadata.samples.
        # Drop entries whose key is absent (truncated turns not in client_turn_samples).
        updated_samples = []
        for s in self.conversation_metadata.samples:
            skey: tuple[str, int] = (str(s.conversation_id), int(s.turn))
            if skey in key_to_sample_index:
                updated_samples.append(
                    replace(s, sample_index=key_to_sample_index[skey])
                )
        self.conversation_metadata.samples = updated_samples

        self.data = client_turn_samples
