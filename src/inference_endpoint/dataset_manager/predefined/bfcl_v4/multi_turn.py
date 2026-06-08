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

"""BFCL v4 multi-turn dataset adapter for agentic function-calling evaluation.

Supports multi-turn evaluation subsets where the model must execute function
calls across multiple conversation turns, with execution results fed back
into the conversation history.

Subsets: multi_turn_base, multi_turn_miss_func, multi_turn_miss_param,
multi_turn_long_context.
"""

from logging import getLogger
from typing import Any

from . import _convert_bfcl_functions_to_tools

logger = getLogger(__name__)

MULTI_TURN_SUBSETS = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

MULTI_TURN_CATEGORY_MAP = {
    "multi_turn": MULTI_TURN_SUBSETS,
}

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated the available functions. "
    "You have access to new functions now. "
    "Please re-evaluate the latest user request and "
    "use the appropriate function(s) to handle the request."
)


class BFCLv4MultiTurnEntry:
    """Represents a single BFCL v4 multi-turn test entry ready for execution.

    Each entry contains:
    - turns: list of user messages per turn (from entry["question"])
    - tools: OpenAI-format tool definitions (from entry["function"])
    - initial_config: initial state for simulated classes
    - involved_classes: list of class names for function execution
    - ground_truth: per-turn list of expected function calls
    - holdout_function: tools to add at specific turns (miss_func category)
    """

    __slots__ = (
        "entry_id",
        "subset",
        "turns",
        "tools",
        "raw_functions",
        "initial_config",
        "involved_classes",
        "ground_truth",
        "holdout_function",
        "excluded_function",
    )

    def __init__(
        self,
        entry_id: str,
        subset: str,
        turns: list[list[dict[str, str]]],
        tools: list[dict[str, Any]],
        raw_functions: list[dict[str, Any]],
        initial_config: dict[str, Any],
        involved_classes: list[str],
        ground_truth: list[Any],
        holdout_function: dict[str, list[dict[str, Any]]] | None = None,
        excluded_function: list[str] | None = None,
    ):
        self.entry_id = entry_id
        self.subset = subset
        self.turns = turns
        self.tools = tools
        self.raw_functions = raw_functions
        self.initial_config = initial_config
        self.involved_classes = involved_classes
        self.ground_truth = ground_truth
        self.holdout_function = holdout_function or {}
        self.excluded_function = excluded_function or []

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    def get_turn_messages(self, turn_idx: int) -> list[dict[str, str]]:
        """Get initial user messages for a given turn.

        For miss_func category, if this turn has holdout functions, returns
        the default prompt indicating new functions are available.
        """
        if str(turn_idx) in self.holdout_function:
            return [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                }
            ]
        return self.turns[turn_idx]

    def get_tools_for_turn(self, turn_idx: int) -> list[dict[str, Any]]:
        """Get the tools available at a specific turn.

        For miss_func, holdout functions are added at their designated turn.
        """
        if str(turn_idx) in self.holdout_function:
            added_funcs = self.holdout_function[str(turn_idx)]
            added_tools = _convert_bfcl_functions_to_tools(added_funcs)
            return self.tools + added_tools
        return self.tools


def load_multi_turn_entries(
    subsets: list[str] | None = None,
) -> list[BFCLv4MultiTurnEntry]:
    """Load BFCL v4 multi-turn test entries from the bfcl-eval package.

    Args:
        subsets: List of subset names to load. Defaults to all multi-turn subsets.

    Returns:
        List of BFCLv4MultiTurnEntry objects ready for execution.
    """
    try:
        from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    except ImportError as e:
        raise ImportError(
            "bfcl-eval is required for BFCL v4 multi-turn evaluation. "
            "Install with: pip install inference-endpoint[bfcl]"
        ) from e

    if subsets is None:
        subsets = MULTI_TURN_SUBSETS

    entries: list[BFCLv4MultiTurnEntry] = []

    for subset in subsets:
        if subset not in MULTI_TURN_SUBSETS:
            raise ValueError(
                f"Unknown multi-turn subset '{subset}'. "
                f"Valid subsets: {MULTI_TURN_SUBSETS}"
            )

        raw_entries = load_dataset_entry(
            subset, include_prereq=True, include_language_specific_hint=False
        )
        gt_entries = load_ground_truth_entry(subset)
        gt_map = {g["id"]: g["ground_truth"] for g in gt_entries}

        for raw in raw_entries:
            entry_id = raw["id"]
            tools = _convert_bfcl_functions_to_tools(raw.get("function", []))
            ground_truth = gt_map.get(entry_id, [])

            entry = BFCLv4MultiTurnEntry(
                entry_id=entry_id,
                subset=subset,
                turns=raw["question"],
                tools=tools,
                raw_functions=raw.get("function", []),
                initial_config=raw.get("initial_config", {}),
                involved_classes=raw.get("involved_classes", []),
                ground_truth=ground_truth,
                holdout_function=raw.get("missed_function"),
                excluded_function=raw.get("excluded_function"),
            )
            entries.append(entry)

    logger.info(
        f"Loaded {len(entries)} multi-turn entries across {len(subsets)} subsets: "
        f"{subsets}"
    )
    return entries
