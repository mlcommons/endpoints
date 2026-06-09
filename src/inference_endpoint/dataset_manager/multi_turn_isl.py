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

"""Offline ISL (Input Sequence Length) computation for multi-turn datasets."""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    _normalize_tool_calls_for_template,
)
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset

logger = logging.getLogger(__name__)


def precompute_isl_for_multi_turn(
    dataloader: MultiTurnDataset, tokenizer_name: str
) -> None:
    """Tokenize pre-built message lists and store token counts in each sample.

    Runs apply_chat_template once per client turn so the hot-path IslTrigger
    sync path (len(token_ids)) is used instead of on-the-fly text tokenization.
    Only affects dataset-history turns; live-history turns override 'messages'
    at runtime so the stored input_tokens are stale (acceptable approximation).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.exception(
            "ISL pre-computation: failed to load tokenizer %s; "
            "falling back to text-tokenization at runtime",
            tokenizer_name,
        )
        return
    skipped = 0
    first_failure_logged = False
    for sample in dataloader.data or []:
        messages = sample.get("messages")
        if not messages:
            continue
        try:
            normalized_messages = []
            for msg in messages:
                if msg.get("tool_calls"):
                    msg = {
                        **msg,
                        "tool_calls": _normalize_tool_calls_for_template(
                            msg["tool_calls"]
                        ),
                    }
                normalized_messages.append(msg)
            tools = sample.get("tools")
            raw = tokenizer.apply_chat_template(
                normalized_messages,
                tools=tools if tools else None,
                tokenize=True,
                add_generation_prompt=True,
            )
            # Some tokenizers (e.g. Qwen3 fast tokenizer) return BatchEncoding
            # instead of a plain list; extract .input_ids in that case.
            token_ids: list[int] = raw.input_ids if hasattr(raw, "input_ids") else raw
            sample["input_tokens"] = token_ids
        except Exception:
            if not first_failure_logged:
                logger.exception(
                    "ISL pre-computation: apply_chat_template failed (first failure shown)"
                )
                first_failure_logged = True
            skipped += 1
    if skipped:
        logger.warning(
            "ISL pre-computation: %d turn(s) skipped (apply_chat_template failed)",
            skipped,
        )
    total_with_messages = len([s for s in (dataloader.data or []) if s.get("messages")])
    if total_with_messages > 0 and skipped == total_with_messages:
        logger.warning(
            "ISL precomputation: all %d turn(s) failed apply_chat_template; "
            "ISL metrics will use text-tokenization fallback. "
            "Check tokenizer/template compatibility.",
            total_with_messages,
        )
