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

"""Offline ISL (Input Sequence Length) computation for multi-turn datasets.

Run from the repo root to print the ISL distribution for a dataset::

    python scripts/multi_turn_isl.py \\
        --dataset path/to/dataset.jsonl \\
        --tokenizer <model-name-or-path>
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    _normalize_tool_calls_for_template,
)
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _precompute_isl(dataloader: MultiTurnDataset, tokenizer_name: str) -> None:
    samples_with_messages = [s for s in (dataloader.data or []) if s.get("messages")]
    if not samples_with_messages:
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.exception("Failed to load tokenizer %s", tokenizer_name)
        return

    first_failure_lock = threading.Lock()

    def _tokenize_sample(sample: dict) -> list[int] | None:
        try:
            normalized_messages = []
            for msg in sample["messages"]:
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
            return token_ids
        except Exception:
            if first_failure_lock.acquire(blocking=False):
                logger.exception("apply_chat_template failed (first failure shown)")
            return None

    n_workers = os.cpu_count() or 4
    skipped = 0
    with ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="ISLPrecompute"
    ) as pool:
        futures = {
            pool.submit(_tokenize_sample, sample): sample
            for sample in samples_with_messages
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Pre-computing ISL",
            unit="turn",
        ):
            sample = futures[future]
            token_ids = future.result()
            if token_ids is not None:
                sample["input_tokens"] = token_ids
            else:
                skipped += 1

    if skipped:
        logger.warning("%d turn(s) skipped (apply_chat_template failed)", skipped)
    if skipped == len(samples_with_messages):
        logger.warning(
            "All %d turn(s) failed apply_chat_template. "
            "Check tokenizer/template compatibility.",
            len(samples_with_messages),
        )


def _isl_distribution(dataloader: MultiTurnDataset) -> dict[str, float]:
    values = sorted(
        len(s["input_tokens"])
        for s in (dataloader.data or [])
        if s.get("input_tokens") is not None
    )
    if not values:
        raise ValueError(
            "No input_tokens found — tokenization may have failed entirely."
        )
    n = len(values)

    def percentile(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, frac = int(idx), idx % 1
        return values[lo] + frac * (values[lo + 1] - values[lo] if lo + 1 < n else 0)

    return {
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / n,
        "p50": percentile(50),
        "p99": percentile(99),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Compute ISL distribution for a multi-turn dataset."
    )
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file.")
    parser.add_argument(
        "--tokenizer", required=True, help="HuggingFace repo ID or local path."
    )
    args = parser.parse_args()

    ds = MultiTurnDataset(pd.read_json(args.dataset, lines=True))
    ds.load()
    _precompute_isl(ds, args.tokenizer)

    stats = _isl_distribution(ds)
    n = sum(1 for s in (ds.data or []) if s.get("input_tokens") is not None)
    print(f"ISL distribution ({n} turns)")
    print(f"  min  : {stats['min']:.0f}")
    print(f"  mean : {stats['mean']:.1f}")
    print(f"  p50  : {stats['p50']:.0f}")
    print(f"  p99  : {stats['p99']:.0f}")
    print(f"  max  : {stats['max']:.0f}")


if __name__ == "__main__":
    main()
