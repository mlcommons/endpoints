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

"""BFCL v4 (Berkeley Function Calling Leaderboard) dataset adapter.

Supports single-turn function-calling evaluation subsets.
Reference: https://gorilla.cs.berkeley.edu/leaderboard.html
"""

import json
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd

from ...dataset import Dataset
from . import presets

logger = getLogger(__name__)

SINGLE_TURN_SUBSETS = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
]

MULTI_TURN_SUBSETS = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

CATEGORY_MAP = {
    "non_live": [
        "simple_python",
        "simple_java",
        "simple_javascript",
        "multiple",
        "parallel",
        "parallel_multiple",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
    ],
    "hallucination": [
        "irrelevance",
        "live_irrelevance",
    ],
    "multi_turn": MULTI_TURN_SUBSETS,
}

# Reverse lookup: subset name -> its category. Used to apply per-category
# sampling rates to the subsets within each category.
SUBSET_TO_CATEGORY = {s: cat for cat, subsets in CATEGORY_MAP.items() for s in subsets}

BFCL_V4_HF_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def _convert_bfcl_functions_to_tools(
    functions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert BFCL function definitions to OpenAI tools format.

    Delegates to bfcl_eval's own convert_to_tool with the GORILLA_TO_OPENAPI
    type mapping, ensuring tool schemas are identical to those used by evalscope.
    """
    try:
        from bfcl_eval.constants.enums import ModelStyle
        from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
        from bfcl_eval.model_handler.utils import convert_to_tool
    except ImportError as e:
        raise ImportError(
            "bfcl-eval is required for the BFCL v4 dataset. "
            "Install with: pip install inference-endpoint[bfcl]"
        ) from e

    return convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)


def _build_messages_from_question(
    question: list[list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Convert BFCL question format to OpenAI messages.

    BFCL questions are nested: [[{"role": "user", "content": "..."}]]
    The outer list represents turns, inner list represents messages in a turn.
    For single-turn, we flatten to a simple message list.

    Note: Some BFCL samples have content=null which is valid per the OpenAI API
    but rejected by llama.cpp and other local servers. We coerce null to "" for
    maximum endpoint compatibility.
    """
    return [
        {"role": msg["role"], "content": msg.get("content") or ""}
        for turn in question
        for msg in turn
    ]


class BFCLv4(
    Dataset,
    dataset_id="bfcl_v4",
):
    """BFCL v4: Berkeley Function Calling Leaderboard Version 4.

    Evaluates LLM function-calling capabilities across multiple categories
    including simple/multiple/parallel function calls, live API scenarios,
    and irrelevance detection.

    Reference: https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
    """

    COLUMN_NAMES = [
        "messages",
        "tools",
        "ground_truth",
        "func_description",
        "subset",
        "sample_id",
    ]

    PRESETS = presets

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        categories: list[str] | None = None,
        subsets: list[str] | None = None,
        sample_pct: float | None = None,
        category_sample_pct: dict[str, float] | None = None,
        subset_floor: int | None = None,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Generate the BFCL v4 dataset for single-turn function-calling evaluation.

        Downloads BFCL v4 data and converts it to the endpoints DataFrame format.
        Each row contains pre-formatted messages and tools ready for the OpenAI adapter.

        Args:
            datasets_dir: Root datasets directory. A subdirectory bfcl_v4/ will be created.
            categories: Category names to include (e.g. ["hallucination", "live"]).
                Valid values: "non_live", "live", "hallucination". If provided, expands
                to the corresponding subsets. Ignored if `subsets` is explicitly given.
            subsets: Explicit list of subset names to include. Overrides `categories`.
                Defaults to all single-turn subsets.
            sample_pct: Percentage (0-100) of samples to use from each subset. Applied
                uniformly to every subset. Acts as the fallback rate for subsets whose
                category is absent from `category_sample_pct`.
            category_sample_pct: Per-category sampling rates, e.g.
                {"non_live": 20, "live": 10, "hallucination": 5}. A subset is sampled at
                its category's rate; categories not listed fall back to `sample_pct`
                (or are kept in full if `sample_pct` is also None).
            subset_floor: Tiny-subset floor. Any subset whose *total* size is <= this
                value is taken in full, bypassing the percentage. Prevents small subsets
                (e.g. live_parallel) from collapsing to one or two noisy samples.
            max_samples: Maximum total samples to include (for smoke testing).
            force: If True, regenerate even if cached file exists.

        Returns:
            DataFrame with columns: messages, tools, ground_truth, subset, sample_id

        Examples:
            # Run only hallucination category
            BFCLv4.generate(datasets_dir, categories=["hallucination"])

            # Run live + hallucination at 25% sampling
            BFCLv4.generate(datasets_dir, categories=["live", "hallucination"], sample_pct=25)

            # Per-category rates with a tiny-subset floor (edge-device <3h budget)
            BFCLv4.generate(
                datasets_dir,
                categories=["non_live", "live", "hallucination"],
                category_sample_pct={"non_live": 20, "live": 10, "hallucination": 5},
                subset_floor=25,
            )
        """
        # Resolve which subsets to include
        if subsets is None:
            if categories is not None:
                subsets = []
                for cat in categories:
                    if cat not in CATEGORY_MAP:
                        raise ValueError(
                            f"Unknown category '{cat}'. "
                            f"Valid categories: {list(CATEGORY_MAP.keys())}"
                        )
                    subsets.extend(CATEGORY_MAP[cat])
            else:
                subsets = SINGLE_TURN_SUBSETS

        # Filter out multi-turn subsets (handled by separate runner)
        multi_turn_subsets_requested = [s for s in subsets if s in MULTI_TURN_SUBSETS]
        subsets = [s for s in subsets if s not in MULTI_TURN_SUBSETS]

        if multi_turn_subsets_requested:
            logger.info(
                "Multi-turn subsets %s will be handled by BFCLMultiTurnRunner "
                "(not included in single-turn DataFrame)",
                multi_turn_subsets_requested,
            )

        if not subsets:
            return pd.DataFrame(columns=cls.COLUMN_NAMES)

        # Load or generate the full dataset (cached as parquet)
        dst_path = datasets_dir / "bfcl_v4" / "bfcl_v4_single_turn.parquet"
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Loading cached dataset from {dst_path}")
            df = pd.read_parquet(dst_path)
            df = cls._deserialize_complex_columns(df)
        else:
            all_rows: list[dict[str, Any]] = []
            for s in SINGLE_TURN_SUBSETS:
                logger.info(f"Loading BFCL v4 subset: {s}")
                rows = cls._load_subset(datasets_dir, s)
                all_rows.extend(rows)
                logger.info(f"  Loaded {len(rows)} samples from {s}")

            if not all_rows:
                raise RuntimeError(
                    "No samples loaded. Ensure BFCL v4 data files are available. "
                    "Install with: pip install bfcl-eval"
                )

            df = pd.DataFrame(all_rows)

            df_to_save = df.copy()
            for col in ("messages", "tools"):
                df_to_save[col] = df_to_save[col].apply(json.dumps)
            df_to_save.to_parquet(dst_path)
            logger.info(f"Cached {len(df)} samples to {dst_path}")

        # Filter to requested subsets
        df = df[df["subset"].isin(subsets)].reset_index(drop=True)
        logger.info(
            f"Selected {len(df)} samples across {len(subsets)} subsets: {subsets}"
        )

        # Apply per-subset sampling (uniform rate, per-category rate, and/or floor).
        # Selection is deterministic (head(n)) so runs are reproducible.
        apply_sampling = (
            sample_pct is not None
            or category_sample_pct is not None
            or subset_floor is not None
        )
        if apply_sampling and len(df) > 0:
            pcts_to_check = [sample_pct] if sample_pct is not None else []
            if category_sample_pct is not None:
                pcts_to_check.extend(category_sample_pct.values())
            for p in pcts_to_check:
                if not (0 < p <= 100):
                    raise ValueError(
                        f"sampling percentage must be in (0, 100], got {p}"
                    )

            sampled_parts = []
            for subset_name, group in df.groupby("subset"):
                total = len(group)
                pct = cls._resolve_subset_pct(
                    subset_name, sample_pct, category_sample_pct
                )
                if subset_floor is not None and total <= subset_floor:
                    n = total
                elif pct is not None:
                    n = max(1, int(total * pct / 100))
                else:
                    n = total
                sampled_parts.append(group.head(n))
                logger.info(f"  {subset_name}: {n}/{total} samples")
            df = pd.concat(sampled_parts).reset_index(drop=True)
            logger.info(f"After sampling: {len(df)} total samples")

        # Final cap
        if max_samples is not None and max_samples < len(df):
            df = df.head(max_samples).reset_index(drop=True)
            logger.info(f"Truncated to {max_samples} samples")

        return df

    @staticmethod
    def _resolve_subset_pct(
        subset_name: str,
        sample_pct: float | None,
        category_sample_pct: dict[str, float] | None,
    ) -> float | None:
        """Resolve the sampling rate for a subset.

        A per-category rate takes precedence; subsets whose category is not listed
        (or have no category) fall back to the uniform `sample_pct`.
        """
        if category_sample_pct is not None:
            category = SUBSET_TO_CATEGORY.get(subset_name)
            if category is not None and category in category_sample_pct:
                return category_sample_pct[category]
        return sample_pct

    @staticmethod
    def _deserialize_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Deserialize JSON-string columns back to Python objects after parquet load."""
        for col in ("messages", "tools"):
            if col in df.columns and len(df) > 0:
                first = df[col].iloc[0]
                if isinstance(first, str):
                    df[col] = df[col].apply(json.loads)
        return df

    @classmethod
    def _load_subset(cls, datasets_dir: Path, subset: str) -> list[dict[str, Any]]:
        """Load a single BFCL v4 subset from the data files.

        Loads both the prompt data and ground truth (from possible_answer/).
        """
        rows: list[dict[str, Any]] = []

        data_file = cls._find_data_file(datasets_dir, subset)
        if data_file is None:
            logger.warning(
                f"Could not find data file for subset '{subset}'. "
                f"Ensure bfcl-eval is installed or data is in {datasets_dir}/bfcl_v4/raw/"
            )
            return rows

        # Load ground truths from possible_answer/ directory
        ground_truths = cls._load_ground_truths(data_file.parent, subset)

        with open(data_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                sample = json.loads(stripped)
                sample_id = sample.get("id", "")

                # Merge ground truth from possible_answer if available
                if sample_id in ground_truths:
                    sample["ground_truth"] = ground_truths[sample_id]

                row = cls._process_sample(sample, subset)
                if row is not None:
                    rows.append(row)

        return rows

    @classmethod
    def _load_ground_truths(cls, data_dir: Path, subset: str) -> dict[str, Any]:
        """Load ground truth answers from the possible_answer/ directory.

        BFCL v4 stores ground truths separately in possible_answer/BFCL_v4_{subset}.json.
        Each line is {"id": "...", "ground_truth": [...]}.
        """
        gt_file = data_dir / "possible_answer" / f"BFCL_v4_{subset}.json"
        if not gt_file.exists():
            return {}

        ground_truths: dict[str, Any] = {}
        with open(gt_file) as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                sample_id = entry.get("id", "")
                if sample_id:
                    ground_truths[sample_id] = entry.get("ground_truth", [])
        return ground_truths

    @classmethod
    def _find_data_file(cls, datasets_dir: Path, subset: str) -> Path | None:
        """Find the BFCL v4 data file for a given subset.

        Search order:
        1. Local raw data directory: datasets_dir/bfcl_v4/raw/BFCL_v4_{subset}.json
        2. bfcl_eval package data directory (installed via pip install bfcl-eval)
        """
        filename = f"BFCL_v4_{subset}.json"

        local_path = datasets_dir / "bfcl_v4" / "raw" / filename
        if local_path.exists():
            return local_path

        try:
            import bfcl_eval
        except ImportError as e:
            raise ImportError(
                "bfcl-eval is required for the BFCL v4 dataset. "
                "Install with: pip install inference-endpoint[bfcl]"
            ) from e

        bfcl_data_dir = Path(bfcl_eval.__file__).parent / "data"
        bfcl_path = bfcl_data_dir / filename
        if bfcl_path.exists():
            return bfcl_path

        return None

    @classmethod
    def _process_sample(
        cls, sample: dict[str, Any], subset: str
    ) -> dict[str, Any] | None:
        """Process a single BFCL sample into the dataset row format.

        Args:
            sample: Raw BFCL JSON sample with 'id', 'question', 'function' fields
            subset: The subset name this sample belongs to

        Returns:
            Dict with messages, tools, ground_truth, subset, sample_id columns,
            or None if the sample cannot be processed.
        """
        sample_id = sample.get("id", "")
        question = sample.get("question", [])
        functions = sample.get("function", [])
        ground_truth = sample.get("ground_truth", [])

        if not question:
            return None

        messages = _build_messages_from_question(question)
        tools = _convert_bfcl_functions_to_tools(functions)

        ground_truth_str = json.dumps(ground_truth) if ground_truth else "[]"

        return {
            "messages": messages,
            "tools": tools,
            "ground_truth": ground_truth_str,
            "func_description": json.dumps(functions),
            "subset": subset,
            "sample_id": sample_id,
        }
