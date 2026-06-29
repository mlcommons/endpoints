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

"""BFCL v4 Scorer for function-calling accuracy evaluation.

Uses the bfcl-eval library's ast_checker to score model outputs against
ground truth function calls. Supports single-turn subsets of the BFCL v4
benchmark: non-live, live, and hallucination categories.

Reference: https://gorilla.cs.berkeley.edu/leaderboard.html
"""

import json
import os
from collections import defaultdict
from typing import Any

import msgspec.json
import numpy as np
import pandas as pd

from ..core.record import EventRecord, EventType, SampleEventType
from ..core.types import merge_tool_calls
from ..dataset_manager.dataset import Dataset
from ..dataset_manager.predefined.bfcl_v4 import CATEGORY_MAP, SINGLE_TURN_SUBSETS
from .extractor import Extractor, FunctionCallExtractor
from .scoring import Scorer

try:
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
except ImportError:
    Language = None
    ast_checker = None

_HALLUCINATION_SUBSETS = set(CATEGORY_MAP["hallucination"])

_SUBSET_LANGUAGE_NAMES = {
    "simple_java": "JAVA",
    "simple_javascript": "JAVASCRIPT",
}

_SIMPLE_AST_SUBSETS = [s for s in CATEGORY_MAP["non_live"] if s.startswith("simple_")]

# Per-category aggregation strategy (matching evalscope/BFCL spec).
# - "sample_weighted": weighted mean by subset sample count
# - "hierarchical": group simple_* subsets into one score, then mean with others
# - "unweighted": simple mean of subset scores
_CATEGORY_AGGREGATION: dict[str, str] = {
    "live": "sample_weighted",
    "non_live": "hierarchical",
    "hallucination": "unweighted",
}

# Subsets that are scored per-sample but excluded from category aggregates.
_UNSCORED_SUBSETS = set(SINGLE_TURN_SUBSETS) - {
    s for subsets in CATEGORY_MAP.values() for s in subsets
}

# Tells ast_checker to convert underscores back to dots in function names.
# The "-FC" suffix triggers this flag inside the checker.
_AST_CHECKER_MODEL_NAME = "gpt-4o-2024-11-20-FC"


class BFCLv4Scorer(Scorer, scorer_id="bfcl_v4"):
    """Scorer for BFCL v4 function-calling benchmark.

    Evaluates model outputs against ground truth function calls using
    bfcl-eval's ast_checker for AST-based comparison (non-live and live
    subsets), and tool-call presence checks for hallucination subsets.

    The scorer produces per-subset accuracy, category aggregates, and
    a weighted overall score for the single-turn categories.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] = FunctionCallExtractor,
        ground_truth_column: str | None = "ground_truth",
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column,
        )
        if ast_checker is None:
            raise ImportError(
                "bfcl-eval is required for BFCL v4 scoring. "
                "Install with: pip install inference-endpoint[bfcl]"
            )

    def get_outputs(self) -> pd.DataFrame:
        """Read COMPLETE events, preferring structured tool_calls for scoring.

        When the model returns structured tool calls, score against the
        serialized tool_calls directly rather than str(TextModelOutput): the
        latter prepends any prose preamble the model emitted alongside the call
        (e.g. "Sure, I'll do that.\\n[{...}]"), which is not valid JSON and
        defeats the function-call parser. The function call is the answer here;
        the prose is chatter. Plain-text responses (no tool calls, e.g.
        hallucination refusals) fall back to the full string.
        """
        events_log_path = self.report_dir / "events.jsonl"
        if not events_log_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_log_path}")

        decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
        outputs: list[dict[str, str]] = []
        with events_log_path.open("r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = decoder.decode(stripped)
                if record.event_type != SampleEventType.COMPLETE:
                    continue
                data = record.data
                tool_calls = getattr(data, "tool_calls", None)
                # Streaming responses store tool_calls as raw per-delta chunks
                # (list-of-lists with fragmented `arguments`); merge_tool_calls
                # reassembles them into complete [{"function": {...}}] objects.
                # Non-streaming tool_calls are already complete and pass through
                # unchanged, so this is safe for both paths.
                merged_tool_calls = merge_tool_calls(tool_calls)
                if merged_tool_calls:
                    output_text = msgspec.json.encode(list(merged_tool_calls)).decode()
                else:
                    output_text = str(data) if data is not None else ""
                outputs.append(
                    {"sample_uuid": record.sample_uuid, "output": output_text}
                )
        return pd.DataFrame(outputs)

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        """Score a single function-calling sample using ast_checker.

        This handles non-hallucination subsets only. Hallucination scoring
        is handled separately in score() via _score_hallucination().
        """
        return self._score_ast(
            value, ground_truth, func_description=None, subset="simple"
        )

    def _score_hallucination(self, raw_output: str) -> float:
        """Score a hallucination sample based on native structured tool_calls.

        Evalscope checks message.tool_calls (structured) -- NOT text content.
        The adapter serializes native tool_calls as JSON with a "function" key
        per item: [{"id":"...","function":{"name":"...","arguments":"..."}}].
        If the raw output doesn't parse as this format, the model returned
        plain text (no structured tool call) which is correct for hallucination.

        Known limitation: if a model outputs text content that happens to be
        valid JSON matching the adapter's tool_calls format, this would
        incorrectly count as "made a tool call." In practice this is rare
        since hallucination prompts elicit refusal text, not JSON.
        """
        has_native_tool_calls = (
            FunctionCallExtractor._try_parse_tool_calls_json(raw_output) is not None
        )
        return 1.0 if not has_native_tool_calls else 0.0

    def _score_ast(
        self,
        value: str,
        ground_truth: str,
        func_description: list[dict] | None = None,
        subset: str = "simple",
    ) -> float:
        """Score a sample using bfcl-eval's AST-based checker."""
        try:
            model_output = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            model_output = []

        has_calls = bool(model_output)

        try:
            expected = (
                json.loads(ground_truth)
                if isinstance(ground_truth, str)
                else ground_truth
            )
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not expected or expected in ({}, []):
            return 1.0 if not has_calls else 0.0

        # Convert extractor format [{"name": "f", "arguments": {...}}]
        # to ast_checker format [{"f": {...}}]
        bfcl_output = []
        for call in model_output if isinstance(model_output, list) else []:
            if not isinstance(call, dict) or "name" not in call:
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            bfcl_output.append({call["name"]: args})

        lang_name = _SUBSET_LANGUAGE_NAMES.get(subset, "PYTHON")
        language = Language[lang_name]

        result = ast_checker(
            func_description=func_description or [],
            model_output=bfcl_output,
            possible_answer=expected,
            language=language,
            test_category=subset,
            model_name=_AST_CHECKER_MODEL_NAME,
        )
        return 1.0 if result["valid"] else 0.0

    def score(  # type: ignore[override]
        self,
    ) -> tuple[dict[str, Any] | float | None, int]:
        """Score all samples and return per-category results.

        Widens the base ``Scorer.score`` return: the first element is a
        per-subset results dict rather than a single float. The accuracy
        consumer in ``commands/benchmark/execute.py`` stores it verbatim, so
        the richer payload is preserved end-to-end.

        Returns:
            (results_dict, n_repeats) where results_dict contains per-subset
            accuracy, category aggregates, and a weighted overall score.
        """
        df = self.get_outputs()

        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]
        df = df.apply(self.match_sample_index, axis=1)

        if self.extractor is not None:
            df["extracted"] = df["output"].apply(
                lambda x: self.extractor.extract(x, default="[]")
            )
        else:
            df["extracted"] = df["output"]

        order = df["sample_index"].to_numpy().astype(int)
        assert self.dataset.dataframe is not None

        ground_truths = self.dataset.dataframe[self.ground_truth_column].to_numpy()[
            order
        ]
        subsets = self.dataset.dataframe["subset"].to_numpy()[order]

        func_descriptions_raw = None
        if "func_description" in self.dataset.dataframe.columns:
            func_descriptions_raw = self.dataset.dataframe[
                "func_description"
            ].to_numpy()[order]

        # Deserialize func_descriptions once outside the loop
        func_descriptions: list[list[dict] | None] | None = None
        if func_descriptions_raw is not None:
            # A NaN/None cell (partial dataset or parquet round-trip) is not a
            # JSON string; treat it as "no description" rather than aborting the
            # whole scoring pass on json.loads(nan).
            func_descriptions = [
                json.loads(fd) if isinstance(fd, str) else None
                for fd in func_descriptions_raw
            ]

        scores_by_subset: dict[str, list[float]] = defaultdict(list)
        all_scores: list[float] = []

        raw_outputs = df["output"].to_numpy()
        extracted = df["extracted"].to_numpy()

        for i in range(len(df)):
            subset = subsets[i]

            if subset in _HALLUCINATION_SUBSETS:
                s = self._score_hallucination(raw_outputs[i])
            else:
                func_desc = func_descriptions[i] if func_descriptions else None
                s = self._score_ast(
                    extracted[i],
                    ground_truths[i],
                    func_description=func_desc,
                    subset=subset,
                )

            scores_by_subset[subset].append(s)
            all_scores.append(s)

        if not all_scores:
            # No samples matched (e.g. empty/filtered events log). np.mean([])
            # would make overall_accuracy the literal string "nan"; emit an
            # explicit zero result instead so results.json is well-formed.
            return {
                "overall_accuracy": "0.00",
                "normalized_single_turn_score": "0.00",
                "category_scores": {},
                "subset_scores": {},
                "unscored_subsets": {},
                "total_samples": 0,
            }, 1

        subset_results = {
            name: float(np.mean(scores)) for name, scores in scores_by_subset.items()
        }

        category_results: dict[str, float] = {}
        for category, category_subsets in CATEGORY_MAP.items():
            present = [s for s in category_subsets if s in subset_results]
            if not present:
                continue

            strategy = _CATEGORY_AGGREGATION.get(category, "unweighted")

            if strategy == "sample_weighted":
                total = sum(len(scores_by_subset[s]) for s in present)
                category_results[category] = (
                    sum(subset_results[s] * len(scores_by_subset[s]) for s in present)
                    / total
                )
            elif strategy == "hierarchical":
                simple_ast = [
                    subset_results[s]
                    for s in _SIMPLE_AST_SUBSETS
                    if s in subset_results
                ]
                top_level = ([float(np.mean(simple_ast))] if simple_ast else []) + [
                    subset_results[s] for s in present if s not in _SIMPLE_AST_SUBSETS
                ]
                if top_level:
                    category_results[category] = float(np.mean(top_level))
            else:
                cat_scores = [subset_results[s] for s in present]
                category_results[category] = float(np.mean(cat_scores))

        # BFCL v4 single-turn categories (non_live, live, hallucination) are equally weighted.
        normalized_score = (
            float(np.mean(list(category_results.values()))) if category_results else 0.0
        )

        n_repeats = (
            max(1, len(all_scores) // self.dataset.num_samples()) if all_scores else 1
        )

        unscored_subsets = {
            s: f"{subset_results[s] * 100:.2f}"
            for s in _UNSCORED_SUBSETS
            if s in subset_results
        }

        results = {
            "overall_accuracy": f"{float(np.mean(all_scores)) * 100:.2f}",
            "normalized_single_turn_score": f"{normalized_score * 100:.2f}",
            "category_scores": {
                k: f"{v * 100:.2f}" for k, v in category_results.items()
            },
            "subset_scores": {k: f"{v * 100:.2f}" for k, v in subset_results.items()},
            "unscored_subsets": unscored_subsets,
            "total_samples": len(all_scores),
        }

        return results, n_repeats
