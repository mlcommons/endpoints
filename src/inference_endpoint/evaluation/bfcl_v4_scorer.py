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

# Relevance subsets present a genuinely relevant function, so a correct response
# DOES emit a tool call — the mirror image of the hallucination (irrelevance)
# subsets. bfcl-eval ships no possible_answer for live_relevance, so routing it
# through the AST path would default ground_truth to "[]" and award credit for
# NOT calling (inverted). Score it on tool-call presence instead.
_RELEVANCE_SUBSETS = {"live_relevance"}

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
        # Populated by score(); exposed via score_breakdown().
        self._breakdown: dict[str, Any] | None = None

    def get_scoring_outputs(self) -> pd.DataFrame:
        """Text to score, preferring structured tool_calls over the raw output.

        When the model returns structured tool calls, score against the
        serialized tool_calls directly rather than str(TextModelOutput): the
        latter prepends any prose preamble the model emitted alongside the call
        (e.g. "Sure, I'll do that.\\n[{...}]"), which is not valid JSON and
        defeats the function-call parser. The function call is the answer here;
        the prose is chatter. Plain-text responses (no tool calls, e.g.
        hallucination refusals) fall back to the full string.

        OSL / response accounting read the base ``get_raw_outputs()`` instead, so
        they count the full generated text rather than this normalized form.
        """
        outputs: list[dict[str, str]] = []
        for sample_uuid, data in self._iter_complete():
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
            outputs.append({"sample_uuid": sample_uuid, "output": output_text})
        # Columned even when empty, matching the base get_raw_outputs invariant,
        # so callers that index "sample_uuid"/"output" never KeyError.
        return pd.DataFrame(outputs, columns=["sample_uuid", "output"])

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
        made_tool_call = FunctionCallExtractor.has_native_tool_calls(raw_output)
        return 1.0 if not made_tool_call else 0.0

    def _score_relevance(self, raw_output: str) -> float:
        """Score a relevance sample: correct when the model DOES call a tool.

        The mirror image of :meth:`_score_hallucination` — live_relevance
        presents a relevant function, so producing a structured tool call is the
        correct behavior.
        """
        made_tool_call = FunctionCallExtractor.has_native_tool_calls(raw_output)
        return 1.0 if made_tool_call else 0.0

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

    def score(self) -> tuple[float | None, int]:
        """Score all samples; return the scalar overall accuracy and n_repeats.

        Conforms to the base ``Scorer.score`` scalar contract: the first element
        is the overall accuracy as a fraction in ``[0, 1]``. The per-subset and
        per-category breakdown (with percentages and sample counts) is cached and
        exposed separately via :meth:`score_breakdown`.
        """
        df = self.get_scoring_outputs()

        # get_scoring_outputs() returns an empty (columned) frame when there are
        # no COMPLETE events; skip the isin filter in that case — the
        # zero-breakdown branch below handles the empty result.
        if not df.empty:
            valid_uuids = self.sample_index_map.keys()
            df = df[df["sample_uuid"].isin(valid_uuids)]
            # One row per issued uuid: a duplicate COMPLETE for a uuid would
            # otherwise be scored twice and, against the issued denominator
            # below, push a subset/overall mean above 1.0.
            df = df.drop_duplicates(subset="sample_uuid")

        if df.empty:
            # No scorable samples: either the events log had no COMPLETE records
            # or none map to a known sample_uuid. Emit the zero breakdown so the
            # accuracy report is well-formed; the sample_index lookup below would
            # otherwise KeyError on the empty frame. total_samples is 0 (nothing
            # completed) while issued_samples records what was attempted, so an
            # all-missing run reads as "0 of N completed", not "0 of 0".
            self._breakdown = self._zero_breakdown(len(self.sample_index_map))
            return 0.0, 1

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
            elif subset in _RELEVANCE_SUBSETS:
                s = self._score_relevance(raw_outputs[i])
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

        # `total_samples` reports how many samples actually completed (the pre-PR
        # meaning the min_sample_count gate / plot / rollup read); captured here,
        # before the missing-as-failure padding grows all_scores to the issued
        # count. The accuracy denominator (issued) is reported separately as
        # `issued_samples`.
        n_completed = len(all_scores)

        # Missing-as-failure: samples issued but never completed (drain-timeout /
        # crash) are absent from `df`. Score each 0.0 under its issued subset so
        # every accuracy mean below — overall, per-subset, per-category, and the
        # sample-weighted category weights — divides by the issued count, not the
        # surviving subset. Without this a partial run reports accuracy inflated
        # over only the samples that came back. (An all-missing run already
        # returned 0.0 above.) After this, len(all_scores) == issued.
        scored_uuids = set(df["sample_uuid"])
        subset_by_index = self.dataset.dataframe["subset"].to_numpy()
        for missing_uuid, sample_index in self.sample_index_map.items():
            if missing_uuid in scored_uuids:
                continue
            scores_by_subset[subset_by_index[sample_index]].append(0.0)
            all_scores.append(0.0)

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
            s: round(subset_results[s] * 100, 2)
            for s in _UNSCORED_SUBSETS
            if s in subset_results
        }

        overall = float(np.mean(all_scores))
        self._breakdown = {
            "overall_accuracy": round(overall * 100, 2),
            "normalized_single_turn_score": round(normalized_score * 100, 2),
            "category_scores": {
                k: round(v * 100, 2) for k, v in category_results.items()
            },
            "subset_scores": {k: round(v * 100, 2) for k, v in subset_results.items()},
            "unscored_subsets": unscored_subsets,
            "total_samples": n_completed,
            "issued_samples": len(all_scores),
        }

        return overall, n_repeats

    @staticmethod
    def _zero_breakdown(issued: int = 0) -> dict[str, Any]:
        """Well-formed all-zero breakdown for a run with no scorable samples.

        ``total_samples`` is 0 (nothing completed); ``issued_samples`` records how
        many were attempted so an all-missing run is distinguishable from an empty
        one.
        """
        return {
            "overall_accuracy": 0.0,
            "normalized_single_turn_score": 0.0,
            "category_scores": {},
            "subset_scores": {},
            "unscored_subsets": {},
            "total_samples": 0,
            "issued_samples": issued,
        }

    def score_breakdown(self) -> dict[str, Any] | None:
        """Per-subset / per-category accuracy breakdown cached by :meth:`score`.

        Percentages are floats in ``[0, 100]``; ``total_samples`` is the scored
        sample count. Returns ``None`` if :meth:`score` has not run yet.
        """
        return self._breakdown
