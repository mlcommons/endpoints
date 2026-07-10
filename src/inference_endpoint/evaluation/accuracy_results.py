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

"""Shared helpers for the accuracy ``breakdown`` block in ``results.json``.

Scorers with a multi-subset result (BFCL, DeepSeek-R1, gpt-oss) attach a
``breakdown`` dict to their ``accuracy_scores`` entry via
:func:`Scorer.score_breakdown`. All of them use the BFCL-shaped keys
(``overall_accuracy`` / ``subset_scores`` / ``total_samples``, percentages in
``[0, 100]``) so the report, plotting, and compliance layers read one shape.

This module owns that contract: the constructor (:func:`build_breakdown`), the
reader (:func:`find_accuracy_breakdown`), and the numeric coercion
(:func:`to_float`). It lives under ``evaluation`` — the layer that *produces*
breakdowns — so ``metrics`` and ``compliance`` can both import it without a
cycle.
"""

from __future__ import annotations

from typing import Any

# Ruleset golden-metric name -> key in the scorer's breakdown block.
ACCURACY_METRIC_KEYS = {
    "bfcl_overall_accuracy": "overall_accuracy",
    "bfcl_normalized_accuracy": "normalized_single_turn_score",
}


def to_float(value: Any) -> float | None:
    """Coerce a metric to float, or None if absent/non-numeric.

    Breakdown metrics are numeric, but older artifacts stored them as formatted
    strings (e.g. ``"86.23"``); coerce defensively before any comparison.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_accuracy_breakdown(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first per-subset breakdown from a run's ``accuracy_scores``.

    ``accuracy_scores`` is a list of per-dataset entries; each entry may carry a
    ``breakdown`` dict (per-subset accuracy + ``total_samples``). Returns the
    first breakdown whose keys include ``overall_accuracy`` — which recognizes
    every BFCL-shaped breakdown (BFCL, DeepSeek-R1).
    """
    accuracy_scores = results.get("accuracy_scores")
    if not isinstance(accuracy_scores, list):
        return None
    for entry in accuracy_scores:
        if not isinstance(entry, dict):
            continue
        block = entry.get("breakdown")
        if isinstance(block, dict) and "overall_accuracy" in block:
            return block
    return None


def build_breakdown(
    overall: float | None,
    subset_scores: dict[str, float],
    total_samples: int,
    *,
    complete: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Build a BFCL-shaped breakdown dict.

    ``overall`` and ``subset_scores`` values are percentages in ``[0, 100]``
    (``overall`` may be ``None`` when no subset was scorable). ``extra`` carries
    scorer-specific detail (e.g. ``per_subset_status``) alongside the shared
    keys.

    ``total_samples`` semantics are producer-defined and not directly comparable
    across scorers: the gpt-oss roll-up uses the summed **unique** problem count,
    while ``LegacyMLPerfDeepSeekR1Scorer`` uses the **evaluated** sample count.
    Callers gating on it (e.g. a min-sample check) should account for this.
    """
    breakdown: dict[str, Any] = {
        "overall_accuracy": round(overall, 2) if overall is not None else None,
        "subset_scores": {k: round(v, 2) for k, v in subset_scores.items()},
        "total_samples": int(total_samples),
        "complete": complete,
    }
    breakdown.update(extra)
    return breakdown
