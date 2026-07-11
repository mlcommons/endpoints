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

Scorers with a multi-subset result (BFCL, DeepSeek-R1) attach a ``breakdown``
dict to their ``accuracy_scores`` entry via :func:`Scorer.score_breakdown`. The
headline accuracy is the entry's scalar ``score``; the breakdown carries the
per-subset detail (``subset_scores`` / ``total_samples``, percentages in
``[0, 100]``) the entry can't. BFCL additionally keeps its gate metrics
(``overall_accuracy`` / ``normalized_single_turn_score``) in the block for the
compliance layer; DeepSeek-R1 does not duplicate the overall there — it reads
back from the entry's ``score``.

This module owns that contract: the breakdown constructor
(:func:`build_breakdown`), the readers (:func:`find_accuracy_entry` /
:func:`find_accuracy_breakdown`), the cross-component mean
(:func:`average_accuracy`), and the numeric coercion (:func:`to_float`). It lives
under ``evaluation`` — the layer that *produces* breakdowns — so ``metrics`` and
``compliance`` can both import it without a cycle.
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


def find_accuracy_entry(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first ``accuracy_scores`` entry carrying a per-subset breakdown.

    ``accuracy_scores`` is a list of per-dataset entries; a multi-subset scorer
    (BFCL, DeepSeek-R1) attaches a ``breakdown`` dict with ``subset_scores``. The
    entry's scalar ``score`` is the headline accuracy; the breakdown holds only
    the per-subset detail. Recognized by the presence of ``subset_scores`` (every
    breakdown-producing scorer emits it) rather than the overall, which
    DeepSeek-R1 no longer stores in the block.
    """
    accuracy_scores = results.get("accuracy_scores")
    if not isinstance(accuracy_scores, list):
        return None
    for entry in accuracy_scores:
        if not isinstance(entry, dict):
            continue
        block = entry.get("breakdown")
        if isinstance(block, dict) and "subset_scores" in block:
            return entry
    return None


def find_accuracy_breakdown(results: dict[str, Any]) -> dict[str, Any] | None:
    """Return the per-subset ``breakdown`` block of the first entry that has one.

    Thin wrapper over :func:`find_accuracy_entry` for consumers that only need the
    breakdown block (e.g. the compliance gate, which reads BFCL's
    ``overall_accuracy`` / ``normalized_single_turn_score`` from it).
    """
    entry = find_accuracy_entry(results)
    return entry.get("breakdown") if entry is not None else None


def average_accuracy(accuracy_scores: list[dict[str, Any]]) -> float | None:
    """Plain mean of the per-dataset scalar scores across accuracy components.

    One component per accuracy dataset (3 for gpt-oss, 1 for DeepSeek-R1), so the
    result equals the single dataset's score when there is only one. The inline
    ``"performance"`` entry — a scored perf dataset, not an accuracy component —
    and any non-numeric score are excluded. Returns ``None`` when no component has
    a numeric score.
    """
    values = [
        float(entry["score"])
        for entry in accuracy_scores
        if isinstance(entry, dict)
        and entry.get("dataset_name") != "performance"
        and isinstance(entry.get("score"), int | float)
        and not isinstance(entry.get("score"), bool)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def build_breakdown(
    subset_scores: dict[str, float],
    total_samples: int,
    *,
    complete: bool = True,
) -> dict[str, Any]:
    """Build a per-subset breakdown dict.

    ``subset_scores`` values are percentages in ``[0, 100]``. The overall/headline
    accuracy is intentionally *not* stored here — it lives on the accuracy entry's
    scalar ``score`` (see :func:`find_accuracy_entry`), so this block carries only
    the per-subset detail the entry can't.

    ``total_samples`` semantics are producer-defined and not directly comparable
    across scorers: the gpt-oss roll-up uses the summed **unique** problem count,
    while ``LegacyMLPerfDeepSeekR1Scorer`` uses the **evaluated** sample count.
    Callers gating on it (e.g. a min-sample check) should account for this.
    """
    return {
        "subset_scores": {k: round(v, 2) for k, v in subset_scores.items()},
        "total_samples": int(total_samples),
        "complete": complete,
    }
