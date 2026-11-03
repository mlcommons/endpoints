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
:func:`find_accuracy_breakdown`), the sample-count-weighted cross-component mean
(:func:`samples_weighted_average_accuracy`), and the numeric coercion
(:func:`to_float`). It lives
under ``evaluation`` — the layer that *produces* breakdowns — so ``metrics`` and
``compliance`` can both import it without a cycle.
"""

from __future__ import annotations

from typing import Any

from ..config.schema import DatasetType


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


def samples_weighted_average_accuracy(
    accuracy_scores: list[dict[str, Any]],
) -> float | None:
    """Sample-count-weighted mean of the per-dataset scalar scores.

    Each accuracy component is weighted by its dataset sample count — the
    ``unit_samples`` field, which is ``dataset.num_samples()`` (the number of rows
    in the loaded dataset, e.g. 30 AIME / 198 GPQA / 1055 LiveCodeBench problems),
    NOT ``total_samples`` (``unit_samples × repeats``, the issued attempts). This
    reproduces the MLCommons gpt-oss reference aggregation that feeds the
    submission ``exact_match``::

        overall = Σ(score_d · unit_samples_d) / Σ unit_samples_d

    Because ``score_d = correct_d / (unit_samples_d · repeats_d)``, the numerator
    term equals ``correct_d / repeats_d`` — i.e. each dataset is normalized
    per-repeat before combining, so it contributes in proportion to its unique
    problems regardless of how many repeats it ran. A plain unweighted mean
    over-weights small datasets (AIME's 30 problems counting equally with
    LiveCodeBench's 1055) and does not match the reference; weighting by
    ``total_samples`` (issued attempts) instead yields the reference's *raw*
    accuracy, which is not the submitted score.

    With one component the weight cancels and the result is that dataset's score
    (DeepSeek-R1). The inline perf-scored entry (``dataset_type == "performance"``)
    and any non-numeric score are excluded; exclusion is by the ``dataset_type``
    discriminator, so a dataset legitimately named ``performance`` still counts.
    A component with ``unit_samples`` absent/``None`` falls back to weight ``1.0``
    (legacy artifacts predating the field; runs produced by this tool always record
    it — ``accuracy.py``), while a *present but* non-positive/non-numeric weight is
    treated as corrupt and the component is skipped rather than counted as one
    sample. Returns ``None`` when no component contributes a numeric score.

    Assumes the components share a scale (gpt-oss: fractions ``[0, 1]``;
    DeepSeek-R1: one percentage ``[0, 100]``); it does not homogenize units.
    """
    numerator = 0.0
    denominator = 0.0
    for entry in accuracy_scores:
        if not isinstance(entry, dict):
            continue
        if entry.get("dataset_type") == DatasetType.PERFORMANCE.value:
            continue
        score = entry.get("score")
        if not isinstance(score, int | float) or isinstance(score, bool):
            continue
        weight = entry.get("unit_samples")
        if weight is None:
            # Legacy artifacts predating unit_samples: contribute unweighted.
            weight = 1.0
        elif (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or weight <= 0
        ):
            # Present but non-positive / non-numeric — a corrupt weight; skip the
            # entry rather than invent a sample count that would skew the mean.
            continue
        numerator += float(score) * float(weight)
        denominator += float(weight)
    if denominator == 0:
        return None
    return numerator / denominator


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
