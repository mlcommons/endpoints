# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Native dataset accuracy entries and metric normalization."""

from __future__ import annotations

import math
from typing import cast

from pydantic import JsonValue, RootModel, model_validator


class AccuracyResult(RootModel[dict[str, dict[str, JsonValue]]]):
    """Parsed dataset accuracy entries."""

    @model_validator(mode="before")
    @classmethod
    def _read_native_report(cls, data: object) -> object:
        """Accept native reports and index their entries without changing the file."""
        if isinstance(data, dict) and "accuracy_scores" in data:
            data = data["accuracy_scores"]
            if not isinstance(data, (dict, list)):
                raise ValueError(
                    "accuracy_scores must be a mapping of datasets or a list of entries"
                )
        if not isinstance(data, list):
            return data
        indexed: dict[str, dict[str, JsonValue]] = {}
        for entry in data:
            if not isinstance(entry, dict):
                raise ValueError("each accuracy_scores entry must be an object")
            name = entry.get("dataset_name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "each accuracy_scores entry needs a non-empty dataset_name"
                )
            if name in indexed:
                raise ValueError(f"dataset {name!r} appears more than once")
            if "score" not in entry:
                raise ValueError(f"dataset {name!r} has no score")
            normalized = dict(entry)
            for native, canonical in (
                ("unit_samples", "num_samples"),
                ("num_repeats", "n_repeats"),
            ):
                if native in entry:
                    if canonical in entry and entry[canonical] != entry[native]:
                        raise ValueError(
                            f"dataset {name!r} gives {native} and {canonical} different values"
                        )
                    normalized[canonical] = entry[native]
            indexed[name] = normalized
        return indexed

    _META_KEYS = frozenset(
        {
            "dataset_name",
            "num_samples",
            "status",
            "extractor",
            "ground_truth_column",
            "n_repeats",
            "complete",
            "unit_samples",
            "total_samples",
            "num_repeats",
            "duration_s",
            "dataset_type",
            "response_counts",
            "extras",
            "eval_method",
            "evaluated_instance_count",
        }
    )

    @classmethod
    def _metric_values(cls, entry: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if "score" in entry:
            score = entry["score"]
            return score if isinstance(score, dict) else {"score": score}
        return {
            name: value for name, value in entry.items() if name not in cls._META_KEYS
        }

    @model_validator(mode="after")
    def _validate_supplied_scores(self) -> AccuracyResult:
        for dataset, entry in self.root.items():
            scores = self._metric_values(entry)
            if "score" in entry and not scores:
                raise ValueError(f"dataset {dataset!r} supplies an empty score mapping")
            normalized: dict[str, JsonValue] = {}
            for metric, value in scores.items():
                message = f"dataset {dataset!r} metric {metric!r} must be a finite numeric score"
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError(message)
                try:
                    number = float(value)
                except (ValueError, OverflowError) as error:
                    raise ValueError(message) from error
                if not math.isfinite(number):
                    raise ValueError(message)
                normalized[metric] = number
            if "score" in entry:
                entry["score"] = (
                    normalized
                    if isinstance(entry["score"], dict)
                    else normalized["score"]
                )
            else:
                entry.update(normalized)
        return self

    def metric_scores(self) -> dict[str, dict[str, float]]:
        """Return validated numeric metrics indexed by dataset."""
        return {
            dataset: {
                metric: cast(float, value)
                for metric, value in self._metric_values(entry).items()
            }
            for dataset, entry in self.root.items()
            if self._metric_values(entry)
        }
