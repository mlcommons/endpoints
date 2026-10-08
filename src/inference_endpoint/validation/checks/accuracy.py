# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-driven accuracy checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue

from inference_endpoint.config.schema import LoadPatternType

from ..artifacts import Artifacts, PointArtifacts
from ..evaluator_base import Evaluator
from ..operations import (
    AccuracyMode,
    AccuracyOperation,
    AccuracyPresenceSource,
    FractionConversion,
    PolicyAction,
)
from ..planner import PlannedCheck
from ..results import CheckResult, Severity
from ..schemas.requirements_v1 import (
    AccuracyCoverageRequirements,
    AccuracyGateRequirements,
    AccuracyPresenceRequirements,
)
from ..types import AccuracyKind, OfflineMode
from ..vocabulary import CheckKind
from .helpers import finding, number, unsupported, warning


def accuracy_entries(
    point: PointArtifacts,
) -> tuple[dict[str, dict[str, JsonValue]], dict[str, dict[str, float]]]:
    """Read the parser's normalized dataset mapping and numeric metric scores."""
    if point.accuracy is None:
        return {}, {}
    return point.accuracy.root, point.accuracy.metric_scores()


def dataset_matches(name: str, required: str) -> bool:
    return (
        name == required
        or name.split("::", 1)[0] == required
        or name.startswith(required + "_")
    )


def weighted_mean(pairs: list[tuple[float, object]]) -> float:
    numeric_pairs = [
        (value, weight) for value, weight in pairs if number(weight) and weight > 0
    ]
    if len(numeric_pairs) == len(pairs):
        return sum(value * weight for value, weight in numeric_pairs) / sum(
            weight for _, weight in numeric_pairs
        )
    return sum(value for value, _ in pairs) / len(pairs)


def accuracy_band(artifacts: Artifacts, point: PointArtifacts) -> str | None:
    if (
        point.concurrency is not None
        and 1
        <= point.concurrency
        <= artifacts.policy.catalogs["regions"]["ultra_low_maximum"]
    ):
        return "ultra_low_concurrency"
    return artifacts.region(point)


class AccuracyPresenceEvaluator(Evaluator, kind=CheckKind.ACCURACY_PRESENCE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        accepted = check.requirements(AccuracyPresenceRequirements).accept
        curves = set()
        for point in artifacts.members(check):
            root, _ = accuracy_entries(point)
            standalone = (point.path / "accuracy_results.json").exists()
            if root and (
                (
                    standalone
                    and AccuracyPresenceSource.STANDALONE_ACCURACY_RESULTS in accepted
                )
                or (
                    not standalone
                    and AccuracyPresenceSource.EMBEDDED_NONEMPTY_ACCURACY_SCORES
                    in accepted
                )
            ):
                curves.add(point.curve)
        minimum = check.requirements(AccuracyPresenceRequirements).minimum_models
        return [
            finding(
                check,
                len(curves) >= minimum,
                f"Accuracy results present for {len(curves)} models; requires {minimum}",
            )
        ]


class AccuracyCoverageEvaluator(Evaluator, kind=CheckKind.ACCURACY_COVERAGE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(AccuracyCoverageRequirements)
        bands = artifacts.resolve(params.mandatory_bands, check)
        counts: defaultdict[str | None, int] = defaultdict(int)
        offline = False
        for point in artifacts.members(check):
            root, _ = accuracy_entries(point)
            if not root:
                continue
            band = artifacts.region(point)
            if band != "margin" or params.margin_counts:
                counts[band] += 1
            if (
                point.concurrency is not None
                and 1
                <= point.concurrency
                <= artifacts.policy.catalogs["regions"]["ultra_low_maximum"]
            ):
                counts["ultra_low_concurrency"] += 1
            offline |= point.get_config("offline") in (
                OfflineMode.DEDICATED,
                OfflineMode.ELECTED,
            )
        output = [
            finding(
                check,
                counts[band] >= params.minimum_results_per_band,
                f"Accuracy band {band}: {counts[band]} results; requires {params.minimum_results_per_band}",
            )
            for band in bands
        ]
        patterns = {
            p.get_config("runtime_settings.load_pattern")
            for p in artifacts.members(check)
            if p.get_config("offline") != OfflineMode.DEDICATED
        }
        if (
            params.single_turn_requires_offline_accuracy
            and LoadPatternType.AGENTIC_INFERENCE not in patterns
        ):
            output.append(
                finding(
                    check,
                    offline,
                    "Dedicated offline accuracy result present"
                    if offline
                    else "Dedicated offline accuracy result missing",
                )
            )
        return output


class AccuracyGateEvaluator(Evaluator, kind=CheckKind.ACCURACY_GATE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        params = check.requirements(AccuracyGateRequirements)
        operation = params.operation
        model_catalog = artifacts.resolve(params.models, check)
        model = (
            model_catalog.get(check.subject.model_id, {})
            if isinstance(model_catalog, Mapping)
            else {}
        )
        profile = model.get("accuracy")
        if not isinstance(profile, Mapping):
            return [
                warning(
                    check, f"No published accuracy profile for {check.subject.model_id}"
                )
            ]
        points = artifacts.members(check)
        if operation is AccuracyOperation.AGENTIC_PROFILE_AVAILABLE:
            return [
                finding(
                    check,
                    profile.get("kind") == AccuracyKind.AGENTIC,
                    f"Accuracy profile kind: {profile.get('kind')}",
                    severity=Severity.WARNING,
                )
            ]
        if operation is AccuracyOperation.PER_POINT_RANGE:
            bounds = artifacts.resolve(params.bounds, check)
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                return unsupported(check, "Accuracy range catalog unavailable")
            output = []
            for point in points:
                value = artifacts.resolve(params.source, check, point)
                if not number(value):
                    output.append(
                        warning(
                            check,
                            "Full-run output sequence length is unavailable",
                            point,
                        )
                    )
                    continue
                low, high = bounds
                passed = (
                    low <= value <= high if params.inclusive else low < value < high
                )
                output.append(
                    finding(
                        check,
                        passed,
                        f"Full-run output sequence length {value}; required range [{low}, {high}]",
                        point=point,
                    )
                )
            return output
        if operation in (
            AccuracyOperation.PER_POINT_FRACTION,
            AccuracyOperation.MEAN_OF_BAND_MEANS,
        ):
            return self._agentic(check, artifacts, points, params, operation)
        return self._single_turn(check, artifacts, points, params, profile, operation)

    def _agentic(
        self,
        check: PlannedCheck,
        artifacts: Artifacts,
        points: list[PointArtifacts],
        params: AccuracyGateRequirements,
        operation: AccuracyOperation,
    ) -> list[CheckResult]:
        assert params.dataset is not None
        assert params.input_range is not None
        assert params.output_scale is not None
        minimum = artifacts.resolve(params.minimum, check)
        if not number(minimum):
            return unsupported(check, "Accuracy minimum unavailable")
        output = []
        by_band: defaultdict[str | None, list[float]] = defaultdict(list)
        for point in points:
            root, scores = accuracy_entries(point)
            selected = [
                (name, metrics)
                for name, metrics in scores.items()
                if dataset_matches(name, params.dataset)
            ]
            if not selected:
                if operation is AccuracyOperation.PER_POINT_FRACTION:
                    output.append(
                        warning(
                            check, f"Missing {params.dataset} accuracy result", point
                        )
                    )
                continue
            values = []
            for name, metrics in selected:
                value = metrics.get("score", next(iter(metrics.values()), None))
                lo, hi = params.input_range
                if not number(value) or not lo <= value <= hi:
                    output.append(
                        finding(
                            check,
                            False,
                            f"Accuracy score {value!r} for {name} is outside [{lo}, {hi}]",
                            point=point,
                        )
                    )
                    continue
                values.append(value * params.output_scale)
            if not values:
                continue
            average = sum(values) / len(values)
            if operation is AccuracyOperation.PER_POINT_FRACTION:
                output.append(
                    finding(
                        check,
                        average >= minimum,
                        f"Accuracy {average:g}%; minimum {minimum:g}%",
                        point=point,
                    )
                )
            else:
                by_band[accuracy_band(artifacts, point)].append(average)
        if operation is AccuracyOperation.MEAN_OF_BAND_MEANS:
            assert params.required_band_count is not None
            bands = artifacts.resolve(params.bands, check)
            available = [band for band in bands if by_band[band]]
            if len(available) < params.required_band_count:
                output.append(
                    warning(
                        check,
                        f"SWE-bench accuracy has {len(available)} populated bands; requires {params.required_band_count}",
                    )
                )
            for band in available:
                if len(by_band[band]) > 1:
                    output.append(
                        warning(
                            check,
                            f"Averaging {len(by_band[band])} SWE-bench results in {band}",
                        )
                    )
            if available:
                means = [sum(by_band[band]) / len(by_band[band]) for band in available]
                mean = sum(means) / len(means)
                output.append(
                    finding(
                        check,
                        mean >= minimum,
                        f"SWE-bench mean of band means {mean:g}%; minimum {minimum:g}%",
                    )
                )
            else:
                output.append(warning(check, "No SWE-bench accuracy results available"))
        return output

    def _single_turn(
        self,
        check: PlannedCheck,
        artifacts: Artifacts,
        points: list[PointArtifacts],
        params: AccuracyGateRequirements,
        profile: Mapping[str, Any],
        operation: AccuracyOperation,
    ) -> list[CheckResult]:
        if operation is AccuracyOperation.ISSUED_COUNT:
            assert params.minimum_repeats is not None
        required = artifacts.resolve(params.required_datasets, check)
        datasets = artifacts.resolve(params.dataset_catalog, check)
        metrics = profile.get("metrics", {})
        if not required or not isinstance(datasets, Mapping):
            return unsupported(check, "Required accuracy datasets unavailable")
        if (
            operation is AccuracyOperation.SINGLE_TURN_METRICS
            and params.aggregation != AccuracyMode.SAMPLE_WEIGHTED_MEAN
        ):
            return unsupported(check, "Unsupported accuracy aggregation")
        output = []
        for point in points:
            root, scores = accuracy_entries(point)
            if not root:
                continue
            selected = {
                name: entry
                for name, entry in root.items()
                if any(dataset_matches(name, ds) for ds in required)
            }
            missing = [
                ds
                for ds in required
                if not any(dataset_matches(name, ds) for name in selected)
            ]
            if missing:
                output.append(
                    finding(
                        check,
                        False,
                        f"Required accuracy datasets missing: {', '.join(missing)}",
                        point=point,
                    )
                )
                continue
            if operation is AccuracyOperation.ISSUED_COUNT:
                assert params.minimum_repeats is not None
                for ds in required:
                    threshold = datasets.get(ds, {}).get(params.threshold)
                    if not number(threshold):
                        output.append(
                            finding(
                                check,
                                False,
                                f"Accuracy sample threshold unavailable for {ds}",
                                point=point,
                                blocked=True,
                            )
                        )
                        continue
                    counts = []
                    for name, entry in selected.items():
                        if not dataset_matches(name, ds):
                            continue
                        count = entry.get("num_samples")
                        repeats = entry.get("n_repeats", params.missing_repeats)
                        if number(count) and number(repeats):
                            counts.append(count * max(repeats, params.minimum_repeats))
                    if counts:
                        total = sum(counts)
                        output.append(
                            finding(
                                check,
                                total >= threshold,
                                f"Dataset {ds}: {total:g} issued accuracy samples; requires {threshold:g}",
                                point=point,
                            )
                        )
                continue
            pairs = defaultdict(list)
            for name, entry in selected.items():
                weight = entry.get("num_samples")
                for metric, value in scores.get(name, {}).items():
                    pairs[
                        metric.lower()
                        if params.metric_matching == AccuracyMode.CASE_INSENSITIVE
                        else metric
                    ].append((value, weight))
            aggregate = {
                metric: weighted_mean(values) for metric, values in pairs.items()
            }
            if (
                set(aggregate) == {"score"}
                and len(metrics) == 1
                and params.unnamed_scalar == AccuracyMode.SOLE_METRIC_ONLY_WITH_WARNING
            ):
                name = next(iter(metrics))
                aggregate[name] = aggregate.pop("score")
                output.append(
                    warning(
                        check,
                        f"Unnamed scalar accuracy is interpreted as sole metric {name}",
                        point,
                    )
                )
            for name, bounds in metrics.items():
                score = aggregate.get(
                    name.lower()
                    if params.metric_matching == AccuracyMode.CASE_INSENSITIVE
                    else name
                )
                if score is None:
                    if params.on_missing_metric != PolicyAction.SKIP:
                        output.append(
                            finding(
                                check,
                                False,
                                f"Accuracy metric {name} missing",
                                point=point,
                            )
                        )
                    continue
                reference = bounds.get("reference", bounds.get("minimum_reference"))
                upper_reference = bounds.get(
                    "reference", bounds.get("maximum_reference")
                )
                lower = (
                    reference * bounds["minimum_multiplier"]
                    if reference is not None and "minimum_multiplier" in bounds
                    else None
                )
                upper = (
                    upper_reference * bounds["maximum_multiplier"]
                    if upper_reference is not None and "maximum_multiplier" in bounds
                    else None
                )
                if lower is None and upper is None:
                    output.append(
                        finding(
                            check,
                            False,
                            f"Accuracy bounds unavailable for {name}",
                            point=point,
                            blocked=True,
                        )
                    )
                    continue
                if (
                    params.fraction_conversion
                    == FractionConversion.LEGACY_VALUE_AND_THRESHOLD_HEURISTIC
                    and 0 <= score <= 1
                    and lower is not None
                    and lower > 1
                ):
                    score *= 100
                passed = (
                    number(score)
                    and (lower is None or score >= lower)
                    and (upper is None or score <= upper)
                )
                output.append(
                    finding(
                        check,
                        passed,
                        f"Accuracy metric {name}={score:g}; accepted bounds [{lower}, {upper}]",
                        point=point,
                    )
                )
        return output
