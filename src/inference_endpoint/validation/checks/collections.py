# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy evaluators for collections."""

from __future__ import annotations

import math
from pathlib import Path

from inference_endpoint.config.schema import LoadPatternType

from ..artifacts import Artifacts, PointArtifacts
from ..evaluator_base import Evaluator
from ..operations import OfflineOperation, PolicyAction, RegionPlacementOperation
from ..outcomes import result
from ..planner import PlannedCheck
from ..results import CheckResult, Severity
from ..schemas.requirements_v1 import (
    Bounds,
    CollectionSizeRequirements,
    CoverageRequirements,
    OfflineRequirements,
    RegionBasisRequirements,
    RegionBoundariesRequirements,
    RegionPlacementRequirements,
)
from ..types import AccuracyKind, OfflineMode
from ..vocabulary import CheckKind


class CollectionSize(Evaluator, kind=CheckKind.COLLECTION_SIZE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(CollectionSizeRequirements)
        count = len(artifacts.resolve(req.source, check) or [])
        minimum = req.minimum if req.minimum is not None else 0
        maximum = req.maximum if req.maximum is not None else math.inf
        members = artifacts.members(check)
        for override in req.overrides if req.overrides is not None else []:
            when = override.when
            agentic = any(
                p.get_config("runtime_settings.load_pattern")
                == LoadPatternType.AGENTIC_INFERENCE
                for p in members
            )
            if (
                when.curve_type == AccuracyKind.SINGLE_TURN
                and not agentic
                and any(p.get_config("offline") == when.has_offline for p in members)
            ):
                minimum = override.minimum
        return [
            result(
                check,
                minimum <= count <= maximum,
                f"Collection has {count} members; allowed [{minimum}, {maximum}]",
            )
        ]


class RegionBasis(Evaluator, kind=CheckKind.REGION_BASIS):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        if check.subject.id in artifacts.derived.region_errors:
            return [
                result(
                    check,
                    False,
                    artifacts.derived.region_errors[check.subject.id],
                    blocked=True,
                )
            ]
        points = artifacts.members(check)
        concurrencies = [p.concurrency for p in points if p.concurrency is not None]
        req = check.requirements(RegionBasisRequirements)
        if not concurrencies:
            return [result(check, False, "No readable concurrency values")]
        finding = result(
            check,
            True,
            f"C_min={min(min(concurrencies), req.upper_clamp)} from {len(concurrencies)} points",
        )
        if (
            len(concurrencies) < len(points)
            and req.on_partial_parse == PolicyAction.WARNING
        ):
            finding = finding.model_copy(update={"severity": Severity.WARNING})
        return [finding]


class RegionBoundaries(Evaluator, kind=CheckKind.REGION_BOUNDARIES):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(RegionBoundariesRequirements)
        cmax = artifacts.resolve("curve.max_supported_concurrency", check)
        points = artifacts.members(check)
        cs = [p.concurrency for p in points if p.concurrency is not None]
        cmin = min(min(cs), req.require_c_min[1]) if cs else None
        valid = (
            cmin is not None
            and req.require_c_min[0] <= cmin <= req.require_c_min[1]
            and isinstance(cmax, (int, float))
            and not isinstance(cmax, bool)
            and cmax > req.require_c_max_greater_than
            and check.subject.id in artifacts.derived.regions
        )
        return [
            result(
                check,
                valid,
                f"Region basis C_min={cmin}, C_max={cmax}; boundaries {artifacts.derived.regions.get(check.subject.id)}",
            )
        ]


class Coverage(Evaluator, kind=CheckKind.COVERAGE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(CoverageRequirements)
        band = req.band
        members = artifacts.members(check)

        def in_band(point: PointArtifacts) -> bool:
            if point.get_config("offline") == OfflineMode.DEDICATED:
                return False
            if isinstance(band, Bounds):
                return (
                    point.concurrency is not None
                    and band.minimum <= point.concurrency <= band.maximum
                )
            actual = artifacts.region(point)
            return (
                actual == band
                or band == "high_concurrency"
                and actual == "margin"
                and req.margin_counts
            )

        count = sum(in_band(p) for p in members)
        return [
            result(
                check,
                count >= req.minimum_count,
                f"{band}: {count} points; requires {req.minimum_count}",
            )
        ]


class RegionPlacement(Evaluator, kind=CheckKind.REGION_PLACEMENT):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(RegionPlacementRequirements)
        point = artifacts.index.points[check.subject.id]
        actual = artifacts.region(point)
        operation = req.operation
        if operation is RegionPlacementOperation.IN_RANGE:
            return [
                result(
                    check,
                    actual is not None and (req.include_margin or actual != "margin"),
                    f"Concurrency {point.concurrency} classified as {actual}",
                )
            ]
        declared = point.get_config("region")
        if (
            declared in req.ignore_declared
            or (actual is None or declared is None)
            and req.on_missing_or_unclassifiable == PolicyAction.SKIP
        ):
            return []
        return [
            result(
                check, declared == actual, f"Declared {declared!r}; computed {actual!r}"
            )
        ]


class Offline(Evaluator, kind=CheckKind.OFFLINE):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(OfflineRequirements)
        points = artifacts.members(check)
        cmax = artifacts.resolve("curve.max_supported_concurrency", check)
        operation = req.operation
        if operation is OfflineOperation.DECLARATION_COUNT:
            elected_concurrency = artifacts.resolve(req.elected_must_equal, check)
            agentic = any(
                p.get_config("runtime_settings.load_pattern")
                == LoadPatternType.AGENTIC_INFERENCE
                for p in points
            )
            offline = [
                p
                for p in points
                if p.get_config("offline")
                in (OfflineMode.ELECTED, OfflineMode.DEDICATED)
            ]
            expected = req.agentic_count if agentic else req.single_turn_count
            findings = [
                result(
                    check,
                    len(offline) == expected,
                    f"{len(offline)} Offline points; expected {expected}",
                )
            ]
            findings.extend(
                result(
                    check,
                    p.concurrency == elected_concurrency,
                    f"Elected concurrency {p.concurrency}; expected {elected_concurrency}",
                    point=p,
                )
                for p in offline
                if p.get_config("offline") == OfflineMode.ELECTED
            )
            return findings
        assert req.throughput_minimum_multiplier is not None
        floor = artifacts.resolve(req.concurrency_floor, check)
        reference = next(
            (
                p
                for p in artifacts.index.points.values()
                if p.curve == Path(check.subject.id)
                and p.concurrency == cmax
                and p.get_config("offline") != OfflineMode.DEDICATED
            ),
            None,
        )
        findings = []
        for point in points:
            if point.get_config("offline") != OfflineMode.DEDICATED:
                continue
            findings.append(
                result(
                    check,
                    floor is not None
                    and point.concurrency is not None
                    and point.concurrency >= floor,
                    f"Offline concurrency {point.concurrency}; floor {floor}",
                    point=point,
                )
            )
            if (
                reference is not None
                and reference.throughput is not None
                and point.throughput is not None
            ):
                minimum = reference.throughput * req.throughput_minimum_multiplier
                findings.append(
                    result(
                        check,
                        point.throughput >= minimum,
                        f"Offline throughput {point.throughput}; minimum {minimum}",
                        point=point,
                    )
                )
        return findings
