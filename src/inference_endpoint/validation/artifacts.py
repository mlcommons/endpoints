# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact parsing and classification, independent of compliance execution."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import JsonValue, TypeAdapter

from .catalogs.seeds import load_seed_sets
from .conditions import Context
from .evidence import (
    AccuracyResult,
    PointConfig,
    PointSummary,
    SystemDescription,
)
from .evidence.loaders import ParsedArtifact, read_artifact
from .models import Policy
from .planner import PlannedCheck, plan_checks
from .power.models import PowerEvidence
from .schemas.requirements_v1 import RegionBasisRequirements
from .types import Declaration, EvidenceKey, OfflineMode, Scope
from .vocabulary import CheckKind

JsonObject = dict[str, JsonValue]
_JSON_MAPPING = TypeAdapter(JsonObject)


def nested(value: object, address: str, default: object = None) -> Any:
    for part in address.split("."):
        if isinstance(value, Mapping):
            value = value.get(part, default)
        else:
            value = getattr(value, part, default)
        if value is default:
            break
    return value


def read_mapping(path: Path) -> JsonObject:
    try:
        value = (
            yaml.safe_load(path.read_text())
            if path.suffix == ".yaml"
            else json.loads(path.read_text())
        )
        return _JSON_MAPPING.validate_python(value) if isinstance(value, dict) else {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def directories(path: Path) -> list[Path]:
    return (
        sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name)
        if path.is_dir()
        else []
    )


@dataclass(frozen=True)
class PointEvidence:
    config: ParsedArtifact[PointConfig]
    summary: ParsedArtifact[PointSummary]
    system: ParsedArtifact[SystemDescription]
    accuracy: ParsedArtifact[AccuracyResult]


@dataclass
class PointDerived:
    power_kw: float | None = None


@dataclass
class PointArtifacts:
    path: Path
    evidence: PointEvidence
    context: Context
    derived: PointDerived = field(default_factory=PointDerived)

    @classmethod
    def from_evidence(
        cls,
        path: Path,
        evidence: PointEvidence,
        available: frozenset[EvidenceKey] = frozenset(),
    ) -> PointArtifacts:
        config, system = evidence.config, evidence.system
        invalid = (
            frozenset()
            if config.value is not None
            else frozenset({EvidenceKey.POINT_CONFIG})
        )
        available_keys = set(available)
        if config.value is not None:
            available_keys.add(EvidenceKey.POINT_CONFIG)
        declared = frozenset(
            item for item in Declaration if item.value in config.fields
        )
        for item in declared:
            if config.get(item.value) is not None:
                available_keys.add(EvidenceKey(item.value))
        context = Context(
            id=str(path),
            scope=Scope.POINT,
            model_id=path.parent.name,
            load_pattern=config.get("runtime_settings.load_pattern"),
            offline=config.get("offline") or OfflineMode.NONE,
            division=config.get("division") or system.get("division"),
            declared=declared,
            speculative_decoding=(
                config.value.speculative_decoding is not None
                or config.value.spec_decode_head is not None
            )
            if config.value is not None
            else None,
            available=frozenset(available_keys),
            invalid=invalid,
        )
        return cls(path, evidence, context)

    @classmethod
    def from_json(
        cls, path: Path, config: JsonObject, summary: JsonObject, system: JsonObject
    ) -> PointArtifacts:
        return cls.from_evidence(
            path,
            PointEvidence(
                ParsedArtifact.from_json(
                    config, PointConfig, path / "point.yaml", "point-config-valid"
                ),
                ParsedArtifact.from_json(
                    summary,
                    PointSummary,
                    path / "result_summary.json",
                    "result-file-valid",
                ),
                ParsedArtifact.from_json(
                    system,
                    SystemDescription,
                    path / "system_desc.json",
                    "system-description-valid",
                ),
                ParsedArtifact(None, frozenset(), (), False),
            ),
        )

    @property
    def config(self) -> PointConfig | None:
        return self.evidence.config.value

    @property
    def summary(self) -> PointSummary | None:
        return self.evidence.summary.value

    @property
    def system(self) -> SystemDescription | None:
        return self.evidence.system.value

    @property
    def accuracy(self) -> AccuracyResult | None:
        return self.evidence.accuracy.value

    @property
    def curve(self) -> Path:
        return self.path.parent

    def get_config(self, address: str, default: object = None) -> Any:
        return self.evidence.config.get(address, default)

    def get_summary(self, address: str, default: object = None) -> Any:
        return self.evidence.summary.get(address, default)

    def get_system(self, address: str, default: object = None) -> Any:
        return self.evidence.system.get(address, default)

    @property
    def concurrency(self) -> int | None:
        value = self.get_config("concurrency")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return int(self.path.name[1:]) if self.path.name[1:].isdigit() else None

    @property
    def throughput(self) -> float | None:
        duration = self.get_summary("duration_ns")
        tokens = self.get_summary("output_sequence_lengths.total")
        if duration is None or tokens is None or duration <= 0:
            return None
        seconds = duration / 1e9
        return tokens / seconds if seconds > 0 else None


@dataclass
class ArtifactIndex:
    points: dict[str, PointArtifacts] = field(default_factory=dict)
    subjects: list[Context] = field(default_factory=list)
    curves: dict[str, list[str]] = field(default_factory=dict)
    systems: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class CatalogEvidence:
    values: dict[str, object] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class DerivedArtifacts:
    regions: dict[str, dict[str, tuple[int | float, int | float]]] = field(
        default_factory=dict
    )
    region_errors: dict[str, str] = field(default_factory=dict)
    power: dict[str, PowerEvidence] = field(default_factory=dict)


@dataclass
class Artifacts:
    root: Path
    policy: Policy
    index: ArtifactIndex = field(default_factory=ArtifactIndex)
    catalogs: CatalogEvidence = field(default_factory=CatalogEvidence)
    derived: DerivedArtifacts = field(default_factory=DerivedArtifacts)

    def members(self, check: PlannedCheck) -> list[PointArtifacts]:
        subject = check.subject
        if subject.scope is Scope.POINT:
            return [self.index.points[subject.id]]
        ids = self.index.curves.get(
            subject.id, self.index.systems.get(subject.id, list(self.index.points))
        )
        if check.selected_members is not None:
            ids = [id for id in ids if id in check.selected_members]
        return [self.index.points[id] for id in ids]

    def resolve(
        self,
        address: str | None,
        check: PlannedCheck,
        point: PointArtifacts | None = None,
    ) -> Any:
        if not isinstance(address, str):
            return address
        if address == "catalogs.approved_spec_decode_heads":
            return self.catalogs.values.get("approved_spec_decode_heads")
        if address.startswith("catalogs."):
            return nested(self.policy.catalogs, address[9:])
        if address == "enrollment.models":
            return self.policy.models
        if address.startswith("context.cohorts."):
            return nested(self.catalogs.values, address[16:])
        if address.startswith("model."):
            return nested(
                self.policy.catalogs,
                "models." + str(check.subject.model_id) + "." + address[6:],
            )
        path = Path(check.subject.id)
        if address == "submission.root":
            return self.root
        if address == "submission.system_directories":
            return directories(self.root / "results")
        if address == "submission.implementation_directories":
            return directories(self.root / "src")
        if address == "implementation.readme":
            return (
                [p for p in path.iterdir() if p.name.lower() == "readme.md"]
                if path.is_dir()
                else []
            )
        if address == "system.model_directories":
            return directories(path)
        members = self.members(check)
        if address == "curve.point_directories":
            return [p.path for p in members]
        if address == "curve.parsed_points":
            return [p for p in members if p.config is not None]
        if address == "curve.model_directory_name":
            return path.name
        if address == "curve.first_declared_model_name":
            return next(
                (
                    p.get_config("model_name")
                    for p in members
                    if p.get_config("model_name")
                ),
                None,
            )
        if address == "curve.max_supported_concurrency":
            return next(
                (
                    p.get_system("max_supported_concurrency")
                    for p in members
                    if p.get_system("max_supported_concurrency") is not None
                ),
                None,
            )
        if point is None and check.subject.scope is Scope.POINT:
            point = self.index.points[check.subject.id]
        if point is None:
            return [self.resolve(address, check, p) for p in members]
        if address in (
            "point.yaml",
            "point.system_desc.json",
            "point.result_summary.json",
            "result_summary.json",
        ):
            return (
                point.path
                / {
                    "point.yaml": "point.yaml",
                    "point.system_desc.json": "system_desc.json",
                    "point.result_summary.json": "result_summary.json",
                    "result_summary.json": "result_summary.json",
                }[address]
            )
        if address == "point.directory":
            return point.path
        if address == "point.directory_concurrency":
            return int(point.path.name[1:])
        if address == "point.power_kw":
            return point.derived.power_kw
        if address == "point.system_description":
            return (
                point.system.model_dump(mode="json", exclude_unset=True)
                if point.system is not None
                else None
            )
        if address.startswith("point.system_description."):
            return point.get_system(address[25:])
        if address.startswith("point."):
            return point.get_config(address[6:])
        if address == "result_summary.raw_derived_system_tps":
            return point.throughput
        if (
            address == "result_summary.tpot.percentiles.90"
            and point.summary is not None
        ):
            return point.summary.tpot.percentiles.get("90")
        if address.startswith("result_summary."):
            return point.get_summary(address[15:])
        if address.startswith("accuracy."):
            configured = point.get_config(address)
            if configured is not None:
                return configured
            if point.accuracy is None:
                return None
            dataset, _, field = address[9:].partition(".")
            entry: JsonObject = next(
                (
                    entry
                    for name, entry in point.accuracy.root.items()
                    if name.split("::")[0]
                    in (
                        {"inline", "agentic_inference_inline", "agentic_combined"}
                        if dataset == "inline"
                        else {dataset}
                    )
                ),
                {},
            )
            return nested(entry, field)
        return None

    def region(self, point: PointArtifacts) -> str | None:
        regions = self.derived.regions.get(str(point.curve), {})
        return next(
            (
                name
                for name, (lo, hi) in regions.items()
                if point.concurrency is not None and lo <= point.concurrency <= hi
            ),
            None,
        )


def load_artifacts(root: Path, policy: Policy) -> Artifacts:
    artifacts = Artifacts(root, policy)
    try:
        artifacts.catalogs.values["seed_sets"] = load_seed_sets()
    except (ValueError, OSError) as exc:
        artifacts.catalogs.errors["seed_sets"] = str(exc)
    artifacts.catalogs.values["approved_spec_decode_heads"] = policy.catalogs.get(
        "approved_spec_decode_heads"
    )
    for system_dir in directories(root / "results"):
        system_ids = []
        for curve_dir in directories(system_dir):
            point_ids = []
            for path in directories(curve_dir):
                if not (path.name.startswith("r") and path.name[1:].isdigit()):
                    continue
                accuracy_path = path / "accuracy_results.json"
                evidence = PointEvidence(
                    read_artifact(
                        path / "point.yaml", PointConfig, "point-config-valid"
                    ),
                    read_artifact(
                        path / "result_summary.json", PointSummary, "result-file-valid"
                    ),
                    read_artifact(
                        path / "system_desc.json",
                        SystemDescription,
                        "system-description-valid",
                    ),
                    read_artifact(
                        accuracy_path
                        if accuracy_path.exists()
                        else path / "results.json",
                        AccuracyResult,
                        "accuracy-valid",
                        accuracy_scores=not accuracy_path.exists(),
                    ),
                )
                available = set()
                if "seed_sets" in artifacts.catalogs.values:
                    available.add(EvidenceKey.SEED_CATALOG)
                if "approved_spec_decode_heads" in artifacts.catalogs.values:
                    available.add(EvidenceKey.APPROVED_SPEC_DECODE_HEADS)
                point = PointArtifacts.from_evidence(
                    path, evidence, frozenset(available)
                )
                artifacts.index.points[str(path)] = point
                point_ids.append(str(path))
            artifacts.index.curves[str(curve_dir)] = point_ids
            system_ids.extend(point_ids)
            members = tuple(artifacts.index.points[id].context for id in point_ids)
            first = members[0] if members else None
            artifacts.index.subjects.append(
                Context(
                    id=str(curve_dir),
                    scope=Scope.CURVE,
                    model_id=curve_dir.name,
                    division=first.division if first else None,
                    members=members,
                )
            )
            _compute_regions(artifacts, curve_dir, point_ids)
        artifacts.index.systems[str(system_dir)] = system_ids
        first_point = artifacts.index.points[system_ids[0]] if system_ids else None
        declarations = (
            frozenset({Declaration.POWER_DESCRIPTOR})
            if (system_dir / "system_power.json").exists()
            else frozenset()
        )
        artifacts.index.subjects.append(
            Context(
                id=str(system_dir),
                scope=Scope.SYSTEM,
                division=first_point.context.division if first_point else None,
                declared=declarations,
            )
        )
    # Refresh collection members after derived evidence has been computed.
    artifacts.index.subjects = [
        s.model_copy(
            update={
                "members": tuple(
                    artifacts.index.points[id].context
                    for id in artifacts.index.curves[s.id]
                ),
                "available": frozenset({EvidenceKey.COMPUTED_REGIONS})
                if s.id in artifacts.derived.regions
                else frozenset(),
            }
        )
        if s.scope is Scope.CURVE
        else s
        for s in artifacts.index.subjects
    ]
    artifacts.index.subjects.extend(p.context for p in artifacts.index.points.values())
    artifacts.index.subjects.append(Context(id=str(root), scope=Scope.SUBMISSION))
    artifacts.index.subjects.append(Context(id=str(root), scope=Scope.VALIDATOR))
    artifacts.index.subjects.extend(
        Context(id=str(p), scope=Scope.IMPLEMENTATION)
        for p in directories(root / "src")
    )
    return artifacts


def _compute_regions(artifacts: Artifacts, curve: Path, ids: list[str]) -> None:
    all_points = [artifacts.index.points[id] for id in ids]
    catalog = artifacts.policy.catalogs.get("regions", {})
    subject = next(s for s in artifacts.index.subjects if s.id == str(curve))
    plans = [
        check
        for check in plan_checks(artifacts.policy, [subject]).ready
        if check.rule.kind is CheckKind.REGION_BASIS
    ]
    if not plans:
        return
    effective_clamps = {
        check.requirements(RegionBasisRequirements).upper_clamp for check in plans
    }
    if len(effective_clamps) != 1:
        artifacts.derived.region_errors[str(curve)] = (
            "Conflicting region-basis partitions cannot share one set of boundaries"
        )
        return
    clamp = effective_clamps.pop()
    selected_ids = {
        str(point.path) for check in plans for point in artifacts.members(check)
    }
    points = [point for point in all_points if str(point.path) in selected_ids]
    cmax = next(
        (
            p.get_system("max_supported_concurrency")
            for p in points
            if p.get_system("max_supported_concurrency") is not None
        ),
        None,
    )
    concurrencies = [p.concurrency for p in points if p.concurrency is not None]
    if (
        not concurrencies
        or not isinstance(cmax, (int, float))
        or isinstance(cmax, bool)
        or not math.isfinite(cmax)
        or cmax <= clamp
    ):
        return
    cmin = min(min(concurrencies), clamp)
    if cmin < 1:
        return
    partitions = catalog.get("logarithmic_partitions", 3)
    if (
        partitions != 3
        or catalog.get("rounding", "half_even") != "half_even"
        or catalog.get("margin_rounding", "ceiling") != "ceiling"
    ):
        return
    interval = math.log2(cmax - cmin) / partitions
    low = round(cmin + 2**interval)
    medium = round(cmin + 2 ** (2 * interval))
    artifacts.derived.regions[str(curve)] = {
        "low_latency": (1, cmin),
        "low_concurrency": (cmin + 1, low),
        "med_concurrency": (low + 1, medium),
        "high_concurrency": (medium + 1, cmax),
        "margin": (cmax + 1, math.ceil(catalog.get("margin_multiplier", 1.1) * cmax)),
    }
    for point in all_points:
        point.context = point.context.model_copy(
            update={
                "available": point.context.available | {EvidenceKey.COMPUTED_REGIONS}
            }
        )
