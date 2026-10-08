# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read once, parse evidence, and retain field presence rather than input documents."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..results import CheckResult
from .accuracy import AccuracyResult
from .point_config import PointConfig
from .point_summary import PointSummary
from .system import SystemDescription


def supplied_fields(value: object, prefix: str = "") -> frozenset[str]:
    fields = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            fields.add(name)
            fields.update(supplied_fields(child, name))
    return frozenset(fields)


def schema_errors(
    error: ValidationError, rule: str, path: Path
) -> tuple[CheckResult, ...]:
    return tuple(
        CheckResult(
            rule=rule,
            key=item["type"],
            message=f"{'.'.join(map(str, item['loc'])) or path.name}: {item['msg']}",
            path=path,
        )
        for item in error.errors(include_url=False)
    )


def evidence_value(value: object, address: str, default: object = None) -> Any:
    for name in address.split("."):
        if isinstance(value, BaseModel):
            # Submitted aliases take precedence over similarly named computed properties.
            name = next(
                (
                    field
                    for field, info in type(value).model_fields.items()
                    if info.alias == name
                ),
                name,
            )
            value = getattr(value, name, default)
        elif isinstance(value, Mapping):
            value = value.get(name, default)
        else:
            return default
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    if isinstance(value, (list, tuple)):
        return [
            item.model_dump(mode="json", exclude_unset=True)
            if isinstance(item, BaseModel)
            else item
            for item in value
        ]
    return value.value if isinstance(value, Enum) else value


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class ParsedArtifact(Generic[T]):  # noqa: UP046 - supported by the repository type checker
    value: T | None
    fields: frozenset[str]
    errors: tuple[CheckResult, ...]
    present: bool

    @classmethod
    def from_json(
        cls, data: object, model: type[T], path: Path, rule: str
    ) -> "ParsedArtifact[T]":
        fields = supplied_fields(data)
        try:
            return cls(model.model_validate(data), fields, (), True)
        except ValidationError as error:
            return cls(None, fields, schema_errors(error, rule, path), True)

    def get(self, address: str, default: object = None) -> Any:
        return (
            evidence_value(self.value, address, default)
            if address in self.fields
            else default
        )


def read_artifact(  # noqa: UP047 - supported by the repository type checker
    path: Path, model: type[T], rule: str, *, accuracy_scores: bool = False
) -> ParsedArtifact[T]:
    try:
        text = path.read_text(encoding="utf-8")
        data = (
            yaml.safe_load(text)
            if path.suffix in {".yaml", ".yml"}
            else json.loads(text)
        )
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        if accuracy_scores:
            # Result-file checks own failures in the inline accuracy source.
            return ParsedArtifact(None, frozenset(), (), False)
        finding = CheckResult(
            rule=rule,
            key="artifact-unreadable",
            message=f"Cannot parse {path.name}: {error}",
            path=path,
        )
        return ParsedArtifact(None, frozenset(), (finding,), path.exists())
    if accuracy_scores:
        data = data.get("accuracy_scores") if isinstance(data, Mapping) else None
        if data is None or data == {} or data == []:
            return ParsedArtifact(None, frozenset(), (), False)
        data = {"accuracy_scores": data}
    return ParsedArtifact.from_json(data, model, path, rule)


def load_point_config(path: Path) -> tuple[PointConfig | None, list[CheckResult]]:
    parsed = read_artifact(path, PointConfig, "point-config-valid")
    return parsed.value, list(parsed.errors)


def load_result_summary(path: Path) -> tuple[PointSummary | None, list[CheckResult]]:
    parsed = read_artifact(path, PointSummary, "result-file-valid")
    return parsed.value, list(parsed.errors)


def load_system_description(
    path: Path,
) -> tuple[SystemDescription | None, list[CheckResult]]:
    parsed = read_artifact(path, SystemDescription, "system-description-valid")
    return parsed.value, list(parsed.errors)


def load_accuracy_result(path: Path) -> tuple[AccuracyResult | None, list[CheckResult]]:
    parsed = read_artifact(path, AccuracyResult, "accuracy-valid")
    return parsed.value, list(parsed.errors)


def load_accuracy_scores(
    path: Path,
) -> tuple[AccuracyResult | None, list[CheckResult], bool]:
    parsed = read_artifact(path, AccuracyResult, "accuracy-valid", accuracy_scores=True)
    return parsed.value, list(parsed.errors), parsed.present
