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

MAX_ARTIFACT_DEPTH = 100


class ArtifactStructureError(ValueError):
    """An artifact contains a cyclic or excessively nested structure."""


def supplied_fields(value: object, prefix: str = "") -> frozenset[str]:
    fields: set[str] = set()
    ancestors: set[int] = set()
    stack: list[tuple[object, str, int, bool]] = [(value, prefix, 0, False)]
    while stack:
        current, address, depth, leaving = stack.pop()
        if not isinstance(current, (Mapping, list, tuple)):
            continue
        identity = id(current)
        if leaving:
            ancestors.remove(identity)
            continue
        if identity in ancestors:
            raise ArtifactStructureError("Artifact contains a recursive alias")
        if depth > MAX_ARTIFACT_DEPTH:
            raise ArtifactStructureError(
                f"Artifact nesting exceeds {MAX_ARTIFACT_DEPTH} levels"
            )
        ancestors.add(identity)
        stack.append((current, address, depth, True))
        if isinstance(current, Mapping):
            for key, child in current.items():
                name = f"{address}.{key}" if address else str(key)
                fields.add(name)
                stack.append((child, name, depth + 1, False))
        else:
            for child in current:
                stack.append((child, address, depth + 1, False))
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
        try:
            fields = supplied_fields(data)
        except ArtifactStructureError as error:
            finding = CheckResult(
                rule=rule, key="artifact-structure", message=str(error), path=path
            )
            return cls(None, frozenset(), (finding,), True)
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
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        yaml.YAMLError,
        RecursionError,
    ) as error:
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
