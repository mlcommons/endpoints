# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable policy definitions independent of file layout and parser versions."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar, overload

from .conditions import Conditions
from .types import EvidenceKey, PolicyFile, Release, Scope, Severity
from .vocabulary import CheckKind, EvidenceReference

_Key = TypeVar("_Key")


@overload
def freeze(  # noqa: UP047 - supported by the repository type checker
    value: Mapping[_Key, Any],
) -> Mapping[_Key, Any]: ...


@overload
def freeze(value: object) -> Any: ...


def freeze(value: object) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(freeze(child) for child in value)
    return value


def thaw(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class Rule:
    id: str
    kind: CheckKind
    scope: Scope
    severity: Severity
    applies_to: Conditions | None
    unless: Conditions | None
    requires: frozenset[EvidenceKey]
    requirements: Mapping[str, Any]
    references: frozenset[EvidenceReference]
    source: PolicyFile
    needs_model: bool = False


@dataclass(frozen=True)
class Override:
    id: str
    rule: str
    reason: str
    applies_to: Conditions
    enabled: bool
    requirements: Mapping[str, Any]


@dataclass(frozen=True)
class Policy:
    release: Release
    catalogs: Mapping[str, Any]
    models: frozenset[str]
    checks: tuple[Rule, ...]
    overrides: tuple[Override, ...]
    documents: Mapping[PolicyFile, Mapping[str, Any]]
    source_digests: Mapping[PolicyFile, str]
    digest: str

    def as_monolith(self) -> dict:
        catalog = thaw(self.documents[PolicyFile.CATALOG])
        groups = {}
        for name in PolicyFile:
            if name is PolicyFile.CATALOG:
                continue
            document = thaw(self.documents[name])
            groups[name.scope.value] = {
                key: value
                for key, value in document.items()
                if key not in {"version", "revision"}
            }
        groups["diagnostics"] = catalog["diagnostics"]
        return {
            "version": self.release.version,
            "revision": self.release.revision,
            "catalogs": catalog["catalogs"],
            "check_groups": groups,
            "enrollment": catalog["enrollment"],
        }
