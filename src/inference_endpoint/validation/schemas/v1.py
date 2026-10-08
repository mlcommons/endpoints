# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire models and normalization for 2026-10-C1 revision 1."""

import hashlib
from types import MappingProxyType

from pydantic import Field, JsonValue, model_validator

from ..conditions import Conditions
from ..models import Override, Policy, Rule, freeze
from ..references import evidence_references, needs_model, validate_catalog_references
from ..types import (
    EvidenceKey,
    FrozenModel,
    Identifier,
    PolicyFile,
    Release,
    Scope,
    Severity,
)
from ..vocabulary import CheckKind
from .registry import BundleParser
from .requirements_v1 import validate_requirements
from .values_v1 import Catalogs, Diagnostics, GroupDocument


class Metadata(FrozenModel):
    kind: CheckKind
    scope: Scope
    severity: Severity
    applies_to: Conditions | None = None
    unless: Conditions | None = None
    requires: tuple[EvidenceKey, ...] = ()

    @model_validator(mode="after")
    def unique_dependencies(self):
        if len(self.requires) != len(set(self.requires)):
            raise ValueError("Duplicate evidence dependency")
        return self


class OverrideDocument(FrozenModel):
    id: Identifier
    rule: Identifier
    reason: Identifier
    applies_to: Conditions
    enabled: bool = Field(default=True, strict=True)
    requirements: dict[str, JsonValue] = Field(default_factory=dict)


class Enrollment(FrozenModel):
    models: tuple[Identifier, ...] = Field(min_length=1)
    check_groups: tuple[Identifier, ...] = Field(min_length=1)
    overrides: tuple[OverrideDocument, ...] = ()


class CatalogDocument(Release):
    catalogs: Catalogs
    enrollment: Enrollment
    diagnostics: Diagnostics


class ParserV1(BundleParser, version="2026-10-C1", first_revision=1, last_revision=1):
    def __call__(self, documents, digests) -> Policy:
        catalog_data = documents[PolicyFile.CATALOG]
        catalog = CatalogDocument.model_validate(catalog_data)
        enrollment = catalog.enrollment
        catalogs = catalog.catalogs.model_dump(mode="json")
        if len(enrollment.models) != len(set(enrollment.models)):
            raise ValueError("Duplicate enrolled model")
        if not set(enrollment.models) <= catalog.catalogs.models.keys():
            raise ValueError("Enrollment references an unknown model")
        if set(enrollment.check_groups) != {
            "submission",
            "system",
            "curve",
            "point",
            "diagnostics",
        }:
            raise ValueError("Enrollment must include each policy check group")
        if len(enrollment.check_groups) != len(set(enrollment.check_groups)):
            raise ValueError("Duplicate enrolled check group")
        groups: list[tuple[PolicyFile, Diagnostics | GroupDocument]] = [
            (PolicyFile.CATALOG, catalog.diagnostics)
        ]
        for file in PolicyFile:
            if file is PolicyFile.CATALOG:
                continue
            parsed_group = GroupDocument.model_validate(documents[file])
            if parsed_group.scope is not file.scope:
                raise ValueError(f"Wrong scope in {file.value}")
            groups.append((file, parsed_group))
        rules: dict[str, Rule] = {}
        for file, group in groups:
            for id_, definition in group.checks.items():
                if id_ in rules:
                    raise ValueError(f"Duplicate rule ID {id_}")
                metadata = Metadata.model_validate(
                    {
                        "scope": group.scope,
                        "severity": group.severity,
                        **{
                            key: value
                            for key, value in definition.items()
                            if key in Metadata.model_fields
                        },
                    }
                )
                parameters = {
                    key: value
                    for key, value in definition.items()
                    if key not in Metadata.model_fields
                }
                try:
                    requirements = validate_requirements(metadata.kind, parameters)
                except ValueError as error:
                    raise ValueError(f"Rule {id_}: {error}") from error
                validate_catalog_references(
                    parameters, catalog_data["catalogs"], catalog_data["enrollment"]
                )
                references = evidence_references(parameters)
                for condition in (metadata.applies_to, metadata.unless):
                    if (
                        condition
                        and condition.models
                        and not set(condition.models) <= set(enrollment.models)
                    ):
                        raise ValueError(
                            f"Rule {id_} condition references an unenrolled model"
                        )
                rules[id_] = Rule(
                    id_,
                    metadata.kind,
                    metadata.scope,
                    metadata.severity,
                    metadata.applies_to,
                    metadata.unless,
                    frozenset(metadata.requires),
                    requirements,
                    references,
                    file,
                    needs_model(parameters),
                )
        overrides = []
        seen_overrides = set()
        for override in enrollment.overrides:
            if override.id in seen_overrides or override.rule not in rules:
                raise ValueError("Duplicate override ID or unknown override rule")
            seen_overrides.add(override.id)
            rule = rules[override.rule]
            try:
                validate_requirements(
                    rule.kind, {**rule.requirements.wire(), **override.requirements}
                )
            except ValueError as error:
                raise ValueError(
                    f"Override {override.id} for {rule.id}: {error}"
                ) from error
            validate_catalog_references(
                override.requirements,
                catalogs,
                catalog_data["enrollment"],
            )
            evidence_references(override.requirements)
            if override.applies_to.models and not set(
                override.applies_to.models
            ) <= set(enrollment.models):
                raise ValueError("Override references an unenrolled model")
            overrides.append(
                Override(
                    override.id,
                    override.rule,
                    override.reason,
                    override.applies_to,
                    override.enabled,
                    freeze(override.requirements),
                )
            )
        digest = hashlib.sha256(
            "".join(f"{file.value}:{digests[file]}\n" for file in PolicyFile).encode()
        ).hexdigest()
        return Policy(
            Release(version=catalog.version, revision=catalog.revision),
            freeze(catalogs),
            frozenset(enrollment.models),
            tuple(rules.values()),
            tuple(overrides),
            freeze(documents),
            MappingProxyType(dict(digests)),
            digest,
        )
