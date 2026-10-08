# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate explicit evidence addresses and references to policy catalog entries."""

import re
from collections.abc import Mapping

from .vocabulary import EvidenceReference


def evidence_references(value: object) -> frozenset[EvidenceReference]:
    found: set[EvidenceReference] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found.update(evidence_references(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(evidence_references(child))
    elif isinstance(value, str) and re.match(
        r"^(point|curve|system|submission|result_summary|accuracy|implementation|model)\.",
        value,
    ):
        found.add(EvidenceReference(value))
    return frozenset(found)


def validate_catalog_references(value: object, catalogs: dict, enrollment: dict):
    if isinstance(value, dict):
        for child in value.values():
            validate_catalog_references(child, catalogs, enrollment)
    elif isinstance(value, list):
        for child in value:
            validate_catalog_references(child, catalogs, enrollment)
    elif isinstance(value, str):
        if value.startswith("catalogs.") or value.startswith("enrollment."):
            root, *components = value.split(".")
            cursor = catalogs if root == "catalogs" else enrollment
            for component in components:
                if not isinstance(cursor, dict) or component not in cursor:
                    raise ValueError(f"Unresolved policy reference {value}")
                cursor = cursor[component]
        elif value.startswith("context.") and value != "context.cohorts.seed_sets":
            raise ValueError(f"Unknown external context reference {value}")


def needs_model(requirements: Mapping) -> bool:
    """Recognize the supported parameter bindings that need model enrollment."""
    return (
        any(
            reference.value.startswith("model.")
            for reference in evidence_references(requirements)
        )
        or requirements.get("models") == "catalogs.models"
        or "constraints_by_model" in requirements
        or requirements.get("matching") == "model_and_repository_and_revision"
    )
