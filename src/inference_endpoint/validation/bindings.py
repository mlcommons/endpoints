# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind model checkpoints and client revisions to published policy approvals."""

import re

from .artifacts import Artifacts
from .evaluator_base import Evaluator
from .operations import BindingMatch
from .outcomes import result
from .planner import PlannedCheck
from .results import CheckResult
from .schemas.requirements_v1 import (
    ArtifactBindingRequirements,
    CatalogIntegrityRequirements,
)
from .vocabulary import CheckKind


class ArtifactBinding(Evaluator, kind=CheckKind.ARTIFACT_BINDING):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        req = check.requirements(ArtifactBindingRequirements)
        matching = req.matching
        findings = []
        for point in artifacts.members(check):
            if matching is BindingMatch.CHECKPOINT:
                addresses = req.sources
                assert addresses is not None
                repository, revision = (
                    artifacts.resolve(address, check, point) for address in addresses
                )
                model = point.context.model_id
                catalog = artifacts.resolve(req.catalog, check)
                approved = catalog.get(model, ()) if catalog is not None else ()
                if not approved:
                    findings.append(
                        result(
                            check,
                            False,
                            f"No approved checkpoint catalog for {model}",
                            point=point,
                            blocked=True,
                        )
                    )
                elif not repository or not revision:
                    findings.append(
                        result(
                            check,
                            False,
                            "Checkpoint repository and revision are required",
                            point=point,
                        )
                    )
                else:
                    matches = any(
                        entry["repository"] == repository
                        and entry["revision"] == revision
                        for entry in approved
                    )
                    findings.append(
                        result(
                            check,
                            matches,
                            "Checkpoint matches model approval"
                            if matches
                            else "Checkpoint is not approved for this model",
                            point=point,
                        )
                    )
                continue
            revision = artifacts.resolve(req.source, check, point)
            if not isinstance(revision, str) or not re.fullmatch(
                r"[0-9a-f]{40}", revision
            ):
                findings.append(
                    result(
                        check,
                        False,
                        "Client revision must be a full lowercase Git SHA-1",
                        point=point,
                    )
                )
                continue
            approved = artifacts.resolve(req.catalog, check)
            if not approved:
                findings.append(
                    result(
                        check,
                        False,
                        "Client revision approval list is empty",
                        point=point,
                        blocked=True,
                    )
                )
                continue
            accepted = revision in approved
            findings.append(
                result(
                    check,
                    accepted,
                    "Client revision is approved"
                    if accepted
                    else "Client revision is not approved",
                    point=point,
                )
            )
        return findings


class CatalogIntegrity(Evaluator, kind=CheckKind.CATALOG_INTEGRITY):
    def __call__(self, check: PlannedCheck, artifacts: Artifacts) -> list[CheckResult]:
        address = check.requirements(CatalogIntegrityRequirements).catalog
        catalog_name = (
            "seed_sets"
            if address == "context.cohorts.seed_sets"
            else "approved_spec_decode_heads"
        )
        problem = artifacts.catalogs.errors.get(catalog_name)
        catalog = artifacts.resolve(address, check)
        if problem is not None or catalog is None:
            return [
                result(
                    check,
                    False,
                    f"Required catalog is unavailable: {problem or address}",
                    blocked=True,
                )
            ]
        return [result(check, True, f"Required catalog is available: {address}")]
