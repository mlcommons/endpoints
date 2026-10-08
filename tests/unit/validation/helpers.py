# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for focused validation tests."""

from dataclasses import replace

from pydantic import JsonValue

from inference_endpoint.validation import bundled_policy_path, load_policy
from inference_endpoint.validation.artifacts import PointArtifacts
from inference_endpoint.validation.evidence.loaders import ParsedArtifact
from inference_endpoint.validation.models import Policy


def selected_policy(*ids: str, policy: Policy | None = None) -> Policy:
    policy = policy if policy is not None else load_policy(bundled_policy_path())
    return replace(
        policy, checks=tuple(rule for rule in policy.checks if rule.id in ids)
    )


def update_evidence(point: PointArtifacts, field: str, **updates: JsonValue) -> None:
    artifact = getattr(point.evidence, field)
    assert artifact.value is not None
    data = artifact.value.model_dump(
        mode="json", exclude_unset=True, exclude_computed_fields=True
    )
    data.update(updates)
    parsed = ParsedArtifact.from_json(data, type(artifact.value), point.path, field)
    assert not parsed.errors
    point.evidence = replace(point.evidence, **{field: parsed})
