# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint matching and exact client revision approval."""

from dataclasses import replace

import pytest

from inference_endpoint.validation import bundled_policy_path, load_policy
from inference_endpoint.validation.artifacts import (
    ArtifactIndex,
    Artifacts,
    PointArtifacts,
)
from inference_endpoint.validation.bindings import ArtifactBinding
from inference_endpoint.validation.conditions import Context
from inference_endpoint.validation.models import freeze
from inference_endpoint.validation.planner import PlannedCheck
from inference_endpoint.validation.types import Decision, Scope
from tests.unit.validation.helpers import update_evidence

pytestmark = pytest.mark.unit


def binding_check(tmp_path, rule_id):
    policy = load_policy(bundled_policy_path())
    context = Context(id=str(tmp_path), scope=Scope.POINT, model_id="kimi-k3")
    rule = next(rule for rule in policy.checks if rule.id == rule_id)
    point = PointArtifacts.from_json(
        tmp_path,
        {"concurrency": 1, "runtime_settings": {"runtime": {}}},
        {"n_samples_completed": 1, "duration_ns": 1e9},
        {},
    )
    point.context = context
    artifacts = Artifacts(
        tmp_path, policy, index=ArtifactIndex(points={context.id: point})
    )
    return PlannedCheck(context, rule, Decision.READY, "Applicable"), artifacts, point


def test_checkpoint_requires_exact_model_repository_revision(tmp_path):
    check, artifacts, point = binding_check(tmp_path, "approved-checkpoint")
    entry = artifacts.policy.catalogs["approved_checkpoints"]["kimi-k3"][0]
    update_evidence(point, "config", checkpoint=dict(entry))
    assert all(result.passed for result in ArtifactBinding()(check, artifacts))
    update_evidence(point, "config", checkpoint={**dict(entry), "revision": "0" * 40})
    assert any(not result.passed for result in ArtifactBinding()(check, artifacts))


def test_empty_client_approval_list_cannot_pass(tmp_path):
    check, artifacts, point = binding_check(tmp_path, "endpoints-client-sha")
    update_evidence(point, "summary", git_sha="a" * 40)
    results = ArtifactBinding()(check, artifacts)
    assert results[0].key == "blocked"
    assert not results[0].passed


@pytest.mark.parametrize(
    ("revision", "passed"),
    [("a" * 40, True), ("b" * 40, True), ("c" * 40, False)],
)
def test_client_revision_requires_exact_allowlist_match(tmp_path, revision, passed):
    check, artifacts, point = binding_check(tmp_path, "endpoints-client-sha")
    catalogs = dict(artifacts.policy.catalogs)
    catalogs["approved_client_revisions"] = ["a" * 40, "b" * 40]
    artifacts.policy = replace(artifacts.policy, catalogs=freeze(catalogs))
    update_evidence(point, "summary", git_sha=revision)
    results = ArtifactBinding()(check, artifacts)
    assert results[0].passed is passed
    assert results[0].key != "blocked"


@pytest.mark.parametrize("revision", [None, "abc123", "A" * 40, "g" * 40])
def test_client_revision_requires_full_lowercase_sha(tmp_path, revision):
    check, artifacts, point = binding_check(tmp_path, "endpoints-client-sha")
    update_evidence(point, "summary", git_sha=revision)
    results = ArtifactBinding()(check, artifacts)
    assert not results[0].passed
    assert "full lowercase Git SHA-1" in results[0].message
