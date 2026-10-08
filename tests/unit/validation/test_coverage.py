# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned source inventory and complete selection coverage for the policy draft."""

import pytest

from inference_endpoint.validation import (
    Context,
    Decision,
    EvidenceKey,
    bundled_policy_path,
    load_policy,
    plan_checks,
)
from inference_endpoint.validation.types import Declaration, Scope

pytestmark = pytest.mark.unit


def test_every_rule_can_be_planned_for_a_supported_context():
    policy = load_policy(bundled_policy_path())
    subjects = []
    for scope in Scope:
        for model, pattern in (
            ("llama3_1-8b", "concurrency"),
            ("kimi-k3", "agentic_inference"),
            ("kimi-k3", "max_throughput"),
        ):
            for offline in ("none", "dedicated", "elected"):
                subjects.append(
                    Context.model_validate(
                        {
                            "id": f"{model}/{pattern}/{offline}",
                            "scope": scope,
                            "model_id": model,
                            "load_pattern": pattern,
                            "offline": offline,
                            "division": "Standardized",
                            "declared": list(Declaration),
                            "speculative_decoding": True,
                            "available": list(EvidenceKey),
                            "invalid": [],
                        }
                    )
                )
    subjects.append(
        Context.model_validate(
            {
                "id": "bad-config",
                "scope": "point",
                "invalid": ["point_config"],
            }
        )
    )
    plan = plan_checks(policy, subjects)
    ready = {check.rule.id for check in plan.ready}
    assert ready == {rule.id for rule in policy.checks}
    assert any(check.decision is Decision.EXCLUDED for check in plan.checks)
    assert any(check.decision is Decision.BLOCKED for check in plan.checks)


def test_missing_evidence_blocks_every_selected_dependent_rule():
    policy = load_policy(bundled_policy_path())
    context = Context.model_validate(
        {
            "id": "agentic",
            "scope": "point",
            "model_id": "kimi-k3",
            "load_pattern": "agentic_inference",
            "offline": "none",
            "declared": list(Declaration),
            "speculative_decoding": True,
            "invalid": [],
        }
    )
    for check in plan_checks(policy, [context]).checks:
        if check.rule.requires and check.decision is not Decision.EXCLUDED:
            assert check.decision is Decision.BLOCKED
            assert check.missing == check.rule.requires
