# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection, exclusion, dependency and override behavior."""

import shutil

import pytest
import yaml

from inference_endpoint.validation import (
    Conditions,
    Context,
    Decision,
    EvidenceKey,
    bundled_policy_path,
    load_policy,
    plan_checks,
)
from inference_endpoint.validation.types import Declaration, Match, OfflineMode

pytestmark = pytest.mark.unit


@pytest.fixture
def policy():
    return load_policy(bundled_policy_path())


def point(**changes):
    return Context.model_validate(
        {
            "id": "r16",
            "scope": "point",
            "model_id": "kimi-k3",
            "load_pattern": "agentic_inference",
            "offline": "none",
            "division": "Standardized",
            "declared": [],
            "speculative_decoding": False,
            "available": list(EvidenceKey),
            "invalid": [],
            **changes,
        }
    )


def decisions(policy, context):
    return {check.rule.id: check for check in plan_checks(policy, [context]).checks}


def test_patterns_select_model_family_checks(policy):
    agentic = decisions(policy, point())
    assert agentic["trajectories-to-issue"].decision is Decision.READY
    assert agentic["approved-spec-decode-head"].decision is Decision.EXCLUDED
    single = decisions(
        policy, point(model_id="llama3_1-8b", load_pattern="concurrency")
    )
    assert single["trajectories-to-issue"].decision is Decision.EXCLUDED
    assert single["point-config-valid"].decision is Decision.READY
    curve = Context.model_validate(
        {
            "id": "llama3_1-8b",
            "scope": "curve",
            "model_id": "llama3_1-8b",
            "load_pattern": "concurrency",
            "available": ["computed_regions"],
        }
    )
    assert decisions(policy, curve)["accuracy-gate"].decision is Decision.READY
    assert (
        decisions(policy, curve)["agentic-accuracy-inline"].decision
        is Decision.EXCLUDED
    )


def test_exclusion_precedes_missing_dependencies(policy):
    checks = decisions(
        policy, point(offline="dedicated", load_pattern="max_throughput", available=[])
    )
    assert checks["concurrency-in-range"].decision is Decision.EXCLUDED
    assert checks["load-pattern"].decision is Decision.EXCLUDED
    assert checks["trajectories-to-issue"].decision is Decision.EXCLUDED


def test_unknown_classification_blocks_without_hiding_unconditional_checks(policy):
    checks = decisions(policy, point(load_pattern=None, offline=None))
    assert checks["trajectories-to-issue"].decision is Decision.BLOCKED
    assert checks["concurrency-in-range"].decision is Decision.BLOCKED
    assert checks["result-file-valid"].decision is Decision.READY


def test_invalid_config_does_not_hide_summary_schema_checks(policy):
    checks = decisions(
        policy,
        point(
            available=[
                key for key in EvidenceKey if key is not EvidenceKey.POINT_CONFIG
            ],
            invalid=[EvidenceKey.POINT_CONFIG],
        ),
    )
    assert checks["result-file-valid"].decision is Decision.READY
    assert checks["point-rules-skipped"].decision is Decision.READY
    assert checks["trajectories-to-issue"].decision is Decision.BLOCKED
    assert checks["trajectories-to-issue"].missing == {EvidenceKey.POINT_CONFIG}


def test_missing_seed_catalog_blocks_instead_of_deselecting(policy):
    checks = decisions(policy, point(available=[EvidenceKey.POINT_CONFIG]))
    # Catalog dependency is independent of the artifact's declared seed values.
    assert checks["seed-set-membership"].decision is Decision.BLOCKED
    assert EvidenceKey.SEED_CATALOG in checks["seed-set-membership"].missing


def test_optional_warmup_and_power_are_selected_by_declared_facts(policy):
    no_warmup = decisions(policy, point())
    assert no_warmup["warmup-logs-retained"].decision is Decision.EXCLUDED
    warmup = decisions(policy, point(declared=[Declaration.WARMUP]))
    assert warmup["warmup-logs-retained"].decision is Decision.READY
    for declared, expected in (
        ([], Decision.EXCLUDED),
        (["power_descriptor"], Decision.READY),
    ):
        context = Context.model_validate(
            {
                "id": "sys",
                "scope": "system",
                "division": "RDI",
                "declared": declared,
            }
        )
        assert decisions(policy, context)["power-descriptor"].decision is expected


def test_curve_member_filter_does_not_exclude_the_whole_curve(policy):
    included = point(id="r16")
    offline = point(id="r4096", offline="dedicated", load_pattern="max_throughput")
    curve = Context.model_validate(
        {
            "id": "kimi",
            "scope": "curve",
            "model_id": "kimi-k3",
            "members": [included, offline],
        }
    )
    check = decisions(policy, curve)["benchmark-type-consistency"]
    assert check.decision is Decision.READY
    assert check.selected_members == ("r16",)
    unknown = point(id="broken", offline=None)
    curve = curve.model_copy(update={"members": (included, unknown)})
    assert (
        decisions(policy, curve)["benchmark-type-consistency"].decision
        is Decision.BLOCKED
    )


def test_conditions_and_contexts_reject_contradictory_evidence():
    with pytest.raises(ValueError):
        Conditions.model_validate({"declared": ["warmup"], "not_declared": ["warmup"]})
    with pytest.raises(ValueError):
        point(invalid=["point_config"])
    condition = Conditions.model_validate(
        {"offline": ["none", "elected"], "speculative_decoding": False}
    )
    assert condition(point()) is Match.YES
    assert condition(point(offline=OfflineMode.DEDICATED)) is Match.NO
    assert condition(point(offline=None)) is Match.UNKNOWN


def test_model_override_changes_only_selected_context(tmp_path):
    directory = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), directory)
    path = directory / "catalog.yaml"
    data = yaml.safe_load(path.read_text())
    override = {
        "id": "kimi-count-exception",
        "rule": "trajectories-to-issue",
        "reason": "Fixture exception",
        "applies_to": {"models": ["kimi-k3"]},
        "requirements": {"operator": "equal"},
    }
    data["enrollment"]["overrides"] = [override]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    policy = load_policy(directory)
    check = decisions(policy, point())["trajectories-to-issue"]
    assert check.override == "kimi-count-exception"
    assert check.rule.requirements.operator == "equal"
    other = decisions(policy, point(model_id="deepseek-v4_1-flash"))[
        "trajectories-to-issue"
    ]
    assert other.override is None
    assert other.rule.requirements.operator == "positive_multiple"
    data["enrollment"]["overrides"].append({**override, "id": "overlapping"})
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="Overlapping overrides"):
        plan_checks(load_policy(directory), [point()])


def test_planning_keeps_independent_checks_and_rejects_unknown_models(policy):
    checks = decisions(policy, point(model_id="unapproved", available=[]))
    assert checks["approved-checkpoint"].decision is Decision.BLOCKED
    assert checks["swebench-template"].decision is Decision.BLOCKED
    assert checks["result-file-valid"].decision is Decision.READY
    with pytest.raises(ValueError, match="Duplicate"):
        plan_checks(policy, [point(), point()])
    with pytest.raises(ValueError, match="at least one"):
        plan_checks(policy, [])
