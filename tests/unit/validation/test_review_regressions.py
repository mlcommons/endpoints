# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catalog defaults and member-level policy exception contracts."""

import shutil

import pytest
import yaml

from inference_endpoint.validation import (
    Context,
    Decision,
    bundled_policy_path,
    load_policy,
    plan_checks,
)

pytestmark = pytest.mark.unit


def with_override(tmp_path, rule, conditions):
    directory = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), directory)
    path = directory / "catalog.yaml"
    data = yaml.safe_load(path.read_text())
    data["enrollment"]["overrides"] = [
        {
            "id": "member-exception",
            "rule": rule,
            "reason": "Exception for matching points",
            "applies_to": conditions,
            "enabled": False,
        }
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_policy(directory)


def test_loaded_catalog_includes_parsed_dataset_defaults():
    dataset = load_policy(bundled_policy_path()).catalogs["datasets"]["cnn_dailymail"]
    assert dataset["is_legacy"] is False
    assert dataset["sample_unit"] == "sample"


def test_curve_exception_uses_selected_member_facts(tmp_path):
    policy = with_override(tmp_path, "offline-ordering", {"offline": ["dedicated"]})
    curve = Context(
        id="kimi",
        scope="curve",
        model_id="kimi-k3",
        offline="none",
        members=(Context(id="r4096", scope="point", offline="dedicated"),),
    )
    check = next(
        check
        for check in plan_checks(policy, [curve]).checks
        if check.rule.id == "offline-ordering"
    )
    assert check.decision is Decision.EXCLUDED
    assert check.override == "member-exception"
    assert check.selected_members == ("r4096",)


def test_curve_exception_partitions_members_without_hiding_other_points(tmp_path):
    policy = with_override(
        tmp_path, "benchmark-type-consistency", {"load_pattern": ["agentic_inference"]}
    )
    curve = Context(
        id="mixed",
        scope="curve",
        model_id="kimi-k3",
        members=(
            Context(
                id="a", scope="point", offline="none", load_pattern="agentic_inference"
            ),
            Context(id="b", scope="point", offline="none", load_pattern="concurrency"),
        ),
    )
    checks = [
        check
        for check in plan_checks(policy, [curve]).checks
        if check.rule.id == "benchmark-type-consistency"
    ]
    assert len(checks) == 2
    assert {(check.decision, check.selected_members) for check in checks} == {
        (Decision.EXCLUDED, ("a",)),
        (Decision.READY, ("b",)),
    }


def test_unknown_point_facts_are_not_filled_from_curve_declarations():
    policy = load_policy(bundled_policy_path())
    curve = Context(
        id="curve",
        scope="curve",
        model_id="kimi-k3",
        offline="dedicated",
        members=(Context(id="unknown", scope="point"),),
    )
    check = next(
        check
        for check in plan_checks(policy, [curve]).checks
        if check.rule.id == "benchmark-type-consistency"
    )
    assert check.decision is Decision.BLOCKED


def test_unknown_member_override_blocks_only_its_partition(tmp_path):
    policy = with_override(
        tmp_path, "benchmark-type-consistency", {"load_pattern": ["agentic_inference"]}
    )
    curve = Context(
        id="mixed",
        scope="curve",
        model_id="kimi-k3",
        load_pattern="concurrency",
        members=(
            Context(
                id="known", scope="point", offline="none", load_pattern="concurrency"
            ),
            Context(id="unknown", scope="point", offline="none"),
        ),
    )
    checks = [
        check
        for check in plan_checks(policy, [curve]).checks
        if check.rule.id == "benchmark-type-consistency"
    ]
    assert {(check.decision, check.selected_members) for check in checks} == {
        (Decision.READY, ("known",)),
        (Decision.BLOCKED, ("unknown",)),
    }


def test_enabled_curve_overrides_keep_distinct_immutable_requirements(tmp_path):
    with_override(tmp_path, "offline-ordering", {"offline": ["dedicated"]})
    path = tmp_path / "2026-10-C1/catalog.yaml"
    data = yaml.safe_load(path.read_text())
    data["enrollment"]["overrides"] = [
        {
            "id": pattern,
            "rule": "offline-ordering",
            "reason": "Pattern-specific ordering tolerance",
            "applies_to": {"load_pattern": [pattern]},
            "requirements": {"throughput_minimum_multiplier": value},
        }
        for pattern, value in [("max_throughput", 0.9), ("agentic_inference", 0.8)]
    ]
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    policy = load_policy(path.parent)
    curve = Context(
        id="curve",
        scope="curve",
        model_id="kimi-k3",
        members=tuple(
            Context(
                id=pattern, scope="point", offline="dedicated", load_pattern=pattern
            )
            for pattern in ["max_throughput", "agentic_inference"]
        ),
    )
    checks = [
        check
        for check in plan_checks(policy, [curve]).checks
        if check.rule.id == "offline-ordering"
    ]
    assert {
        (check.override, check.rule.requirements["throughput_minimum_multiplier"])
        for check in checks
    } == {("max_throughput", 0.9), ("agentic_inference", 0.8)}
    assert all(check.decision is Decision.READY for check in checks)
    with pytest.raises(TypeError):
        checks[0].rule.requirements["throughput_minimum_multiplier"] = 1
    original = next(rule for rule in policy.checks if rule.id == "offline-ordering")
    assert original.requirements["throughput_minimum_multiplier"] == 0.98


def test_overlapping_overrides_are_detected_on_each_curve_member(tmp_path):
    with_override(tmp_path, "offline-ordering", {"offline": ["dedicated"]})
    path = tmp_path / "2026-10-C1/catalog.yaml"
    data = yaml.safe_load(path.read_text())
    data["enrollment"]["overrides"].append(
        {**data["enrollment"]["overrides"][0], "id": "second"}
    )
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    curve = Context(
        id="curve",
        scope="curve",
        model_id="kimi-k3",
        members=(Context(id="point", scope="point", offline="dedicated"),),
    )
    with pytest.raises(ValueError, match="Overlapping overrides.*point"):
        plan_checks(load_policy(path.parent), [curve])
