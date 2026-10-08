# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecDecodeHead names and accepted declaration forms preserve approval checks."""

import json
import shutil
from dataclasses import replace

import pytest
import yaml
from pydantic import ValidationError

from inference_endpoint.validation import (
    bundled_policy_path,
    load_policy,
    validate_submission,
)
from inference_endpoint.validation.evidence.point_config import (
    PointConfig,
    SpecDecodeHeadDeclaration,
)
from inference_endpoint.validation.schemas.values_v1 import Catalogs

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "field", ["speculative_decoding", "spec_decode_head", "drafter"]
)
@pytest.mark.parametrize("approved", [True, False])
def test_spec_decode_head_declarations_select_and_run_approval(
    tmp_path, field, approved
):
    policy = load_policy(bundled_policy_path())
    head = policy.catalogs["approved_spec_decode_heads"][0]
    declaration = {
        "repository": head["repository"],
        "revision": head["revision"] if approved else "0" * 40,
    }
    point = tmp_path / "results" / "system" / head["model"] / "r16"
    point.mkdir(parents=True)
    (point / "point.yaml").write_text(
        json.dumps(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                field: declaration,
            }
        )
    )
    selected = replace(
        policy,
        checks=tuple(
            rule for rule in policy.checks if rule.id == "approved-spec-decode-head"
        ),
    )
    report = validate_submission(tmp_path, policy=selected)
    assert len(report.results) == 1
    assert report.passed is approved


@pytest.mark.parametrize("canonical", [None, {}, {"repository": "different"}])
def test_conflicting_spec_decode_head_aliases_are_rejected(canonical):
    with pytest.raises(ValidationError, match="input alias must agree"):
        PointConfig.model_validate(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "spec_decode_head": canonical,
                "drafter": {"repository": "head", "revision": "a" * 40},
            }
        )


def test_spec_decode_head_input_alias_serializes_with_canonical_name():
    parsed = PointConfig.model_validate(
        {
            "concurrency": 16,
            "runtime_settings": {"runtime": {}},
            "drafter": {"repository": "head", "revision": "a" * 40},
        }
    )
    payload = parsed.model_dump(mode="json", exclude_unset=True)
    assert payload["spec_decode_head"] == {"repository": "head", "revision": "a" * 40}
    assert "drafter" not in payload


@pytest.mark.parametrize("head_index", range(6))
@pytest.mark.parametrize(
    "target,passed", [("2026-10-C0", False), ("2026-10-C1", True), ("2026-11-C0", True)]
)
def test_published_weight_checksums_and_approval_lead_time(
    tmp_path, head_index, target, passed
):
    policy = load_policy(bundled_policy_path())
    head = policy.catalogs["approved_spec_decode_heads"][head_index]
    assert head["approved_cohort"] == "2026-09-C1"
    point = tmp_path / "results/system" / head["model"] / "r16"
    point.mkdir(parents=True)
    (point / "point.yaml").write_text(
        json.dumps(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "target_cohort": target,
                "speculative_decoding": {
                    "weight_checksum": "git-sha1:" + head["revision"]
                },
            }
        )
    )
    selected = replace(
        policy,
        checks=tuple(
            rule
            for rule in policy.checks
            if rule.id
            in {
                "approved-spec-decode-head",
                "spec-decode-head-approval-lead-time",
            }
        ),
    )
    report = validate_submission(tmp_path, policy=selected)
    assert len(report.results) == 2
    assert next(
        r for r in report.results if r.rule == "approved-spec-decode-head"
    ).passed
    assert report.passed is passed
    assert not report.warnings


@pytest.mark.parametrize(
    "identity",
    [
        {"weight_checksum": "sha256:" + "a" * 64},
        {"weight_checksum": "git-sha1:short"},
        {"weight_checksum": "git-sha1:" + "A" * 40},
        {"weight_checksum": {"revision": "a" * 40}},
        {"weight_checksum": "git-sha1:" + "a" * 40, "revision": "b" * 40},
    ],
)
def test_malformed_or_conflicting_weight_identity_is_rejected(identity):
    with pytest.raises(ValidationError):
        PointConfig.model_validate(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "spec_decode_head": identity,
            }
        )


@pytest.mark.parametrize(
    "identity,model",
    [
        ({"repository": "wrong"}, None),
        ({"revision": "0" * 40}, None),
        ({}, "deepseek-v4_1-flash"),
        (
            {
                "target_checksum": "git-sha1:" + "a" * 40,
                "configuration": {"exit_layer": 24},
            },
            None,
        ),
    ],
)
def test_weight_identity_must_match_model_and_supplied_repository(
    tmp_path, identity, model
):
    policy = load_policy(bundled_policy_path())
    head = policy.catalogs["approved_spec_decode_heads"][0]
    if "target_checksum" not in identity:
        checksum_revision = identity.get("revision", head["revision"])
        identity = {"weight_checksum": "git-sha1:" + checksum_revision, **identity}
    point = tmp_path / "results/system" / (model or head["model"]) / "r16"
    point.mkdir(parents=True)
    (point / "point.yaml").write_text(
        json.dumps(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "spec_decode_head": identity,
            }
        )
    )
    selected = replace(
        policy,
        checks=tuple(
            rule for rule in policy.checks if rule.id == "approved-spec-decode-head"
        ),
    )
    report = validate_submission(tmp_path, policy=selected)
    assert len(report.errors) == 1
    assert not report.passed


def test_weight_checksum_round_trip_preserves_string_contract():
    identity = {
        "weight_checksum": "git-sha1:" + "a" * 40,
        "repository": "head",
        "revision": "a" * 40,
    }
    point = PointConfig.model_validate(
        {
            "concurrency": 16,
            "runtime_settings": {"runtime": {}},
            "spec_decode_head": identity,
        }
    )
    assert point.spec_decode_head.weight_checksum.revision == "a" * 40
    assert (
        point.model_dump(mode="json", exclude_unset=True)["spec_decode_head"]
        == identity
    )


@pytest.mark.parametrize("checksum", ["git-sha1:" + "a" * 40, "sha256:" + "a" * 64])
@pytest.mark.parametrize(
    "change,passed",
    [
        (None, True),
        ("checksum", False),
        ("configuration", False),
        ("boolean_type", False),
        ("number_type", False),
        ("model", False),
        ("cohort", False),
    ],
)
def test_configuration_identity_approval_and_age(tmp_path, checksum, change, passed):
    # Synthetic policy entry exercises support without adding an official approval.
    policy_dir = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), policy_dir)
    catalog_path = policy_dir / "catalog.yaml"
    document = yaml.safe_load(catalog_path.read_text())
    model = document["catalogs"]["approved_spec_decode_heads"][0]["model"]
    configuration = {"exit_layer": 24, "layers": [1, 2], "settings": {"enabled": True}}
    document["catalogs"]["approved_spec_decode_heads"].append(
        {
            "model": model,
            "method": "self_speculative",
            "target_checksum": checksum,
            "configuration": configuration,
            "approved_cohort": "2026-09-C1",
        }
    )
    catalog_path.write_text(yaml.safe_dump(document))
    policy = load_policy(policy_dir)
    declaration = {"target_checksum": checksum, "configuration": configuration}
    if change == "checksum":
        declaration["target_checksum"] = checksum.replace("a", "b")
    if change == "configuration":
        declaration["configuration"] = {"exit_layer": 12}
    if change == "boolean_type":
        declaration["configuration"] = {**configuration, "settings": {"enabled": 1}}
    if change == "number_type":
        declaration["configuration"] = {**configuration, "layers": [1.0, 2]}
    if change == "model":
        model = "deepseek-v4_1-flash"
    point = tmp_path / "submission/results/system" / model / "r16"
    point.mkdir(parents=True)
    (point / "point.yaml").write_text(
        json.dumps(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "target_cohort": "2026-10-C0" if change == "cohort" else "2026-10-C1",
                "speculative_decoding": declaration,
            }
        )
    )
    selected = replace(
        policy,
        checks=tuple(
            rule
            for rule in policy.checks
            if rule.id
            in {
                "approved-spec-decode-head",
                "spec-decode-head-approval-lead-time",
            }
        ),
    )
    report = validate_submission(tmp_path / "submission", policy=selected)
    assert report.passed is passed
    assert not report.warnings
    if passed:
        assert len(report.results) == 2
        payload = PointConfig.model_validate(
            json.loads((point / "point.yaml").read_text())
        )
        assert (
            payload.model_dump(mode="json", exclude_unset=True)["speculative_decoding"]
            == declaration
        )


@pytest.mark.parametrize(
    "identity",
    [
        {"target_checksum": "git-sha1:" + "a" * 40},
        {"configuration": {"exit_layer": 24}},
        {"target_checksum": "git-sha1:" + "a" * 40, "configuration": {}},
        {"target_checksum": "invalid", "configuration": {"exit_layer": 24}},
        {
            "target_checksum": "git-sha1:" + "a" * 40,
            "configuration": {"exit_layer": 24},
            "weight_checksum": "git-sha1:" + "a" * 40,
        },
    ],
)
def test_incomplete_or_mixed_configuration_identity_is_rejected(identity):
    with pytest.raises(ValidationError):
        PointConfig.model_validate(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "speculative_decoding": identity,
            }
        )


def test_bundled_catalog_has_no_configuration_based_approvals():
    policy = load_policy(bundled_policy_path())
    assert all(
        "target_checksum" not in entry
        for entry in policy.catalogs["approved_spec_decode_heads"]
    )


@pytest.mark.parametrize(
    "change", ["missing_configuration", "empty_configuration", "mixed_identity"]
)
def test_invalid_configuration_catalog_identity_is_rejected(change):
    policy = load_policy(bundled_policy_path())
    document = policy.as_monolith()["catalogs"]
    entry = {
        "model": "kimi-k3",
        "method": "self_speculative",
        "target_checksum": "sha256:" + "a" * 64,
        "configuration": {"exit_layer": 24},
    }
    if change == "missing_configuration":
        del entry["configuration"]
    elif change == "empty_configuration":
        entry["configuration"] = {}
    else:
        entry.update(repository="head", revision="a" * 40)
    document["approved_spec_decode_heads"].append(entry)
    with pytest.raises(ValidationError):
        Catalogs.model_validate(document)


@pytest.mark.parametrize(
    "declared,approved,expected",
    [
        ({"enabled": True}, {"enabled": 1}, False),
        ({"enabled": False}, {"enabled": 0}, False),
        ({"exit_layer": 1}, {"exit_layer": 1.0}, False),
        (
            {"nested": {"layers": [True, {"temperature": False}]}},
            {"nested": {"layers": [1, {"temperature": 0}]}},
            False,
        ),
        ({"layers": [1, 2]}, {"layers": [2, 1]}, False),
        ({"layers": [1, 2]}, {"layers": [1]}, False),
        ({"enabled": True}, {"enabled": True, "extra": None}, False),
        ({"value": None}, {"value": "null"}, False),
        ({"value": float("inf")}, {"value": float("inf")}, False),
        ({"value": float("nan")}, {"value": float("nan")}, False),
        (
            {"layers": [1, 2.0, True, None, "value"]},
            {"layers": [1, 2.0, True, None, "value"]},
            True,
        ),
        (
            {"one": 1, "two": {"enabled": False}},
            {"two": {"enabled": False}, "one": 1},
            True,
        ),
    ],
)
def test_configuration_identity_preserves_json_types(declared, approved, expected):
    checksum = "git-sha1:" + "a" * 40
    head = SpecDecodeHeadDeclaration.model_validate(
        {"target_checksum": checksum, "configuration": declared}
    )
    assert head.matches(target_checksum=checksum, configuration=approved) is expected


def test_configuration_identity_aliases_cannot_hide_type_mismatch():
    checksum = "git-sha1:" + "a" * 40
    with pytest.raises(ValidationError, match="input alias must agree"):
        PointConfig.model_validate(
            {
                "concurrency": 16,
                "runtime_settings": {"runtime": {}},
                "spec_decode_head": {
                    "target_checksum": checksum,
                    "configuration": {"enabled": True},
                },
                "drafter": {
                    "target_checksum": checksum,
                    "configuration": {"enabled": 1},
                },
            }
        )
